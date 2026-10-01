"""CPU tests for the federated steering framework.

    python tests/test_fedsteer.py        (pytest also works if installed)

Uses a tiny random Llama with the real Llama-3.2 tokenizer (from the local HF cache).
"""

import copy
import os
import random
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from fedsteer.data import ChatFormatter, ClientQuantiles, build_clients, collate
from fedsteer.fed import FedConfig, FedSteerTrainer, load_snapshot_into
from fedsteer.lora import (SteerLinear, SteerLoraConfig, average_states, direction_products,
                           get_shared_state, inject_steer_lora, load_state, steer_layers)
from fedsteer.model import generate_at_alpha

TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER)
        _TOK.pad_token = _TOK.eos_token
    return _TOK


def tiny_model(seed=0, lora_seed=1234, rank=4, warp="none", offset=False):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tokenizer()), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    inject_steer_lora(model, SteerLoraConfig(rank_private=rank, rank_shared=rank, shared_seed=lora_seed,
                                             warp=warp, offset=offset))
    return model


def randomize_lora(model, seed=1, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, m in steer_layers(model):
            m.lora_B_p.copy_(torch.randn(m.lora_B_p.shape, generator=g) * scale)
            m.lora_B_d.copy_(torch.randn(m.lora_B_d.shape, generator=g) * scale)


def logits(model, ids, alpha):
    with model.steer_control.use_alpha(alpha):
        return model(input_ids=ids).logits


def toy_records(n_clients=2, n=24, seed=0):
    rng = random.Random(seed)
    recs = []
    for c in range(n_clients):
        for i in range(n):
            k = rng.randint(1 + c, 4 + 2 * c)
            recs.append({"client": f"c{c}", "split": "train", "prompt": f"Say something {i % 5}.",
                         "target": " ".join(["word"] * k), "score": k})
    return recs


# --------------------------------------------------------------------- layers

def test_injection_and_freezing():
    model = tiny_model()
    layers = list(steer_layers(model))
    assert len(layers) == 2 * 7
    trainable = {n.split(".")[-1] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A_p", "lora_B_p", "lora_B_d", "u"}, trainable


def test_zero_direction_is_alpha_invariant():
    model = tiny_model()
    ids = torch.randint(0, 1000, (2, 7))
    assert torch.allclose(logits(model, ids, 0.0), logits(model, ids, 1.0))


def test_layer_output_linear_in_alpha():
    model = tiny_model()
    randomize_lora(model)
    layer = next(m for _, m in steer_layers(model))
    x = torch.randn(3, 5, layer.base.in_features)
    outs = {}
    for a in (0.0, 0.3, 1.0):
        with model.steer_control.use_alpha(a):
            outs[a] = layer(x)
    assert torch.allclose(outs[0.3], outs[0.0] + 0.3 * (outs[1.0] - outs[0.0]), atol=1e-5)


def test_gain_scales_direction_only():
    model = tiny_model()
    randomize_lora(model)
    layer = next(m for _, m in steer_layers(model))
    x = torch.randn(2, 4, layer.base.in_features)
    with model.steer_control.use_alpha(0.0):
        y0 = layer(x)
    with model.steer_control.use_alpha(1.0):
        y1 = layer(x)
        with torch.no_grad():
            model.steer_control.u.fill_(torch.log(torch.tensor(2.0)))
        y2 = layer(x)
    assert torch.allclose(y2 - y0, 2 * (y1 - y0), atol=1e-5)


def test_per_example_alpha_matches_separate_calls():
    model = tiny_model()
    randomize_lora(model)
    ids = torch.randint(0, 1000, (2, 6))
    both = logits(model, ids, torch.tensor([0.0, 1.0]))
    assert torch.allclose(both[0], logits(model, ids[:1], 0.0)[0], atol=1e-5)
    assert torch.allclose(both[1], logits(model, ids[1:], 1.0)[0], atol=1e-5)


def test_shared_A_identical_across_clients_and_distinct_across_layers():
    m1, m2 = tiny_model(seed=0), tiny_model(seed=99)
    a1 = {n: m.lora_A_d for n, m in steer_layers(m1)}
    a2 = {n: m.lora_A_d for n, m in steer_layers(m2)}
    assert all(torch.equal(a1[n], a2[n]) for n in a1)
    names = list(a1)
    assert not torch.equal(a1[names[0]], a1[names[1]][: a1[names[0]].shape[0]]) or \
        a1[names[0]].shape != a1[names[1]].shape
    m3 = tiny_model(lora_seed=4321)
    assert not torch.equal(a1[names[0]], dict(steer_layers(m3))[names[0]].lora_A_d)


def test_fedavg_on_B_is_exact_product_average():
    model = tiny_model()
    states = []
    for s in range(3):
        randomize_lora(model, seed=s)
        states.append(get_shared_state(model))
    avg_products = average_states([direction_products(model, st) for st in states])
    products_of_avg = direction_products(model, average_states(states))
    assert all(torch.allclose(avg_products[k], products_of_avg[k], atol=1e-6) for k in avg_products)


def test_gradients_reach_only_lora_and_gain():
    model = tiny_model()
    randomize_lora(model)
    with torch.no_grad():
        model.steer_control.u.fill_(0.1)
    ids = torch.randint(0, 1000, (2, 6))
    with model.steer_control.use_alpha(torch.tensor([0.2, 0.9])):
        model(input_ids=ids, labels=ids).loss.backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, n
        else:
            assert p.grad is None, n


def test_gradient_checkpointing_matches():
    ids = torch.randint(0, 1000, (2, 6))
    grads = []
    for ckpt in (False, True):
        model = tiny_model()
        randomize_lora(model)
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        with model.steer_control.use_alpha(torch.tensor([0.3, 0.8])):
            model(input_ids=ids, labels=ids).loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.allclose(grads[0][k], grads[1][k], atol=1e-5) for k in grads[0])


def test_alpha_must_be_set():
    model = tiny_model()
    try:
        model(input_ids=torch.randint(0, 1000, (1, 4)))
    except RuntimeError as e:
        assert "alpha is not set" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


# ----------------------------------------------------------------------- data

def test_quantiles_and_alpha_labels():
    q = ClientQuantiles.fit([1, 2, 2, 3, 10])
    assert q.cdf(1) == 0.1 and q.cdf(2) == 0.4 and q.cdf(10) == 0.9
    assert q.quantile(0.0) == 1 and q.quantile(1.0) == 10 and q.quantile(0.5) == 2
    ex, qs = build_clients(toy_records(), ["c0", "c1"])
    for c in ("c0", "c1"):
        a = [e["alpha"] for e in ex[c]]
        assert 0 < min(a) and max(a) < 1
        # higher score -> higher alpha within a client
        pairs = sorted((e["score"], e["alpha"]) for e in ex[c])
        assert all(p[1] <= q[1] + 1e-12 for p, q in zip(pairs, pairs[1:]))
    # budgets subsample but keep the full-data quantile scale
    ex_small, qs_small = build_clients(toy_records(), ["c0", "c1"], max_train=5)
    assert len(ex_small["c0"]) == 5 and qs_small["c0"].sorted_scores == qs["c0"].sorted_scores


def test_formatter_masks_prompt():
    fmt = ChatFormatter(tokenizer())
    enc = fmt.encode("Summarize this.", "Short answer here.")
    n_prompt = len(fmt.prompt_ids("Summarize this."))
    assert enc["input_ids"][:n_prompt] == fmt.prompt_ids("Summarize this.")
    assert all(l == -100 for l in enc["labels"][:n_prompt])
    target = tokenizer().decode([t for t in enc["labels"] if t != -100])
    assert target.startswith("Short answer here.") and "<|eot_id|>" in target
    batch = collate([dict(enc, alpha=0.3), dict(fmt.encode("Hi", "Yes"), alpha=0.9)], tokenizer().pad_token_id)
    assert batch["alpha"].tolist() == [0.30000001192092896, 0.8999999761581421]


# ----------------------------------------------------------------------- warps

WARPS = ("none", "kumaraswamy", "kumaraswamy_mix", "step")


def _randomize_warp(w, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in w.named_parameters():
            if n == "w_raw":
                p.copy_(torch.rand((), generator=g))                       # in [0, 1]
            else:
                p.copy_(torch.randn((), generator=g) * 1.5)                # includes steep / extreme shapes


def test_warps_identity_at_init():
    from fedsteer.warp import make_warp
    x = torch.linspace(0, 1, 101)
    for kind in WARPS:
        assert torch.allclose(make_warp(kind)(x), x, atol=1e-6), kind


def test_warps_monotone_with_exact_endpoints():
    from fedsteer.warp import make_warp
    x = torch.linspace(0, 1, 2001)
    for kind in WARPS[1:]:
        for seed in range(20):
            w = make_warp(kind)
            _randomize_warp(w, seed)
            h = w(x)
            assert torch.all(h[1:] - h[:-1] >= -1e-6), (kind, seed)
            assert abs(h[0].item()) < 1e-6 and abs(h[-1].item() - 1) < 1e-5, (kind, seed, h[0], h[-1])
            assert torch.all((h >= -1e-6) & (h <= 1 + 1e-6)), kind


def test_warp_gradients_finite_and_correct():
    from fedsteer.warp import make_warp
    for kind in WARPS[1:]:
        w = make_warp(kind)
        _randomize_warp(w, 3)
        with torch.no_grad():                   # keep w strictly inside (0, 1) for the step warp
            if hasattr(w, "w_raw"):
                w.w_raw.fill_(0.6)
        x = torch.tensor([0.0, 0.3, 0.7, 1.0])
        w(x).sum().backward()
        for n, p in w.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (kind, n, p.grad)
        # finite differences at an interior alpha
        for n, p in w.named_parameters():
            w.zero_grad()
            w(torch.tensor([0.37])).sum().backward()
            g = p.grad.item()
            eps = 1e-3
            with torch.no_grad():
                p += eps
                up = w(torch.tensor([0.37])).item()
                p -= 2 * eps
                dn = w(torch.tensor([0.37])).item()
                p += eps
            fd = (up - dn) / (2 * eps)
            assert abs(g - fd) < 1e-3 + 2e-2 * abs(fd), (kind, n, g, fd)


def test_warp_identity_model_matches_linear_model():
    ids = torch.randint(0, 1000, (2, 6))
    ref = tiny_model(warp="none")
    randomize_lora(ref)
    for kind in WARPS[1:]:
        m = tiny_model(warp=kind)
        randomize_lora(m)
        for a in (0.0, 0.4, 1.0):
            assert torch.allclose(logits(m, ids, a), logits(ref, ids, a), atol=1e-5), (kind, a)


def test_warp_changes_coefficient_not_endpoints():
    m = tiny_model(warp="kumaraswamy_mix")
    randomize_lora(m)
    ids = torch.randint(0, 1000, (1, 6))
    before = {a: logits(m, ids, a) for a in (0.0, 0.5, 1.0)}
    _randomize_warp(m.steer_control.warp, 7)
    after = {a: logits(m, ids, a) for a in (0.0, 0.5, 1.0)}
    assert torch.allclose(before[0.0], after[0.0], atol=1e-5)       # h(0) = 0
    assert torch.allclose(before[1.0], after[1.0], atol=1e-4)       # h(1) = 1
    assert not torch.allclose(before[0.5], after[0.5], atol=1e-5)   # interior alphas move


def test_fed_training_with_warp_is_private_and_warms_up():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, warp="kumaraswamy_mix", rounds=3, warp_warmup_rounds=1, lr_warp=0.2)
        tr.fit()
        wkeys = [k for k in tr.clients["c0"]["gain"] if k.startswith("steer_control.warp.")]
        assert len(wkeys) == 3
        assert not any(k.startswith("steer_control") for k in tr.server)                 # never aggregated
        init = {"steer_control.warp.log_p": 0.0, "steer_control.warp.log_q": 0.0, "steer_control.warp.w_logit": 0.0}
        for c in ("c0", "c1"):
            assert any(abs(tr.clients[c]["gain"][k].item() - init[k]) > 1e-6 for k in wkeys), c
        assert any(tr.clients["c0"]["gain"][k] != tr.clients["c1"]["gain"][k] for k in wkeys)
        assert "warp" in tr.history[-1]["clients"]["c0"] and "warp_penalty" in tr.history[-1]["clients"]["c0"]
        assert "warp_penalty" not in tr.history[0]["clients"]["c0"]                     # round 0: warm-up
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, warp="kumaraswamy_mix", rounds=2, warp_warmup_rounds=5, lr_warp=0.2)
        tr.fit()
        assert all(tr.clients[c]["gain"][k].item() == 0.0 for c in ("c0", "c1")
                   for k in tr.clients[c]["gain"] if k.startswith("steer_control.warp."))


def test_isotonic_remap():
    from fedsteer.calibrate import fit_remap, invert_monotone, isotonic_increasing
    assert isotonic_increasing([1, 3, 2, 4]).tolist() == [1, 2.5, 2.5, 4]
    assert isotonic_increasing([3, 2, 1]).tolist() == [2, 2, 2]
    grid = [0, 0.5, 1]
    assert invert_monotone(grid, [0.2, 0.4, 0.8], 0.6) == 0.75
    assert invert_monotone(grid, [0.2, 0.4, 0.8], 0.1) == 0.0      # below the achievable range: clip
    assert invert_monotone(grid, [0.2, 0.4, 0.8], 0.9) == 1.0
    r = fit_remap(grid, [0.3, 0.2, 0.9], [0.5])                    # non-monotone input is pooled first
    assert r["fitted"] == [0.25, 0.25, 0.9] and 0.5 < r["mapped_alpha"]["0.5"] < 1.0


def test_eval_snapshot_with_warp_and_posthoc_remap():
    import importlib.util
    from types import SimpleNamespace
    from fedsteer.metrics import SCORERS
    spec = importlib.util.spec_from_file_location(
        "eval_direction", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_direction.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, warp="kumaraswamy_mix", rounds=2, warp_warmup_rounds=1, lr_warp=0.2)
        tr.fit()
        _, qs = build_clients(toy_records(), ["c0", "c1"])
        recs = {c: [r for r in toy_records() if r["client"] == c][:3] for c in ("c0", "c1")}
        args = SimpleNamespace(max_prompts=3, max_new_tokens=4, batch_size=4, dev_loss=True, shared=None, split="test",
                               remap_prompts=3, remap_grid=3)
        snap = os.path.join(d, "snapshots", "round_0002.pt")
        model = tiny_model(warp="kumaraswamy_mix")
        fmt = ChatFormatter(tokenizer())
        res = ev.evaluate_snapshot(model, fmt, snap, None, recs, ["c0", "c1"], qs, [0.0, 0.5, 1.0],
                                   SCORERS["words"], args, remap_by_client=recs)
        r = res["clients"]["c0"]
        assert "warp" in r and r["warp"]["kind"] == "kumaraswamy_mix"
        assert set(r["remap"]["mapped_alpha"]) == {"0.0", "0.5", "1.0"}
        assert all(0.0 <= a <= 1.0 for a in r["remap"]["mapped_alpha"].values())
        assert [smp["alpha"] for smp in res["samples"]["c0"]] == [0.0, 0.5, 1.0]   # metrics use target alphas
        assert "loss" in res["summary"]


# ------------------------------------------------- offset, adapter modes, global alpha

def test_offset_shifts_the_alpha_zero_point():
    m = tiny_model(offset=True)
    randomize_lora(m)
    layer = next(mm for _, mm in steer_layers(m))
    x = torch.randn(2, 3, layer.base.in_features)
    with m.steer_control.use_alpha(0.0):
        y_off0 = layer(x)
    with torch.no_grad():
        m.steer_control.o.fill_(0.5)
    with m.steer_control.use_alpha(0.0):
        y_half = layer(x)
    with m.steer_control.use_alpha(0.5):           # o = 0.5 at alpha 0 == o = 0 at alpha 0.5 (s = 1)
        with torch.no_grad():
            m.steer_control.o.fill_(0.0)
        y_ref = layer(x)
    assert torch.allclose(y_half, y_ref, atol=1e-5) and not torch.allclose(y_half, y_off0)
    with torch.no_grad():
        m.steer_control.o.fill_(10.0)              # clamped to offset_max = 2
    assert abs(m.steer_control.offset().item() - 2.0) < 1e-6


def test_offset_is_private_and_trained_after_warmup():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, offset=True, rounds=3)
        tr.fit()
        o = {c: tr.clients[c]["gain"]["steer_control.o"].item() for c in ("c0", "c1")}
        assert all(v != 0.0 for v in o.values()) and o["c0"] != o["c1"]
        assert not any(k.startswith("steer_control") for k in tr.server)
        assert "offset" in tr.history[-1]["clients"]["c0"]


def test_adapter_none_trains_only_direction_and_scalars():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, adapter="none", rounds=2)
        tr.fit()
        assert tr.clients["c0"]["private"] == {}
        assert all(p.abs().sum() == 0 for n, p in tr.model.named_parameters() if n.endswith("lora_B_p"))
        assert any(v.abs().sum() > 0 for v in tr.server.values())
        assert not any(k.endswith("lora_B_p") for k in tr.server)


def test_adapter_shared_equals_share_private_alias():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = _trainer("fedavg", d1, adapter="shared", rounds=2)
        a.fit()
        b = _trainer("fedavg", d2, share_private=True, rounds=2)
        b.fit()
        assert any(k.endswith("lora_B_p") for k in a.server)
        assert all(torch.allclose(a.server[k], b.server[k]) for k in a.server)


def test_global_alpha_has_shared_meaning():
    from fedsteer.data import GlobalQuantiles, alpha_reference_from_json, client_support, fit_local_quantiles
    recs = toy_records()          # c1 has systematically larger scores than c0
    ex_l, _ = build_clients(recs, ["c0", "c1"], alpha_mode="local")
    ex_g, refs = build_clients(recs, ["c0", "c1"], alpha_mode="global")
    ref = refs["c0"]
    assert refs["c1"] is ref
    by_score = {}
    for c in ("c0", "c1"):
        for e in ex_g[c]:
            by_score.setdefault(e["score"], set()).add(round(e["alpha"], 9))
    assert all(len(v) == 1 for v in by_score.values())       # same score -> same alpha on every client
    shared = set(e["score"] for e in ex_g["c0"]) & set(e["score"] for e in ex_g["c1"])
    s = next(iter(shared))
    la = {c: next(e["alpha"] for e in ex_l[c] if e["score"] == s) for c in ("c0", "c1")}
    assert la["c0"] > la["c1"]                               # local alpha differs for the same score
    mean_a = {c: np.mean([e["alpha"] for e in ex_g[c]]) for c in ("c0", "c1")}
    assert mean_a["c0"] < 0.5 < mean_a["c1"]                  # clients occupy different parts of the axis
    assert abs((mean_a["c0"] + mean_a["c1"]) / 2 - 0.5) < 0.02  # equal-weight mixture is uniform overall
    local = fit_local_quantiles(recs, ["c0", "c1"])
    lo0, hi0 = client_support(local["c0"], ref)
    lo1, hi1 = client_support(local["c1"], ref)
    assert lo0 < lo1 and hi0 < hi1
    ref2 = alpha_reference_from_json(ref.to_json())
    assert all(abs(ref2.cdf(x) - ref.cdf(x)) < 1e-12 for x in range(0, 12))
    assert ref.quantile(ref.cdf(5.0)) <= 5.0


def test_support_split_metrics():
    from fedsteer.metrics import metrics_for_client
    q = ClientQuantiles.fit(list(range(1, 101)))
    alphas = [0.1, 0.5, 0.9]
    # client whose data covers [0.4, 1.0]; its outputs reach 0.2 when asked for 0.1
    scores = [[q.quantile(0.2), q.quantile(0.5), q.quantile(0.9)]] * 4
    m = metrics_for_client(scores, alphas, q, support=(0.4, 1.0))
    assert m["support"] == [0.4, 1.0]
    assert m["pct_err_out_support"] > 0.05 and m["pct_err_in_support"] < 0.02
    assert m["reach_rate"] == 1.0                             # 0.2 < 0.4: left its own range downward
    m2 = metrics_for_client([[q.quantile(0.45), q.quantile(0.5), q.quantile(0.9)]] * 4, alphas, q,
                            support=(0.4, 1.0))
    assert m2["reach_rate"] == 0.0


def test_run_stamps_and_provenance():
    import json as _json
    from fedsteer.runinfo import make_stamp, record_run_info
    os.environ["FEDSTEER_STAMP"] = "20260101-000000_j123"
    try:
        assert make_stamp() == "20260101-000000_j123"
    finally:
        del os.environ["FEDSTEER_STAMP"]
    old = os.environ.pop("SLURM_JOB_ID", None)
    try:
        assert "_j" not in make_stamp() and len(make_stamp()) == 15
        os.environ["SLURM_JOB_ID"] = "999"
        assert make_stamp().endswith("_j999")
    finally:
        os.environ.pop("SLURM_JOB_ID", None)
        if old is not None:
            os.environ["SLURM_JOB_ID"] = old
    with tempfile.TemporaryDirectory() as d:
        record_run_info(d, "created", config="c.yaml")
        record_run_info(d, "resumed", overrides=["fed.rounds=5"])
        info = _json.load(open(os.path.join(d, "run_info.json")))
        assert [e["event"] for e in info["events"]] == ["created", "resumed"]
        assert "commit" in info["events"][0]["git"]


# ------------------------------------------------------- regularizers, full eval, quality

def test_decorrelation_matches_bruteforce_cosine():
    from fedsteer.regularize import decorrelation
    m = tiny_model()
    randomize_lora(m)
    layers = [mm for _, mm in steer_layers(m)]
    brute = []
    for l in layers:
        P = l.lora_B_p @ l.lora_A_p
        D = l.lora_B_d @ l.lora_A_d
        brute.append(((P * D).sum() ** 2 / ((P * P).sum() * (D * D).sum())).item())
    assert abs(decorrelation(layers).item() - np.mean(brute)) < 1e-5
    with torch.no_grad():                       # make P exactly parallel to D in every layer
        for l in layers:
            r = min(l.lora_A_p.shape[0], l.lora_A_d.shape[0])
            l.lora_A_p[:r].copy_(l.lora_A_d[:r]); l.lora_A_p[r:].zero_()
            l.lora_B_p[:, :r].copy_(2 * l.lora_B_d[:, :r]); l.lora_B_p[:, r:].zero_()
    assert abs(decorrelation(layers).item() - 1.0) < 1e-4


def test_fedprox_zero_at_anchor():
    from fedsteer.regularize import fedprox
    m = tiny_model()
    randomize_lora(m)
    layers = [mm for _, mm in steer_layers(m)]
    anchors = [l.lora_B_d.detach().clone() for l in layers]
    assert fedprox(layers, anchors).item() == 0.0
    with torch.no_grad():
        layers[0].lora_B_d.add_(0.1)
    expected = 0.5 * (0.1 ** 2) * layers[0].lora_B_d.numel()
    assert abs(fedprox(layers, anchors).item() - expected) < 1e-5


def test_training_with_all_regularizers_logs_terms():
    from fedsteer.regularize import RegConfig
    reg = RegConfig(private_wd=0.05, shared_wd=0.01, decorr=0.1, fedprox_mu=0.1, gain_l2=0.01, offset_l2=0.01)
    with tempfile.TemporaryDirectory() as d:
        model = tiny_model(offset=True)
        ex, _ = build_clients(toy_records(), ["c0", "c1"])
        cfg = FedConfig(rounds=3, local_steps=2, batch_size=2, bf16=False, warmup_steps=1, gain_warmup_rounds=1,
                        save_every=1)
        tr = FedSteerTrainer(model, ChatFormatter(tokenizer()), ex, cfg, d, reg=reg)
        wd = {g["name"]: g["weight_decay"] for g in tr.opt.param_groups}
        assert wd["private"] == 0.05 and wd["shared"] == 0.01
        tr.fit()
        r0, r2 = tr.history[0]["clients"]["c0"]["reg"], tr.history[2]["clients"]["c0"]["reg"]
        assert set(r0) == {"decorr", "fedprox"}                       # gain/offset priors wait for warm-up
        assert set(r2) == {"decorr", "fedprox", "gain_l2", "offset_l2"}
        assert all(np.isfinite(v) for v in r2.values())


def test_monitor_full_eval_writes_selectable_files():
    import json as _json
    import subprocess
    from fedsteer.monitor import MonitorConfig, make_monitor
    recs = toy_records()
    for i, r in enumerate(recs):
        r["url"] = f"u{i}"
        if i % 4 == 0:
            r["split"] = "dev"
    ex, qs = build_clients(recs, ["c0", "c1"])
    fmt = ChatFormatter(tokenizer())
    with tempfile.TemporaryDirectory() as d:
        mon = make_monitor(recs, ["c0", "c1"], qs, fmt,
                           MonitorConfig(loss_every=1, steer_every=0, full_every=2, full_prompts=3,
                                         full_alphas=[0.0, 1.0], full_max_new_tokens=4, full_batch_size=4,
                                         scorer="words"), out_dir=d)
        cfg = FedConfig(rounds=2, local_steps=1, batch_size=2, bf16=False, warmup_steps=1, save_every=2)
        tr = FedSteerTrainer(tiny_model(), fmt, ex, cfg, d, eval_fn=mon)
        tr.fit()
        files = os.listdir(os.path.join(d, "evals"))
        assert len(files) == 1 and files[0].startswith("eval_round_0002_dev__")
        ev = _json.load(open(os.path.join(d, "evals", files[0])))
        c0 = ev["clients"]["c0"]
        assert len(c0["outputs"]) == 3 and len(c0["outputs"][0]) == 2 and len(c0["record_ids"]) == 3
        assert "loss" in c0 and "loss" in ev["summary"] and ev["snapshot"].endswith("snapshots/round_0002.pt")
        assert os.path.exists(ev["snapshot"])
        assert "full" in tr.history[1]["eval"] and "full" not in tr.history[0]["eval"]
        py = sys.executable
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        best = subprocess.run([py, os.path.join(root, "scripts", "summarize_sweep.py"), "--run", d, "--print_best"],
                              capture_output=True, text=True).stdout.strip()
        assert best.endswith("round_0002.pt"), best


def test_quality_text_utils():
    from fedsteer.quality import chunk_words, split_sentences, surface
    assert split_sentences("A b c. U.S. growth rose. \"Yes,\" she said. End!") == \
        ["A b c.", "U.S. growth rose.", '"Yes," she said.', "End!"]
    art = " ".join(f"Sentence number {i} has five words." for i in range(200))
    chunks = chunk_words(art, 350)
    assert all(len(c.split()) <= 350 for c in chunks) and " ".join(chunks) == " ".join(split_sentences(art))
    s = surface("the cat sat the cat sat the cat sat")
    assert s["length"] == 9 and s["rep3"] > 0.5 and s["empty"] == 0.0
    assert surface("")["empty"] == 1.0


def test_near_tie_metrics():
    from fedsteer.metrics import near_identical, normalize_output, text_tie_metrics
    a = ("The lawyers of the woman accusing high-powered attorney Sanford Rubenstein of rape have penned a letter "
         "to Manhattan District Attorney Cyrus Vance, pleading with his office to make an arrest. \u201cHer...")
    b = a[:-3] + " com..."
    assert a != b and near_identical(a, b)                                  # cut-off at a different point
    assert normalize_output("One two three \u2026") == "One two"            # ellipsis + cut word dropped
    one_word = a.replace("have penned", "had penned")
    assert near_identical(a, one_word)                                      # 1 token of ~40 differs
    longer = ("The council voted to approve the budget on Tuesday. Mayor Smith said work starts in May "
              "and will finish by the end of the year.")
    assert not near_identical("The council voted to approve the budget on Tuesday.", longer)   # real change
    assert not near_identical("Council rejects budget.", "Mayor praises schools.")
    outputs = [[a, b, longer], ["x y z", "x y z", "x y z"]]
    scores = [[30.0, 31.0, 40.0], [5.0, 5.0, 5.0]]
    m = text_tie_metrics(outputs, scores, [0.0, 0.5, 1.0])
    assert m["text_tie_rate"] == 0.5                  # article 2's two pairs are exact ties
    assert m["near_tie_rate"] == 0.75                 # + article 1's first pair
    assert m["near_no_effect_rate"] == 0.5 and m["endpoint_near_tie_rate"] == 0.5
    assert m["adjacent_increase_rate_nt"] == 0.25     # only a->longer... (b -> longer) counts as an increase
    # concordance: article 1 pairs (a,b) near-tie 0.5, (a,long) 1, (b,long) 1; article 2 all ties 0.5
    assert abs(m["concordance_nt"] - (0.5 + 1 + 1 + 0.5 * 3) / 6) < 1e-9


def test_shared_calibration_is_one_mapping_for_all_clients():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, calibration="shared", warp="kumaraswamy_mix", offset=True, rounds=3,
                      warp_warmup_rounds=1, lr_warp=0.2)
        tr.fit()
        ctl_keys = [k for k in tr.server if k.startswith("steer_control.")]
        assert set(ctl_keys) == {"steer_control.u", "steer_control.o", "steer_control.warp.log_p",
                                 "steer_control.warp.log_q", "steer_control.warp.w_logit"}
        assert any(tr.server[k].abs().item() > 0 for k in ctl_keys)           # it was trained
        # clients never store their own calibration in shared mode
        init = {k: v for k, v in tr.clients["c0"]["gain"].items()}
        assert all(v.abs().item() == 0 for v in init.values())
        # every client sees the same calibration when loaded
        vals = []
        for c in ("c0", "c1"):
            tr.load_client(c)
            vals.append({n: p.detach().clone() for n, p in tr.model.named_parameters() if n.startswith("steer_control.")})
        assert all(torch.equal(vals[0][k], vals[1][k]) for k in vals[0])
        # snapshots restore the shared calibration for any client
        snap = torch.load(os.path.join(d, "snapshots", "round_0003.pt"), weights_only=False)
        m = tiny_model(warp="kumaraswamy_mix", offset=True)
        load_snapshot_into(m, snap, "c1")
        got = {n: p for n, p in m.named_parameters() if n.startswith("steer_control.")}
        assert all(torch.allclose(got[k], tr.server[k]) for k in ctl_keys)


def test_shared_calibration_refused_in_local_mode():
    with tempfile.TemporaryDirectory() as d:
        try:
            _trainer("local", d, calibration="shared")
        except ValueError as e:
            assert "local mode" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_no_offset_model_has_no_offset_parameter():
    m = tiny_model(offset=False)
    assert m.steer_control.o is None
    assert not any(n == "steer_control.o" for n, _ in m.named_parameters())


# ----------------------------------------------------------------------- baselines

def test_b1_prompt_construction_and_shots():
    from fedsteer.baselines import level_instruction, pick_shots, prompt_with_level
    assert "Target extractiveness: 25 on a 0-100 scale" in level_instruction(0.25)
    pool = [{"url": f"u{i}", "alpha": a, "article": f"art {i} " * 300, "target": f"sum {i}"}
            for i, a in enumerate([0.1, 0.4, 0.42, 0.9])]
    shots = pick_shots(pool, 0.41, 2, exclude_url="u2")
    assert [s["url"] for s in shots] == ["u1", "u0"] or [s["url"] for s in shots] == ["u1", "u3"]
    assert "u2" not in [s["url"] for s in shots]                    # never the article being summarized
    p = prompt_with_level({"article": "THE ARTICLE"}, 0.41, shots)
    assert p.startswith("Write a short summary") and p.rstrip().endswith("THE ARTICLE")
    assert "Target extractiveness: 41" in p and p.count("Summary: sum") == 2
    assert len(shots[0]["article"].split()) > 150 and "..." in p   # example articles are shortened
    assert prompt_with_level({"article": "X"}, 0.0, []).count("Article:") == 1


def test_b4_activation_hook_and_vector():
    from fedsteer.baselines import add_to_residual, caa_vector
    m = tiny_model()
    randomize_lora(m)
    ids = torch.randint(0, 1000, (1, 6))
    base = logits(m, ids, 0.0)
    v = torch.randn(m.config.hidden_size)
    with add_to_residual(m, 0, v):
        steered = logits(m, ids, 0.0)
    assert not torch.allclose(base, steered)
    assert torch.allclose(logits(m, ids, 0.0), base)                # hook removed afterwards
    with add_to_residual(m, 0, torch.zeros_like(v)):
        assert torch.allclose(logits(m, ids, 0.0), base, atol=1e-6)
    fmt = ChatFormatter(tokenizer())
    pool = [{"prompt": f"Say {i}.", "target": "word " * (i + 1), "alpha": i / 19} for i in range(20)]
    vec, hnorm = caa_vector(m, fmt, pool, layer=0, n_per_side=4)
    assert vec.shape == (m.config.hidden_size,) and torch.isfinite(vec).all() and hnorm > 0


def test_b3_merge_equals_average_of_local_directions():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("local", d, rounds=2)
        tr.fit()
        snap = os.path.join(d, "snapshots", "round_0002.pt")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, os.path.join(root, "scripts", "merge_local_directions.py"),
                              "--snapshot", snap], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        merged = torch.load(os.path.join(d, "merged_round_0002.pt"))
        k = next(iter(merged))
        s0, s1 = tr.clients["c0"]["shared_local"][k], tr.clients["c1"]["shared_local"][k]
        assert torch.allclose(merged[k], (s0 + s1) / 2)
        assert all(key.endswith("lora_B_d") for key in merged)


# ------------------------------------------------------------------ extractive

def test_fragment_stats_known_cases():
    from fedsteer.extractive import fragment_stats, fragments, publication
    art = "The cat sat on the mat . A dog barked loudly ."
    # verbatim copy of an 8-token span -> one fragment, density = 8
    st = fragment_stats("the cat sat on the mat .", art)
    assert st["coverage"] == 1.0 and abs(st["density"] - 7.0) < 1e-9  # 7 tokens: the cat sat on the mat .
    # two separate copied spans (3 and 2 tokens) plus 1 novel token
    st = fragment_stats("cat sat on dog barked wow", art)
    assert abs(st["coverage"] - 5 / 6) < 1e-9 and abs(st["density"] - (9 + 4) / 6) < 1e-9
    # fully novel
    assert fragment_stats("completely new words", art)["coverage"] == 0.0
    # greedy longest match: 'the' occurs twice; the longer continuation must win
    assert fragments("the mat .".split(), "the cat the mat .".split()) == [3]
    # whitespace variants do not break fragments
    assert fragment_stats("the\xa0cat  sat", art)["density"] == 3.0
    assert publication("http://www.nytimes.com/2013/a.html") == "nytimes.com"
    assert publication("http://blogs.wsj.com/x") == "wsj.com"
    assert publication("https://www.dailymail.co.uk/news/x") == "dailymail.co.uk"
    assert publication("http://www.9news.com.au/x") == "9news.com.au"


# --------------------------------------------------------------------- federation

def _trainer(mode, out_dir, **kw):
    model = tiny_model(warp=kw.pop("warp", "none"), offset=kw.pop("offset", False))
    ex, _ = build_clients(toy_records(), ["c0", "c1"])
    cfg = FedConfig(mode=mode, rounds=kw.pop("rounds", 3), local_steps=3, batch_size=4, bf16=False,
                    warmup_steps=1, gain_warmup_rounds=1, save_every=1, lr_private=5e-3, lr_shared=5e-3, **kw)
    return FedSteerTrainer(model, ChatFormatter(tokenizer()), ex, cfg, out_dir)


def test_fedavg_trains_and_clients_stay_private():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d)
        tr.fit()
        assert tr.round == 3 and len(tr.history) == 3
        assert all(torch.isfinite(torch.tensor(v["loss"])) for h in tr.history for v in h["clients"].values())
        assert any(v.abs().sum() > 0 for v in tr.server.values())            # direction learned
        p0, p1 = tr.clients["c0"]["private"], tr.clients["c1"]["private"]
        assert any(not torch.equal(p0[k], p1[k]) for k in p0)                # private adapters differ
        g0 = tr.clients["c0"]["gain"]["steer_control.u"]
        assert g0.item() != 0.0                                              # gain trained after warmup
        assert os.path.exists(os.path.join(d, "snapshots", "round_0003.pt"))


def test_fix_gain_ablation():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, fix_gain=True, rounds=2)
        tr.fit()
        assert all(st["gain"]["steer_control.u"].item() == 0.0 for st in tr.clients.values())


def test_share_private_ablation_aggregates_adapter():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, share_private=True, rounds=2)
        tr.fit()
        assert any(k.endswith("lora_B_p") for k in tr.server)


def test_local_mode_keeps_separate_directions_and_writes_merge():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("local", d, rounds=2)
        tr.fit()
        s0, s1 = tr.clients["c0"]["shared_local"], tr.clients["c1"]["shared_local"]
        assert any(not torch.equal(s0[k], s1[k]) for k in s0)
        merged = torch.load(os.path.join(d, "merged_shared.pt"))
        k = next(iter(merged))
        assert torch.allclose(merged[k], (s0[k] + s1[k]) / 2)


def test_resume_is_exact():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        full = _trainer("fedavg", d1, rounds=4)
        full.fit()
        part = _trainer("fedavg", d2, rounds=2)
        part.fit()
        resumed = _trainer("fedavg", d2, rounds=4)
        resumed.fit()
        for k in full.server:
            assert torch.allclose(full.server[k], resumed.server[k], atol=1e-6), k
        for c in full.clients:
            for k, v in full.clients[c]["private"].items():
                assert torch.allclose(v, resumed.clients[c]["private"][k], atol=1e-6), (c, k)


def test_metrics_edge_cases():
    from fedsteer.metrics import constant_output_pct_err, metrics_for_client
    q = ClientQuantiles.fit(list(range(1, 101)))
    alphas = [0.1, 0.5, 0.9]
    perfect = metrics_for_client([[q.quantile(a) for a in alphas]] * 3, alphas, q)
    assert perfect["concordance"] == 1.0 and perfect["pct_calib_err"] < 0.02 and perfect["spearman"] > 0.99
    flat = metrics_for_client([[50.0] * 3] * 3, alphas, q)
    # constant rows count as rho = 0 (not skipped), and are flagged as no-effect
    assert flat["spearman"] == 0.0 and flat["no_effect_rate"] == 1.0 and flat["concordance"] == 0.5
    assert abs(flat["pct_calib_err"] - constant_output_pct_err(alphas)) < 0.02


def test_monitor_logs_heldout_loss_and_steering():
    from fedsteer.monitor import MonitorConfig, make_monitor
    recs = toy_records()
    for i, r in enumerate(recs):
        if i % 4 == 0:
            r["split"] = "dev"
    ex, qs = build_clients(recs, ["c0", "c1"])
    fmt = ChatFormatter(tokenizer())
    mon = make_monitor(recs, ["c0", "c1"], qs, fmt,
                       MonitorConfig(loss_every=1, steer_every=2, steer_prompts=2, scorer="words",
                                     max_new_tokens=4, batch_size=4))
    with tempfile.TemporaryDirectory() as d:
        model = tiny_model()
        cfg = FedConfig(rounds=2, local_steps=1, batch_size=2, bf16=False, warmup_steps=1, save_every=1)
        tr = FedSteerTrainer(model, fmt, ex, cfg, d, eval_fn=mon)
        tr.fit()
        e1, e2 = tr.history[0]["eval"], tr.history[1]["eval"]
        assert "loss_mean" in e1 and "steer_summary" not in e1          # round 1: loss only
        assert "loss_mean" in e2 and "steer_summary" in e2              # round 2: loss + steering
        assert all(torch.isfinite(torch.tensor(v["loss"])) for v in e2["clients"].values())
    # the monitor must not change the training trajectory: rerun without it and compare
    with tempfile.TemporaryDirectory() as d:
        tr2 = FedSteerTrainer(tiny_model(), fmt, ex, cfg, d)
        tr2.fit()
    for k in tr.server:
        assert torch.allclose(tr.server[k], tr2.server[k], atol=1e-6), k


def test_snapshot_eval_and_generation():
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, rounds=2)
        tr.fit()
        snap = torch.load(os.path.join(d, "snapshots", "round_0002.pt"), weights_only=False)
        model = tiny_model(seed=5)
        load_snapshot_into(model, snap, "c1")
        tr.load_client("c1")
        # base weights differ (seed=5), so compare the loaded adapter/gain/direction state
        ref = {n: p for n, p in tr.model.named_parameters() if p.requires_grad}
        got = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert all(torch.allclose(ref[n], got[n]) for n in ref)
        fmt = ChatFormatter(tokenizer())
        outs = generate_at_alpha(model, fmt, ["Say something 1.", "Say something 2."], [0.0, 1.0],
                                 max_new_tokens=5, bf16=False)
        assert len(outs) == 2 and all(len(o) == 1 for o in outs)
        outs = generate_at_alpha(model, fmt, ["Hi."], 0.5, max_new_tokens=4, do_sample=True,
                                 num_return_sequences=3, bf16=False)
        assert len(outs[0]) == 3


def test_generation_with_zero_alpha_ignores_direction():
    model = tiny_model()
    randomize_lora(model)
    fmt = ChatFormatter(tokenizer())
    a = generate_at_alpha(model, fmt, ["Hello there."], 0.0, max_new_tokens=6, bf16=False)
    with torch.no_grad():
        for _, m in steer_layers(model):
            m.lora_B_d.zero_()
    b = generate_at_alpha(model, fmt, ["Hello there."], 0.0, max_new_tokens=6, bf16=False)
    assert a == b


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

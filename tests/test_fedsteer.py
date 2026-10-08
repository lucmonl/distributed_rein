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
                           get_shared_state, inject_steer_lora, load_state, steer_layers,
                           trainable_parameter_groups)
from fedsteer.model import generate_at_alpha

TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER)
        _TOK.pad_token = _TOK.eos_token
    return _TOK


def tiny_model(seed=0, lora_seed=1234, rank=4, warp="none", offset=False, warp_scope="model"):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tokenizer()), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    inject_steer_lora(model, SteerLoraConfig(rank_private=rank, rank_shared=rank, shared_seed=lora_seed,
                                             warp=warp, offset=offset, warp_scope=warp_scope))
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


def _ctl(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("steer_control.")}


def test_private_offset_with_shared_shape():
    """NR-60: calibration=shared + private_offset keeps o per client, still averages the shape."""
    kw = dict(calibration="shared", warp="kumaraswamy_mix", warp_scope="module", offset=True,
              private_offset=True, fix_gain=True, warp_warmup_rounds=1, lr_warp=0.2)
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, rounds=3, **kw)
        assert all(p is not tr.control.o for p in tr.shared_params)        # not reset, not averaged
        tr.fit()
        assert "steer_control.o" not in tr.server
        warp_keys = [k for k in tr.server if k.startswith("steer_control.warp.")]
        assert warp_keys and any(tr.server[k].abs().sum().item() > 0 for k in warp_keys)   # shape trained
        # clients hold only their offset, trained after the gain warm-up, and the offsets differ
        assert all(set(st["gain"]) == {"steer_control.o"} for st in tr.clients.values())
        o = {c: tr.clients[c]["gain"]["steer_control.o"].item() for c in ("c0", "c1")}
        assert all(v != 0.0 for v in o.values()) and o["c0"] != o["c1"]
        assert abs(tr.history[-1]["clients"]["c1"]["offset"] - o["c1"]) < 1e-6
        # the offset's optimizer state survives the per-round reset of the shared parameters
        tr.load_client("c0", inference=False)
        tr._load_opt("c0")
        assert tr.control.o in tr.opt.state and not any(p in tr.opt.state for p in tr.warp_params)
        # loaded clients share the shape and gain, each with its own offset
        vals = {}
        for c in ("c0", "c1"):
            tr.load_client(c)
            vals[c] = _ctl(tr.model)
            assert vals[c]["steer_control.o"].item() == o[c]
        assert all(torch.equal(vals["c0"][k], vals["c1"][k]) for k in vals["c0"] if k != "steer_control.o")
        assert all(torch.equal(vals["c0"][k], tr.server[k].to(vals["c0"][k].dtype)) for k in warp_keys)
        # snapshot -> evaluation restores the shared shape and the client's own offset
        snap = torch.load(os.path.join(d, "snapshots", "round_0003.pt"), weights_only=False)
        for c in ("c0", "c1"):
            m = tiny_model(warp="kumaraswamy_mix", offset=True, warp_scope="module")
            load_snapshot_into(m, snap, c)
            got = _ctl(m)
            assert got["steer_control.o"].item() == o[c]
            assert all(torch.allclose(got[k], tr.server[k]) for k in warp_keys + ["steer_control.u"])
    # resume reproduces the uninterrupted run, offsets included
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        full = _trainer("fedavg", d1, rounds=4, **kw)
        full.fit()
        _trainer("fedavg", d2, rounds=2, **kw).fit()
        resumed = _trainer("fedavg", d2, rounds=4, **kw)
        resumed.fit()
        for k in full.server:
            assert torch.allclose(full.server[k], resumed.server[k], atol=1e-6), k
        for c in full.clients:
            assert torch.allclose(full.clients[c]["gain"]["steer_control.o"],
                                  resumed.clients[c]["gain"]["steer_control.o"], atol=1e-6), c


def test_private_offset_is_a_noop_where_the_offset_is_already_private():
    """private_offset changes nothing with calibration=private or in local mode, and it needs
    lora.offset; without it, shared calibration still shares the offset (default path)."""
    for mode in ("fedavg", "local"):
        runs = []
        for flag in (False, True):
            with tempfile.TemporaryDirectory() as d:
                tr = _trainer(mode, d, rounds=3, calibration="private", warp="kumaraswamy_mix", offset=True,
                              private_offset=flag, warp_warmup_rounds=1)
                tr.fit()
                runs.append(tr)
        a, b = runs
        for k in a.server:
            assert torch.equal(a.server[k], b.server[k]), (mode, k)
        for c in a.clients:
            for key in ("gain", "private"):
                for k in a.clients[c][key]:
                    assert torch.equal(a.clients[c][key][k], b.clients[c][key][k]), (mode, c, k)
    with tempfile.TemporaryDirectory() as d:
        try:
            _trainer("fedavg", d, calibration="shared", offset=False, private_offset=True)
        except ValueError as e:
            assert "lora.offset" in str(e)
        else:
            raise AssertionError("expected ValueError")
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("fedavg", d, rounds=2, calibration="shared", offset=True)
        tr.fit()
        assert "steer_control.o" in tr.server and tr.server["steer_control.o"].item() != 0.0
        assert all(st["gain"]["steer_control.o"].item() == 0.0 for st in tr.clients.values())


def test_local_reset_opt_state_resets_only_the_direction():
    """NR-60: local + local_reset_opt_state empties D_i's Adam state each round and keeps the
    adapter, shape and offset states; without the flag local mode is unchanged; fedavg ignores it."""
    kw = dict(calibration="private", warp="kumaraswamy_mix", offset=True, warp_warmup_rounds=1)

    def opt_state_at_round_start(tr, cid):
        tr.load_client(cid, inference=False)
        tr._load_opt(cid)
        return tr.opt.state

    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("local", d, rounds=3, local_reset_opt_state=True, **kw)
        n_reset = []
        orig = tr._load_opt

        def spy(cid):                                    # check the state at every round start
            orig(cid)
            if tr.round > 0:
                n_reset.append(cid)
                assert not any(p in tr.opt.state for p in tr.shared_params), (tr.round, cid)
        tr._load_opt = spy
        tr.fit()
        assert len(n_reset) == 4                         # rounds 1, 2 x two clients
        st = opt_state_at_round_start(tr, "c0")
        groups = trainable_parameter_groups(tr.model)
        assert all(p not in st for p in groups["shared"])
        for name in ("private", "warp"):
            assert all(p in st for p in groups[name]), name
        assert tr.control.o in st
    with tempfile.TemporaryDirectory() as d:
        tr = _trainer("local", d, rounds=2, **kw)
        tr.fit()
        st = opt_state_at_round_start(tr, "c0")
        assert all(p in st for p in trainable_parameter_groups(tr.model)["shared"])   # kept by default
    for mode in ("fedavg", "local"):
        runs = []
        for flag in (False, True):
            with tempfile.TemporaryDirectory() as d:
                tr = _trainer(mode, d, rounds=3, local_reset_opt_state=flag, **kw)
                tr.fit()
                runs.append(tr)
        a, b = runs
        same = all(torch.equal(a.clients[c]["shared_local"][k], b.clients[c]["shared_local"][k])
                   for c in a.clients for k in a.clients[c]["shared_local"]) if mode == "local" else \
            all(torch.equal(a.server[k], b.server[k]) for k in a.server)
        assert same == (mode == "fedavg"), mode          # fedavg unaffected; local reset changes D_i


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


# ------------------------------------------------------------------ E2 / E3 helpers

def _snap_params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _changed(before, m):
    return {n for n, p in m.named_parameters() if not torch.equal(before[n], p.detach())}


def test_adapter_sft_changes_only_private_adapter_and_restores_direction():
    from fedsteer.adapt import train_adapter_sft
    m = tiny_model(offset=True, warp="kumaraswamy_mix")
    randomize_lora(m)
    flags = {n: p.requires_grad for n, p in m.named_parameters()}
    before = _snap_params(m)
    ex, _ = build_clients(toy_records(), ["c0"])
    losses = train_adapter_sft(m, ChatFormatter(tokenizer()), ex["c0"], steps=3, lr=5e-3, batch_size=2, bf16=False)
    assert len(losses) == 3
    ch = _changed(before, m)
    assert ch and all(n.endswith(("lora_A_p", "lora_B_p")) for n in ch), ch
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters() if n.endswith("lora_B_d"))
    assert {n: p.requires_grad for n, p in m.named_parameters()} == flags


def test_fit_calibration_changes_only_control():
    from fedsteer.adapt import fit_calibration
    m = tiny_model(offset=True, warp="kumaraswamy_mix")
    randomize_lora(m)
    before = _snap_params(m)
    ex, _ = build_clients(toy_records(), ["c0"])
    fit_calibration(m, ChatFormatter(tokenizer()), ex["c0"][:6], steps=3, lr=0.05, batch_size=2, bf16=False)
    ch = _changed(before, m)
    assert ch and all(n.startswith("steer_control.") for n in ch), ch


def test_local_direction_trains_from_zero_with_adapter_frozen():
    from fedsteer.adapt import train_local_direction
    m = tiny_model()
    randomize_lora(m)
    before = _snap_params(m)
    ex, _ = build_clients(toy_records(), ["c0"])
    train_local_direction(m, ChatFormatter(tokenizer()), ex["c0"][:6], steps=3, lr=5e-3, batch_size=2, bf16=False)
    ch = _changed(before, m)
    assert any(n.endswith("lora_B_d") for n in ch)
    assert not any(n.endswith(("lora_A_p", "lora_B_p")) for n in ch)
    # started from zero: after 3 small steps the direction is far smaller than the random one before
    nb = sum(before[n].norm() for n in before if n.endswith("lora_B_d"))
    na = sum(p.norm() for n, p in m.named_parameters() if n.endswith("lora_B_d"))
    assert na < 0.5 * nb


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
    model = tiny_model(warp=kw.pop("warp", "none"), offset=kw.pop("offset", False),
                       warp_scope=kw.pop("warp_scope", "model"))
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


def test_failure_penalized_calibration():
    from fedsteer.metrics import metrics_for_client, summarize
    q = ClientQuantiles.fit([0, 1])  # CDF(0) = .25, CDF(1) = .75
    alphas = [0.25, 0.75]
    valid = metrics_for_client([[0, 1]], alphas, q)
    assert valid["pct_calib_err_penalized"] == valid["pct_calib_err"] == 0

    # Keep the valid cell of an incomplete sweep. The old metric still drops that
    # sweep entirely; the new one averages errors [0, 0, .5, 1] across all cells.
    mixed = metrics_for_client([[0, 1], [1, float("nan")]], alphas, q)
    assert mixed["pct_calib_err"] == 0
    assert mixed["unscorable_row_rate"] == 0.5
    assert mixed["pct_calib_err_penalized"] == 0.375

    # Even with no complete sweep, finite cells contribute and the score exists.
    partial = metrics_for_client([[0, float("inf")], [float("-inf"), 1]], alphas, q)
    assert partial["pct_calib_err"] is None
    assert partial["pct_calib_err_penalized"] == 0.5
    failed = metrics_for_client([[float("nan"), float("inf")]], alphas, q)
    assert failed["pct_calib_err_penalized"] == 1
    empty = metrics_for_client(np.empty((0, 2)), alphas, q)
    assert empty["pct_calib_err_penalized"] is None

    # Summary weights clients equally, not by their numbers of generated cells;
    # an all-failed client must count, with the largest error marked worst.
    summary = summarize({"valid": valid, "mixed": mixed, "failed": failed, "empty": empty})
    assert summary["pct_calib_err_penalized"] == {"mean": 1.375 / 3, "worst": 1.0}


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

# ------------------------------------------------- coverage-aware calibration (plan 2.1)


class _raises:
    """Minimal stand-in for pytest.raises (the file also runs without pytest)."""

    def __init__(self, exc, match=""):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, typ, val, tb):
        if typ is None:
            raise AssertionError(f"{self.exc.__name__} not raised")
        if not issubclass(typ, self.exc) or self.match not in str(val):
            return False
        return True

def _cov_trainer(out_dir, rounds=3, **kw):
    model = tiny_model(warp=kw.pop("warp", "kumaraswamy_mix"), offset=kw.pop("offset", False),
                       warp_scope=kw.pop("warp_scope", "model"))
    ex, _ = build_clients(toy_records(), ["c0", "c1"], alpha_mode="global")
    cfg = FedConfig(mode=kw.pop("mode", "fedavg"), rounds=rounds, local_steps=3, batch_size=4, bf16=False,
                    warmup_steps=1, gain_warmup_rounds=1, save_every=1, lr_private=5e-3, lr_shared=5e-3,
                    calibration=kw.pop("calibration", "coverage"), fix_gain=kw.pop("fix_gain", True), warp_reg=0.0,
                    warp_warmup_rounds=kw.pop("warp_warmup_rounds", 1), lr_warp=0.2,
                    cov_grid=5, cov_bandwidth=0.5, cov_lambda_max=kw.pop("cov_lambda_max", 5.0),
                    cov_tau_local=2.0, cov_tau_peer=2.0, **kw)
    return FedSteerTrainer(model, ChatFormatter(tokenizer()), ex, cfg, out_dir)


def test_coverage_counts_are_triangular():
    from fedsteer.coverage import evidence_counts, make_grid
    g = make_grid(3)                                     # 0, 0.5, 1
    c = evidence_counts([0.0, 0.25, 0.5, 1.0], g, 0.5)
    assert np.allclose(c, [1.0 + 0.5, 0.5 + 1.0, 1.0])   # distance >= b contributes 0
    assert np.allclose(evidence_counts([0.5], g, 0.5), [0, 1, 0])
    assert np.allclose(evidence_counts([], g, 0.2), 0)
    assert np.allclose(evidence_counts([0.2], make_grid(11), 0.2), [0, 0.5, 1, 0.5] + [0] * 7)
    with _raises(ValueError):
        make_grid(1)


def test_coverage_borrow_weights_gate_on_peer_evidence():
    from fedsteer.coverage import borrow_weights
    counts = {"a": np.array([10.0, 0.0, 0.0]), "b": np.array([0.0, 10.0, 0.0])}
    lam = borrow_weights(counts, lambda_max=2.0, tau_local=10.0, tau_peer=10.0)
    assert np.allclose(lam["a"], [0.0, 2.0 * 1.0 * 0.5, 0.0])   # borrows only where b has data
    assert np.allclose(lam["b"], [2.0 * 0.5, 0.0, 0.0])
    one = borrow_weights({"a": np.array([0.0, 5.0])}, 1.0, 1.0, 1.0)
    assert np.allclose(one["a"], 0.0)                          # nobody else: no borrowing


def test_coverage_pooling_and_monotone_projection():
    from fedsteer.coverage import pooled_target, project_monotone
    prev = np.linspace(0, 1, 5)
    # count-weighted pooling: each client contributes only where it has evidence
    vals = {"a": np.array([0, 0.1, 0.2, 0.3, 1]), "b": np.array([0, 0.5, 0.6, 0.9, 1])}
    cnts = {"a": np.array([1, 3, 1, 0, 0.0]), "b": np.array([0, 1, 1, 2, 1.0])}
    m, z, diag = pooled_target(vals, cnts, prev)
    assert np.allclose(m[1:4], [(0.3 + 0.5) / 4, 0.4, 0.9])
    assert z[0] == 0 and z[-1] == 1 and np.all(np.diff(z) >= 0)
    # a non-monotone raw mean is pooled; endpoints and uncovered points are fixed
    z = project_monotone(np.array([0, 0.8, 0.2, 0.5, 1]), np.array([1, 1, 1, 0, 1.0]), prev)
    assert np.allclose(z, [0, 0.5, 0.5, 0.75, 1])
    # free values beyond a fixed neighbour are clipped to it (exact box-constrained solution)
    z = project_monotone(np.array([0, 0.9, 0.9, 0.5, 1]), np.array([1, 1, 1, 0, 1.0]), prev)
    assert np.allclose(z, [0, 0.75, 0.75, 0.75, 1])
    _, _, diag = pooled_target({"a": np.array([0, 0.8, 0.2, 0.5, 1])},
                               {"a": np.array([1, 1, 1, 0, 1.0])}, prev)
    assert diag["uncovered"] == [3] and abs(diag["proj_adjust_max"] - 0.3) < 1e-9


def test_coverage_rejects_incompatible_configs():
    with tempfile.TemporaryDirectory() as d:
        for kw in ({"fix_gain": False}, {"offset": True}, {"warp": "none"}, {"mode": "local"},
                   {"clients_per_round": 1}):
            with _raises(ValueError, match="coverage"):
                _cov_trainer(d, **kw)


def test_coverage_borrow_gradient_reaches_only_the_warp():
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d)
        tr.cov["ready"] = True
        tr.cov["z"] = [[0, 0.6, 0.8, 0.9, 1]]
        tr.load_client("c0")
        tr.model.zero_grad(set_to_none=True)
        lam = torch.tensor(tr.cov["lambda"]["c0"]).clamp_min(1.0)
        z = torch.tensor(tr.cov["z"])
        loss = (lam * (tr._warp_on_grid() - z) ** 2).mean()
        assert not z.requires_grad
        loss.backward()
        got = {n for n, p in tr.model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        assert got and all(n.startswith("steer_control.warp.") for n in got)


def test_coverage_training_keeps_gain_one_and_private_warps():
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d, rounds=3, warp_warmup_rounds=1)
        assert os.path.exists(os.path.join(d, "coverage.json"))
        tr.fit()
        assert not any(k.startswith("steer_control") for k in tr.server)           # warps never averaged
        for c in ("c0", "c1"):
            assert tr.clients[c]["gain"]["steer_control.u"].item() == 0.0          # s = 1 throughout
        w0, w1 = (tr.clients[c]["gain"] for c in ("c0", "c1"))
        assert any(not torch.equal(w0[k], w1[k]) for k in w0 if k.startswith("steer_control.warp."))
        h = [r["coverage"] for r in tr.history]
        assert "z" not in h[0] and not h[0]["borrow_active"]       # round 0: warps frozen, no target
        assert "z" in h[1] and not h[1]["borrow_active"]           # first trained values -> table
        assert h[2]["borrow_active"] and "borrow_loss" in tr.history[2]["clients"]["c0"]
        z = tr.cov["z"][0]
        assert z[0] == 0 and z[-1] == 1 and all(b >= a for a, b in zip(z, z[1:]))
        assert tr.cov["updates"] == 2 and tr.cov["ready"]


def test_coverage_save_resume_and_eval_round_trip():
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d, rounds=2)
        tr.fit()
        tr2 = _cov_trainer(d, rounds=3)
        assert tr2.maybe_resume() and tr2.round == 2
        assert tr2.cov == tr.cov
        tr2.fit()
        assert tr2.round == 3 and tr2.cov["updates"] == 2
        snap = torch.load(os.path.join(d, "snapshots", "round_0003.pt"), weights_only=False)
        assert snap["coverage"]["z"] == tr2.cov["z"]
        m = tiny_model(warp="kumaraswamy_mix")
        load_snapshot_into(m, snap, "c1")                          # each client evaluates its own warp
        for k, v in snap["clients"]["c1"]["gain"].items():
            assert torch.equal(dict(m.named_parameters())[k].detach().cpu(), v)
        # resume refuses a state whose gain moved away from s = 1
        st = torch.load(os.path.join(d, "state.pt"), weights_only=False)
        st["clients"]["c0"]["gain"]["steer_control.u"] = torch.tensor(0.3)
        torch.save(st, os.path.join(d, "state.pt"))
        with _raises(ValueError, match="s != 1"):
            _cov_trainer(d, rounds=4).maybe_resume()
        # and a state saved under different coverage settings
        with tempfile.TemporaryDirectory() as d2:
            _cov_trainer(d2, rounds=1).fit()
            with _raises(ValueError, match="lambda_max"):
                _cov_trainer(d2, rounds=2, cov_lambda_max=1.0).maybe_resume()



# ------------------------------------------------------------ per-layer warps (warp_scope)

def test_warp_scope_builds_one_warp_per_layer():
    from fedsteer.warp import WarpBank
    n_mod = len(list(steer_layers(tiny_model())))
    m = tiny_model(warp="kumaraswamy_mix")                       # model scope: names unchanged
    assert not isinstance(m.steer_control.warp, WarpBank)
    assert {n for n, _ in m.named_parameters() if n.startswith("steer_control.warp.")} == {
        "steer_control.warp.log_p", "steer_control.warp.log_q", "steer_control.warp.w_logit"}
    m = tiny_model(warp="kumaraswamy_mix", warp_scope="module")
    assert len(m.steer_control.warp) == n_mod
    assert sorted({mod.warp_idx for _, mod in steer_layers(m)}) == list(range(n_mod))
    m = tiny_model(warp="kumaraswamy_mix", warp_scope="block")   # tiny model: 2 blocks
    assert len(m.steer_control.warp) == 2
    for name, mod in steer_layers(m):
        assert mod.warp_idx == int(name.split(".layers.")[1].split(".")[0])
    m = tiny_model(warp="none", warp_scope="module")             # no warp: scope is ignored
    assert not isinstance(m.steer_control.warp, WarpBank)
    with _raises(ValueError, match="warp_scope"):
        tiny_model(warp="kumaraswamy_mix", warp_scope="layerz")


def test_per_layer_warps_identity_at_init_and_independent():
    a = tiny_model(warp="kumaraswamy_mix")
    b = tiny_model(warp="kumaraswamy_mix", warp_scope="module")
    randomize_lora(a), randomize_lora(b)
    ids = torch.tensor([[1, 5, 9, 2]])
    for al in (0.0, 0.3, 1.0):                                    # identical at init
        assert torch.allclose(logits(a, ids, al), logits(b, ids, al), atol=1e-5)
    ctl = b.steer_control
    _randomize_warp(ctl.warp.bank[3], 7)                          # change ONE layer's warp
    y = torch.zeros(1, 4)
    with ctl.use_alpha(0.4):
        c3, c0 = ctl.coef_for(y, 3), ctl.coef_for(y, 0)
    assert abs(float(c0) - 0.4) < 1e-6 and abs(float(c3) - 0.4) > 1e-3
    with ctl.use_alpha(1.0):                                      # endpoints stay fixed per layer
        assert abs(float(ctl.coef_for(y, 3)) - 1.0) < 1e-5
    assert not torch.allclose(logits(a, ids, 0.4), logits(b, ids, 0.4), atol=1e-6)
    with ctl.use_alpha(0.4), _raises(ValueError, match="warp index"):
        ctl.coef_for(y)


def test_per_layer_warps_get_their_own_gradients():
    m = tiny_model(warp="kumaraswamy_mix", warp_scope="module")
    randomize_lora(m)
    with m.steer_control.use_alpha(torch.tensor([0.3, 0.7])):
        m(input_ids=torch.tensor([[1, 5, 9, 2], [3, 4, 6, 8]]), labels=torch.tensor([[1, 5, 9, 2], [3, 4, 6, 8]])).loss.backward()
    g = torch.stack([w.log_p.grad for w in m.steer_control.warp.bank])
    assert torch.isfinite(g).all() and (g != 0).sum() > len(g) // 2
    assert len(set(round(float(x), 8) for x in g)) > 1            # layers are not tied


def test_per_layer_warps_in_federated_modes():
    for calib in ("private", "shared"):
        with tempfile.TemporaryDirectory() as d:
            tr = _trainer("fedavg", d, warp="kumaraswamy_mix", warp_scope="module", rounds=2,
                          warp_warmup_rounds=1, lr_warp=0.2, calibration=calib)
            tr.fit()
            n = len(tr.model.steer_control.warp)
            if calib == "private":
                st = tr.clients["c0"]["gain"]
                assert sum(k.startswith("steer_control.warp.bank.") for k in st) == 3 * n
                lp = torch.stack([st[f"steer_control.warp.bank.{l}.log_p"] for l in range(n)])
                assert len(set(lp.tolist())) > 1                  # per-layer warps differ
                assert not any(k.startswith("steer_control") for k in tr.server)
            else:
                assert sum(k.startswith("steer_control.warp.bank.") for k in tr.server) == 3 * n
            desc = tr.history[-1]["clients"]["c0"]["warp"]
            assert desc["scope"] == "module" and desc["n_warps"] == n and "curve_min" in desc
            snap = torch.load(os.path.join(d, "snapshots", "round_0002.pt"), weights_only=False)
            m = tiny_model(warp="kumaraswamy_mix", warp_scope="module")
            load_snapshot_into(m, snap, "c1")
            src = snap["server"] if calib == "shared" else snap["clients"]["c1"]["gain"]
            k = "steer_control.warp.bank.5.log_q"
            assert torch.equal(dict(m.named_parameters())[k].detach().cpu(), src[k])


def test_per_layer_warps_with_coverage_borrowing():
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d, rounds=3, warp_scope="module")
        n = len(tr.model.steer_control.warp)
        assert tr.cov["n_warps"] == n and np.asarray(tr.cov["z"]).shape == (n, 5)
        tr.fit()
        z = np.asarray(tr.cov["z"])
        assert z.shape == (n, 5) and np.all(z[:, 0] == 0) and np.all(z[:, -1] == 1)
        assert np.all(np.diff(z, axis=1) >= 0)
        assert len({tuple(np.round(r, 8)) for r in z}) > 1       # one table per layer, not tied
        last = tr.history[-1]
        assert "z_min" in last["coverage"] and "values_max" in last["coverage"]
        assert "borrow_loss" in last["clients"]["c0"]
        tr2 = _cov_trainer(d, rounds=4, warp_scope="module")
        assert tr2.maybe_resume() and np.allclose(tr2.cov["z"], z)
        with _raises(ValueError, match="warp_scope"):
            _cov_trainer(d, rounds=4).maybe_resume()               # model scope vs saved bank



# --------------------------------------------- consensus calibration (NR-56: tie in support)

def test_consensus_weights_tie_only_where_data_and_saturate():
    from fedsteer.coverage import saturating_weights
    w = saturating_weights({"a": np.array([0.0, 100.0, 10000.0])}, tau=100.0, scale=2.0)
    assert np.allclose(w["a"], [0.0, 1.0, 2.0 * 10000 / 10100])
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d, calibration="consensus")
        for c in ("c0", "c1"):
            cnt, lam = np.asarray(tr.cov["counts"][c]), np.asarray(tr.cov["lambda"][c])
            assert np.all(lam[cnt == 0] == 0) and np.all(lam[cnt > 0] > 0)      # no tie off support
            assert np.all(np.asarray(tr.cov["pool_weights"][c]) < 1.0)            # saturating pool
        assert tr.cov["mode"] == "consensus" and tr.cov["pool"] == "saturating"


def test_table_map_interpolates_per_layer():
    m = tiny_model(warp="kumaraswamy_mix", warp_scope="block")
    ctl = m.steer_control
    grid = [0.0, 0.5, 1.0]
    ctl.set_table(grid, [[0.0, 0.2, 1.0], [0.0, 0.8, 1.0]])
    y = torch.zeros(3, 4)
    with ctl.use_alpha(torch.tensor([0.25, 0.5, 0.75])):
        assert torch.allclose(ctl.coef_for(y, 0).reshape(-1), torch.tensor([0.1, 0.2, 0.6]))
        assert torch.allclose(ctl.coef_for(y, 1).reshape(-1), torch.tensor([0.4, 0.8, 0.9]))
    with ctl.use_alpha(1.0):
        assert abs(float(ctl.coef_for(y, 1)) - 1.0) < 1e-6
    with _raises(ValueError, match="increasing"):
        ctl.set_table(grid, [[0.0, 0.6, 1.0], [0.0, 0.5, 0.9]])
    with _raises(ValueError, match="shape"):
        ctl.set_table(grid, [[0.0, 0.5, 1.0]])
    ctl.set_table(None)
    with ctl.use_alpha(0.3):                                    # back to the (identity) warps
        assert abs(float(ctl.coef_for(y, 0)) - 0.3) < 1e-6


def test_consensus_trains_local_warps_and_infers_with_table():
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d, rounds=3, calibration="consensus", warp_scope="module")
        tr.fit()
        n = tr.control.n_warps()
        z = np.asarray(tr.cov["z"])
        assert z.shape == (n, 5) and np.all(np.diff(z, axis=1) >= 0)
        assert "tie_loss" in tr.history[2]["clients"]["c0"] and "borrow_loss" not in tr.history[2]["clients"]["c0"]
        assert set(tr.history[2]["coverage"]["support_gap"]) == {"c0", "c1"}
        tr.load_client("c0", inference=False)                 # training: own warps
        assert tr.control._table is None
        tr.load_client("c0")                                  # monitor / evaluation: g-bar
        assert tr.control._table is not None
        y = torch.zeros(1, 4)
        with tr.control.use_alpha(0.5):
            got = float(tr.control.coef_for(y, 7))
        assert abs(got - z[7, 2]) < 1e-5                      # grid point 0.5 of layer 7's table
        snap = torch.load(os.path.join(d, "snapshots", "round_0003.pt"), weights_only=False)
        m = tiny_model(warp="kumaraswamy_mix", warp_scope="module")
        load_snapshot_into(m, snap, "c1")
        with m.steer_control.use_alpha(0.5):
            assert abs(float(m.steer_control.coef_for(y, 7)) - z[7, 2]) < 1e-5
        load_snapshot_into(m, snap, "c1", use_table=False)    # diagnostic: client's own warps
        assert m.steer_control._table is None
        with m.steer_control.use_alpha(0.5):
            own = float(m.steer_control.coef_for(y, 7))
        assert abs(own - float(m.steer_control.warp.bank[7](torch.tensor(0.5)))) < 1e-6
    with tempfile.TemporaryDirectory() as d:                  # a non-consensus snapshot clears the table
        tr = _trainer("fedavg", d, warp="kumaraswamy_mix", rounds=1)
        tr.fit()
        snap = torch.load(os.path.join(d, "snapshots", "round_0001.pt"), weights_only=False)
        m = tiny_model(warp="kumaraswamy_mix")
        m.steer_control.set_table([0.0, 1.0], [[0.0, 1.0]])
        load_snapshot_into(m, snap, "c0")
        assert m.steer_control._table is None



# ------------------------------------------- aligned calibration (NR-58: g-bar in train & test)

def test_pool_and_project_identity_gradient_and_blocks():
    from fedsteer.calibrate import isotonic_increasing
    from fedsteer.coverage import pool_and_project
    den = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0])
    m_ok = torch.tensor([[0.0, 0.2, 0.5, 0.7, 1.0]])
    num = (m_ok * den).requires_grad_(True)
    z, d = pool_and_project(num, den)
    assert torch.allclose(z, m_ok) and d["proj_adjust_max"] < 1e-7 and d["proj_layers"] == 0
    z[0, 2].backward()                                    # identity: dz_k/dnum_k = 1/den_k only
    assert torch.allclose(num.grad[0], torch.tensor([0, 0, 1.0, 0, 0]))
    m_bad = torch.tensor([[0.0, 0.6, 0.3, 0.7, 1.0]])     # one violation
    num = (m_bad * den).requires_grad_(True)
    z, d = pool_and_project(num, den)
    ref = isotonic_increasing(m_bad[0].numpy(), den.numpy())
    assert np.allclose(z[0].detach().numpy(), ref) and d["proj_layers"] == 1
    z[0, 1].backward()                                    # pooled block {1, 2}: shared value
    assert torch.allclose(num.grad[0], torch.tensor([0, 1 / 3, 1 / 3, 0, 0]), atol=1e-6)
    den0 = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])        # point 2 uncovered: interpolated
    z, d = pool_and_project(torch.tensor([[0.0, 0.2, 0.0, 0.8, 1.0]]), den0)
    assert abs(float(z[0, 2]) - 0.5) < 1e-6 and d["uncovered"] == [2]
    assert float(z[0, 0]) == 0.0 and float(z[0, -1]) == 1.0


def test_pool_and_project_own_gradient_only_where_client_has_weight():
    from fedsteer.coverage import pool_and_project
    own = torch.tensor([[0.0, 0.3, 0.5, 0.7, 1.0]], requires_grad=True)
    own_w = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])       # client has no data at points 2, 3
    others = torch.tensor([[0.0, 0.2, 0.45, 0.8, 1.0]]) * 2.0
    z, _ = pool_and_project(others + own_w * own, torch.full((5,), 2.0) + own_w)
    z[0, 1:4].sum().backward()
    assert own.grad[0, 1] > 0 and own.grad[0, 2] == 0 and own.grad[0, 3] == 0


def test_aligned_trains_through_gbar_and_infers_with_the_same_map():
    from fedsteer.coverage import pool_and_project
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d, rounds=3, calibration="aligned", warp_scope="module", cov_lambda_max=0.01)
        seen = []                                         # the table each training forward used
        orig = tr.control.set_live_table
        tr.control.set_live_table = lambda g, z: (seen.append(z.requires_grad), orig(g, z))
        tr.fit()
        assert seen and any(seen)                         # live table carries gradient once warps train
        assert tr.control._table is None                  # cleared after local training
        n = tr.control.n_warps()
        V = {c: torch.tensor(v, dtype=torch.float64) for c, v in tr.cov["values"].items()}
        W = {c: torch.tensor(v, dtype=torch.float64) for c, v in tr.cov["pool_weights"].items()}
        z_ref, _ = pool_and_project(sum(W[c] * V[c] for c in V), sum(W.values()))
        assert np.allclose(tr.cov["z"], z_ref.numpy()) and np.asarray(tr.cov["z"]).shape == (n, 5)
        # a client's training view with everyone's final values IS the inference table
        loo = sum(W[c] * V[c] for c in V if c != "c0")
        z_c0, _ = pool_and_project(loo + W["c0"] * V["c0"], sum(W[c] for c in V if c != "c0") + W["c0"])
        assert torch.allclose(z_c0, z_ref)
        log = tr.history[-1]
        assert {"disagreement", "proj_layers", "support_gap"} <= set(log["coverage"])
        assert "tie_loss" in log["clients"]["c0"] and "proj_adjust_max" in log["clients"]["c0"]
        w0 = tr.clients["c0"]["gain"]                     # warps trained (left the identity)
        assert any(float(w0[k]) != 0.0 for k in w0 if k.endswith("log_p"))
        tr.load_client("c1")                              # inference: the table
        y = torch.zeros(1, 4)
        with tr.control.use_alpha(0.5):
            assert abs(float(tr.control.coef_for(y, 3)) - tr.cov["z"][3][2]) < 1e-5
        snap = torch.load(os.path.join(d, "snapshots", "round_0003.pt"), weights_only=False)
        m = tiny_model(warp="kumaraswamy_mix", warp_scope="module")
        load_snapshot_into(m, snap, "c1")
        with m.steer_control.use_alpha(0.5):
            assert abs(float(m.steer_control.coef_for(y, 3)) - tr.cov["z"][3][2]) < 1e-5
        tr2 = _cov_trainer(d, rounds=4, calibration="aligned", warp_scope="module", cov_lambda_max=0.01)
        assert tr2.maybe_resume() and tr2.cov["values"] == tr.cov["values"]


def test_aligned_with_private_offset():
    """NR-62: aligned + lora.offset: o_i stays private and adds to the pooled table
    (coefficient = o_i + g-bar_l(alpha)); only the shapes are pooled; coverage and consensus
    still reject an offset."""
    with tempfile.TemporaryDirectory() as d:
        for mode in ("coverage", "consensus"):
            try:
                _cov_trainer(d, calibration=mode, warp_scope="module", offset=True)
            except ValueError as e:
                assert "lora.offset" in str(e)
            else:
                raise AssertionError(f"{mode} should reject an offset")
    with tempfile.TemporaryDirectory() as d:
        tr = _cov_trainer(d, rounds=3, calibration="aligned", warp_scope="module", cov_lambda_max=0.01,
                          offset=True)
        tr.fit()
        assert not any(k.startswith("steer_control") for k in tr.server)          # nothing pooled by FedAvg
        o = {c: float(tr.clients[c]["gain"]["steer_control.o"]) for c in ("c0", "c1")}
        assert all(v != 0.0 for v in o.values()) and o["c0"] != o["c1"]
        assert "offset" in tr.history[-1]["clients"]["c0"]
        y = torch.zeros(1, 4)
        for c in ("c0", "c1"):
            tr.load_client(c)                                     # inference: table + own offset
            with tr.control.use_alpha(0.5):
                assert abs(float(tr.control.coef_for(y, 3)) - (tr.cov["z"][3][2] + o[c])) < 1e-5
            with tr.control.use_alpha(0.0):
                assert abs(float(tr.control.coef_for(y, 3)) - o[c]) < 1e-6
        snap = torch.load(os.path.join(d, "snapshots", "round_0003.pt"), weights_only=False)
        m = tiny_model(warp="kumaraswamy_mix", warp_scope="module", offset=True)
        load_snapshot_into(m, snap, "c1")
        with m.steer_control.use_alpha(0.5):
            assert abs(float(m.steer_control.coef_for(y, 3)) - (tr.cov["z"][3][2] + o["c1"])) < 1e-5
        tr2 = _cov_trainer(d, rounds=4, calibration="aligned", warp_scope="module", cov_lambda_max=0.01,
                           offset=True)
        assert tr2.maybe_resume() and float(tr2.clients["c1"]["gain"]["steer_control.o"]) == o["c1"]


def test_aligned_with_shared_offset():
    """NR-60b: aligned + shared_offset: one offset o for all clients, averaged by the server like
    the direction (coefficient = o + g-bar_l(alpha)); refused outside aligned/fedavg/offset."""
    with tempfile.TemporaryDirectory() as d:
        for kw in (dict(calibration="aligned", offset=False), dict(calibration="consensus", offset=True),
                   dict(calibration="aligned", offset=True, mode="local")):
            try:
                _cov_trainer(d, warp_scope="module", shared_offset=True, **kw)
            except ValueError as e:
                assert "offset" in str(e) or "mode" in str(e), e
            else:
                raise AssertionError(f"shared_offset should be refused for {kw}")
    with tempfile.TemporaryDirectory() as d:
        kw = dict(calibration="aligned", warp_scope="module", cov_lambda_max=0.01, offset=True, shared_offset=True)
        tr = _cov_trainer(d, rounds=3, **kw)
        assert any(p is tr.control.o for p in tr.shared_params)                # reset + averaged
        tr.fit()
        o = float(tr.server["steer_control.o"])
        assert o != 0.0
        assert [k for k in tr.server if k.startswith("steer_control")] == ["steer_control.o"]
        assert all("steer_control.o" not in st["gain"] for st in tr.clients.values())
        # the server's o is the average of the clients' uploads: the last round's per-client values
        last = tr.history[-1]["clients"]
        assert abs(o - sum(v["offset"] for v in last.values()) / len(last)) < 1e-6
        y = torch.zeros(1, 4)
        for c in ("c0", "c1"):                                 # every client: o + table
            tr.load_client(c)
            with tr.control.use_alpha(0.5):
                assert abs(float(tr.control.coef_for(y, 3)) - (tr.cov["z"][3][2] + o)) < 1e-5
        snap = torch.load(os.path.join(d, "snapshots", "round_0003.pt"), weights_only=False)
        m = tiny_model(warp="kumaraswamy_mix", warp_scope="module", offset=True)
        load_snapshot_into(m, snap, "c1")
        with m.steer_control.use_alpha(0.0):
            assert abs(float(m.steer_control.coef_for(y, 3)) - o) < 1e-6
        tr2 = _cov_trainer(d, rounds=4, **kw)
        assert tr2.maybe_resume() and float(tr2.server["steer_control.o"]) == o


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

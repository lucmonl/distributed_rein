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


def tiny_model(seed=0, lora_seed=1234, rank=4, warp="none"):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tokenizer()), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    inject_steer_lora(model, SteerLoraConfig(rank_private=rank, rank_shared=rank, shared_seed=lora_seed,
                                             warp=warp))
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
    model = tiny_model(warp=kw.pop("warp", "none"))
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

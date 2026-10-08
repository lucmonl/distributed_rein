# Coordinator log: cross-dataset decisions

Conventions: `../../CONVENTIONS.md`. Archive before 2026-10-07: `../EXPERIMENT_LOG.md`.

## CAL-1. Calibration design across datasets: synthesis of completed runs; MOL-18 NORESET result; NR-58 interim (2026-10-07)

Analysis only; no code or jobs. A single place for the calibration questions (gain, sharing, offset, how the shared function is learned) that were spread over NR-, MOL- and MATH- entries.

**New: MOL-18 NORESET (11180423, done 10-07 11:03, never analysed).** Report `exp_log/reports/mol18_noreset_offset_test.txt` (archive) (test, strict scorer, gain 1).

| arm | penalized | pct err (worst) | out-of-support | reach |
|---|---:|---:|---:|---:|
| PRIV (fedavg, Adam state of D reset every round) | 0.207 | 0.199 (0.279) | 0.226 | 0.327 |
| NORESET (fedavg, state kept) | 0.209 | 0.206 (0.294) | 0.239 | 0.274 |
| LOCAL | 0.207 | 0.205 (0.303) | 0.238 | 0.279 |
| PRIV_OFF | **0.193** | **0.188** (0.279) | **0.211** | 0.336 |
| LOCAL_OFF | 0.208 | 0.204 (0.303) | 0.238 | 0.277 |

- PRIV − NORESET: better on 5/8, worse on 1/8 (CHEMBL228 +0.017). NORESET − LOCAL: better 1/8, worse 2/8.
- So **without the per-round reset, federated is no better than local on ChEMBL**; the reset accounts for most of PRIV's conditional advantage. PRIV_OFF − LOCAL_OFF (5/8 vs 0/8) also contains the reset. A local arm with a periodic Adam reset is the missing control for every fed-vs-local claim on this task.

**NR-58 aligned, interim (dev, round 40 of 100):** 0.190 vs. A_L 0.191, B_L 0.200, consensus λ = 1 0.194 at the same round; nypost.com 0.241 vs. A_L 0.238. Tracking A_L, no separation yet.

**Synthesis (single seed everywhere):**

| Question | Newsroom | ChEMBL | Math |
|---|---|---|---|
| Gain fixed at 1 | Costs reuters +0.016 with one warp (entry 40, NR-48); per-layer shared warps recover it (A_L 0.150 vs. learned-gain 0.151, tie 7/8) | Learned private gain: fed loses 0/8 vs 3/8 (MOL-15); gain 1: tie (MOL-18) | **Untested**; gain at clamp 4 in single-client G2, 2.88 shared in E1-fed |
| Shape shared vs private | Shared better (B −A: worse 4/8, better 0/8; B_L − A_L worse); shapes differ by layer, not client (NR-57) | **Shared shape at gain 1 untested**; COV ≈ PRIV overall, helps one-sided specialists, hurts middle-only CHEMBL4078 | Shared only |
| Offset (level) | Shared offset hurts (0.158 vs 0.151); private offset helps private calibration (0.155 vs 0.168); **untested at gain 1 / per-layer / with shared shape**; endpoints are 43% of A's error with no warp headroom (NR-53) | Private offset gives the first fed win surviving the failure penalty (MOL-22); α = 0 unmovable otherwise | Diagnosis (MATH-13): coefficient-0 levels differ 0.07–0.68; **untested** |
| How the shared function is learned | FedAvg of warp params (A_L) best; coverage borrowing ties; consensus loses (mismatch, NR-57/59); aligned pending | Only coverage borrowing tested | None |

**Missing, by dataset:**
- Newsroom: aligned final + aligned with count pooling; any private offset under a shared shape at gain 1; matched local baseline at gain 1 with per-layer warps (± offset); E2 for consensus/aligned (the script refuses them); seeds.
- ChEMBL: shared shape at gain 1 (single or per-layer); per-layer warps; consensus/aligned; shared shape + private offset; local with periodic Adam reset.
- Math: no federated calibration ablation at all; answer-in-prompt E1 pair (Delta 22703349/50) running; MATH-14 per-layer/gain-16 single-client G2 pending.

**Code constraints for the next design:** the offset sits in the `gain` group, so `calibration=shared` cannot keep it private; `coverage`/`consensus`/`aligned` reject `lora.offset=true`.

## CAL-2. Per-dataset logs, shared terminology and job names; NR-60 (private offset) assigned to newsroom-iter1 (2026-10-07)

**User decisions.** Logs are classified by dataset; terminology and SLURM names are simple and shared across datasets; this session coordinates and three sessions (newsroom-iter1, molecule-iter1, math-cot-iter1) run the experiments.

**Done here.**
- `CONVENTIONS.md` (repo root): session/cluster ownership, ID prefixes (NR-60+, MOL-23+, MATH-16+, CAL-*), paths, terminology (knob, adapter, direction, calibration = offset + gain · shape), arm labels (`fed-shared-off`, `local`, …), job names (`nr60-fed-shared-off`), shared-code rules.
- New folders: `sbatch/logs/{newsroom,molecule,math-cot}/`, `exp_log/{newsroom,molecule,math-cot}/{LOG.md,launch/,reports/}`, `exp_log/coordinator/LOG.md`.
- `exp_log/EXPERIMENT_LOG.md` frozen as the archive (note at the top); CAL-1 moved here. Old job logs stay flat in `sbatch/logs/`.
- Not done here: sbatch `--output` headers. Each session updates its own sbatch files on all its hosts.

**NR-60 assigned (newsroom-iter1).** Private offset on Newsroom, fed and local, matched to NR-54 (Llama-3.2-1B, rotation 0, 4k, 100 rounds, gain fixed at 1, per-layer shape, warp_reg 0). Needs an opt-in flag so a shared shape can carry a private offset (today `calibration=shared` also shares the offset).

| Arm | Compared with |
|---|---|
| `fed-shared-off` | `fed-shared` = NR-54 A_L (11190971) |
| `fed-private-off` | `fed-private` = NR-54 B_L (11190972); `local-off` (only the direction differs) |
| `local-off` | `local` |
| `local` | new: the matched local baseline at gain 1 with per-layer shapes (none existed) |

**Conflict rules in force now.** Shared-code grant: newsroom-iter1 for `fedsteer/fed.py`, `fedsteer/lora.py`, `e2_heldout.py`, `e3_drift.py`, `tests/test_fedsteer.py`. Until it reports the change as complete and tested: molecule-iter1 submits no new cc jobs, math-cot-iter1 does not sync code to Delta/dtai (pending jobs there snapshot the tree when they start). molecule and math have no new experiments; their running jobs continue.

**CAL-2 follow-up (15:00): math-cot-iter1 confirmed the switch (MATH-16).** Its four math sbatch files now log to `sbatch/logs/math-cot/`. `sync_to_delta.sh` gained a `--no-code` mode; a doc/header-only sync ran on Delta and dtai. Shared code on Delta/dtai is still the old tree (hold in force). Running: E1 `fed-shared-gain-1shape` / `local-gain-1shape` (Delta 22703349/50). Pending: G1 22705844 and MATH-14 P/PG.

## CAL-3. Optimizer-reset control added to NR-60; molecule switch confirmed (2026-10-07)

- **molecule-iter1 confirmed the switch.** The four chembl sbatch files log to `sbatch/logs/molecule/%x.o%j` on cc and Anvil. It wrote MOL-23, which confirms CAL-1's NORESET numbers and adds:
  - On the dev selection metric, local is the best of fed-private / fed-private-noreset / local, by 0.009.
  - The reset's effect is reach: out-of-support 0.226 vs 0.239, reach 0.327 vs 0.274, endpoint near-ties 0.037 vs 0.071. In-support accuracy barely changes.
- **Decision: the local-with-reset control comes now, not after.** NR-60 gets a second opt-in flag, `fed.local_reset_opt_state` (local runs reset their direction's Adam state every round), and a fifth arm, `nr60-local-off-reset`. The flag goes into the same shared-code pass, under the same newsroom-iter1 grant. CONVENTIONS.md gains the `-reset` / `-noreset` arm suffixes.
- **Anvil code sync:** newsroom-iter1 does the full `setup_anvil.sh --code-only` after its tests pass. molecule-iter1 copied only its four sbatch files, to avoid pushing half-edited code.
- **Molecule next (after the code lands):** fed-shared and fed-shared-off at gain 1 with per-layer shapes, plus local-off-reset as the control, in one race. Spec to follow.

**CAL-3 follow-up: Anvil budget and reference overrides.**
- Anvil `cis260796-ai` balance at 15:30 was **473.1 of 500 SU** (1 SU per GPU-hour). Allotted if Anvil wins its race: NR-60 ≤ 45 SU (5 arms), MOL next race ≤ 20 SU (3 arms). Losing copies are cancelled while PENDING and cost ~0.
- molecule-iter1 caught two silent traps; both are now in CONVENTIONS.md §3.
  - The code default is one shape per client (`lora.warp_scope=model`), so reference arms must pass `lora.warp_scope=module`.
  - `configs/chembl_fedavg.yaml` ships `fed.warp_reg=0.01`, so reference arms must pass `fed.warp_reg=0`.
- `fed.local_reset_opt_state` is boolean: reset every round, matching fedavg.
- Molecule Anvil cap raised to **25 SU**: about 6 SU per 100-round arm × 3 arms, plus room for one retry. The molecule smoke test runs on cc after the release, not on Anvil. Total allotted: 70 of 473 SU.

## CAL-4. NR-60 code landed; holds released; MOL-24 assigned (2026-10-07)

- newsroom-iter1 finished the shared-code change:
  - Files: `fedsteer/fed.py`, `e2_heldout.py`, `e3_drift.py`, `tests/test_fedsteer.py`.
  - Flags: `fed.private_offset`, `fed.local_reset_opt_state`, both default false.
  - Tests: 74/74 pass. The defaults are bit-identical to the old code (four configs, 3 rounds, old vs. new).
  - Anvil code synced; md5s match cc.
  - Grant closed; no shared-code grants are open.
- Holds released: molecule may submit on cc again; math-cot may sync code to Delta/dtai. Its pending MATH-14 jobs will then snapshot the new code, whose defaults are unchanged.
- **MOL-24 assigned (molecule-iter1):** one race group on pruned ChEMBL, reference overrides (per-layer shape), strict scorer, penalized selection, matched to MOL-18/MOL-22 otherwise. Anvil cap 25 SU.

| Arm | Question |
|---|---|
| `mol24-fed-shared` | first shared-shape run at gain 1 on ChEMBL ("plain shared" bar) |
| `mol24-fed-shared-off` | private offset under a shared shape (= NR-60's primary arm) |
| `mol24-local-off-reset` | the fair local bar (reset on both sides) |

The MOL-22 arms (`fed-private-off`, `local-off`, one shape per client) remain context, not matched references.
- **NR-60 submitted** (newsroom-iter1): one race group, watcher pid 1157389, log `sbatch/logs/newsroom/race-NR-60.log`. Arm: cc / Anvil job IDs:
  - fed-shared-off 11206103 / 21174317
  - fed-private-off 11206104 / 21174318
  - local-off 11206105 / 21174319
  - local 11206106 / 21174320
  - local-off-reset 11206107 / 21174353

  Expected ≈ 31–35 SU if Anvil wins.

## CAL-5. Design note: what to pool (shapes h_{i,l} vs calibrations g_{i,l} = o_i + h_{i,l}), and offset per client vs per layer (2026-10-07)

Analysis only; a proposal, not yet decided by the user. Notation as in CONVENTIONS.md §3: g_{i,l}(α) = o_i + s·h_{i,l}(α) with s = 1; h̄_l is the pooled shape, ḡ_l the pooled calibration.
- **Pool the shapes h_{i,l} into h̄_l; keep o_i outside the pool and private.** Each client then uses g_{i,l}(α) = o_i + h̄_l(α).
  - The alternative pools the calibrations, ḡ_l = pool_i(o_i + h_{i,l}). That averages the o_i, i.e. it makes the offset shared again. A shared offset measured worse (entry 20: 0.158 vs 0.151).
  - Levels genuinely differ by client: at the same coefficient, output percentiles span 0.27–0.52 on Newsroom (NR-53), 0.17–0.54 on ChEMBL (MOL-21), 0.07–0.68 on math (MATH-13).
  - Coverage weights would make ḡ_l confound level with coverage. At each α it follows the clients that hold data there (on Newsroom at α 0.8–0.9, nypost/reuters hold 54–66% of the data, NR-59), so differences in o_i would appear as a bend in ḡ_l. This is the contamination of NR-52/59. h_{i,l} is pinned at h(0) = 0 and h(1) = 1 and carries no level.
- **Offset: one scalar o_i per client first** (what NR-60 runs).
  - Per-layer offsets o_{i,l} would add 112 (Llama-1B) to 196 (Qwen-7B) private parameters per client and reopen the absorption problem of private per-layer shapes (fed-private, NR-57).
  - New clients (E2) would need more fitting.
- **Known weakness of a scalar o_i:** at α = 0 it gives every layer the same coefficient o_i, a per-layer profile h̄_l never produces. Fallback: an input shift g_{i,l}(α) = h̄_l(α + δ_i), with h̄_l extended linearly outside [0, 1].
- **Not covered by o_i + h̄_l:** slope differences between clients (MATH-13). A private gain s_i fitted only in support was harmful (entry 41, MOL-15). Open.

## CAL-6. User decision (relayed by molecule-iter1): the per-round Adam reset is part of FedAvg, not a confound; molecule launches on hold (2026-10-07)

- **User position:** FedAvg overwrites the clients' shared parameters with the server average every round, so resetting their Adam moments is part of the protocol. `fed` (reset) vs `local` (state carried) is therefore the fair comparison, and the reset-control tests are cancelled.
- **Consequence for the synthesis:** MOL-23's numbers stand, read as *how* FedAvg helps on ChEMBL (reach: out-of-support 0.226 vs 0.239, reach 0.327 vs 0.274), not as a confound. CAL-1's "missing control" item and CAL-3's rationale are withdrawn. `fed.local_reset_opt_state` stays in the code (default false, harmless) but is unused. The `-reset` / `-noreset` arm suffixes remain defined only for reading old runs.
- **molecule-iter1 actions:**
  - Cancelled the smoke 11206295 and, at the user's request, the exp45 cc pair 11169098 / 11169100.
  - MOL-24's third arm is now `local-off` with per-layer shapes.
  - **All molecule launches are on hold until the user says otherwise.** MOL-24 is not submitted and does not count against Anvil SU.
- **New matched data point (exp45 cc pair, pruned ChEMBL, strict, round 100, single seed, no bootstrap):** `fed-private-gain-1shape` vs `local-gain-1shape`:
  - penalized 0.215 vs 0.205;
  - pct err 0.209 vs 0.199;
  - out-of-support 0.246 vs 0.229;
  - reach 0.263 vs 0.292;
  - fed worse on 6/8 clients.

  This confirms CAL-1's learned-gain row (MOL-15): with a learned private gain and one shape per client, fed loses.
- **Open, asked of the user:** whether `nr60-local-off-reset` (cc 11206107, Anvil 21174353, both PENDING) should be cancelled under the same decision.
- **User answer: cancel the Newsroom reset arm.** The coordinator cancelled `nr60-local-off-reset` on cc (11206107) and Anvil (21174353) while both were PENDING, at 16:58 CDT (sacct; an earlier note said 15:55). NR-60 continues with 4 arms. At about that time the cc copies of fed-shared-off (11206103) and fed-private-off (11206104) were already RUNNING. Watcher pid 1157389 handles the race; the cancelled pair counts as decided.
- CONVENTIONS.md §3 corrected after molecule-iter1's check:
  - **Every** config in `configs/` (newsroom, chembl, math) ships `fed.warp_reg=0.01`, and none sets `lora.warp_scope`. The text had named only `chembl_fedavg.yaml`.
  - o_i is a scalar per client in the code (`lora.py:87`); a per-layer o_{i,l} is not implemented.
- NR-60 race decided for **cc** at 16:58 CDT. The watcher cancelled the four Anvil copies while PENDING (0 SU). fed-shared-off 11206103 and fed-private-off 11206104 are RUNNING on ccc0284; local-off 11206105 and local 11206106 are PENDING on cc.
- math-cot-iter1 synced code to Delta and dtai (MATH-17).
  - Tests: 74/74 on cc and dtai. 73/74 on Delta: `test_b3_merge_equals_average_of_local_directions` fails in the full run only, with an MKL/libgomp threading conflict in its subprocess, and passes alone. It is an environment issue predating NR-60.
  - Queued MATH-14 P/PG and G1 will start on the new code (flags off). The E1 pair is RUNNING and unaffected.

## CAL-7. NR-61 (fed-aligned, test) and NR-60 interim: pooled-function designs still lose to fed-shared; offsets point at level as the cause (2026-10-07)

- **NR-61 (newsroom-iter1, test, single seed, round 100):**
  - Pct err: fed-aligned 0.153 vs fed-shared 0.150.
  - Paired vs fed-shared: worse on 3/8 (nypost +0.022, reuters +0.009, forbes +0.006), better on 1/8 (aol −0.007).
  - Paired vs fed-consensus-l1: better on 4/8, worse on 0/8.
  - The isotonic projection never acted, and the tie was ≤ 8e-5, so ḡ_l was in effect the saturating-weighted mean of the h_{i,l}.
  - The loss sits at α = 0 / 0.25 on nypost and reuters, as NR-59 predicted.
  - Ranking: fed-shared 0.150 < borrow-l1 0.151 < aligned 0.153 < borrow-l0p1 0.154 < consensus 0.156–0.158 ≈ private 0.158. **No pooled-function design beats FedAvg of the shape parameters.**
- **NR-60 interim (dev, round 50):**
  - Both offset arms lead their no-offset counterparts: fed-shared-off 0.173 vs 0.176, fed-private-off 0.175 vs 0.181.
  - Every o_i is negative; the largest are nypost (−0.63) and reuters (−0.42), the very clients aligned and consensus lose on.
  - The dev α = 0 cell improves 0.260 → 0.240.
- **Coordinator reading (hypothesis):** the copy-heavy clients disagree with the pooled shape because their *level* differs (P_i absorbs the copy style), not because their shape does. Saturating weights then down-weight them where they own the data.
  - If o_i absorbs the level, their h_{i,l} should move toward the others', and the pooling weights would matter less.
  - So the informative aligned test is `fed-aligned-off` (pool h_{i,l}, private o_i; CAL-5), not `fed-aligned-cnt` on its own.
  - Decide after NR-60's test results. Nothing launched; the user is designing the next aligned variant.
- **Level-vs-shape check, early (newsroom-iter1, round 50).** Measure: mean over the 112 layers of |h_{i,l}(α) − the other clients' mean h(α)|. Values are fed-private → fed-private-off.
  - nypost at α = 0.5 / 0.75 (inside its support [0.51, 0.95]): 0.328 → 0.234 and 0.313 → 0.194, i.e. 29–38% smaller.
  - reuters: 0.121 → 0.098 and 0.097 → 0.065.
  - Other clients at α = 0.5: 0.060–0.133 → 0.048–0.101.
  - **Reading:** the private offset removes roughly a third of nypost's shape gap. nypost stays the most distinct client, so its difference is partly level and still largely shape.
  - Weaker than CAL-7's hypothesis, which was mostly level. Redo at the selected round.
  - nypost's α = 0.25 value is outside its support (an unconstrained extrapolation), so read in-support α only.

## CAL-8. NR-62 `fed-aligned-off` launched at the user's direct request (2026-10-07)

- newsroom-iter1 launched it on the user's instruction: cc 11209264, Anvil 21179348, watcher pid 2741896, ≈ 6–7 SU if Anvil wins.
- Model: g_{i,l}(α) = o_i + h̄_l(α).
  - h̄_l is built as in NR-58 (saturating weights, K = 21, λ_max = 0.01, the same table in training and inference).
  - o_i is private. s = 1.
- **Checked in the code:** the pooled values are each client's shape values on the grid (`_warp_on_grid`), without o_i. So this pools the h_{i,l} into h̄_l, as CAL-5 proposed. Newsroom's message wrote the table as ḡ_l; by CONVENTIONS §3 it is h̄_l.
- **Shared-code change:**
  - `fedsteer/fed.py` `_check_coverage_config` now accepts `lora.offset` for `aligned` only; plus one new test, 75/75 pass.
  - It was made without a coordinator grant, but at the user's direct request. It is opt-in, because that config used to raise an error. Recorded as a closed grant.
  - Delta/dtai do not have it, and math does not need it.
- **Comparisons:** fed-aligned-off vs fed-aligned (NR-58), the primary one; vs fed-shared-off (NR-60); and whether nypost/theguardian's support_gap and disagreement fall relative to NR-58.
- **NR-60a `fed-shared-soff`, added by the user directly** (newsroom-iter1): g_{i,l}(α) = o + h_l(α), with one offset averaged by FedAvg. It uses the existing default path (no code change). cc 11209326, Anvil 21179398, watcher pid 2762675, ≈ 6–7 SU if Anvil wins.
  - The contrast with fed-shared-off isolates whether the offset is shared or private.
  - Prior evidence (entry 20, learned gain, one shape): a shared offset hurt (0.158 vs 0.151).
  - ⚠️ It runs in its own race, so it may land on Anvil while the rest of NR-60 ran on cc. Record the host in every table.
- **User correction:** the shared-offset arm builds on aligned calibration.
  - `fed-shared-soff` was cancelled while PENDING (cc 11209326, Anvil 21179398) and replaced by `fed-aligned-soff`: g_{i,l}(α) = o + h̄_l(α), with o averaged by FedAvg.
  - Needs a new opt-in flag `fed.shared_offset` (aligned only).
  - newsroom-iter1 holds a grant on `fedsteer/fed.py` and `tests/test_fedsteer.py` while it builds this. molecule and math may not sync code until it reports tests passing.
  - Contrast: fed-aligned-soff vs fed-aligned-off (NR-62); the two differ only in whether o is shared.
- The molecule pre-hold smokes 11206293 / 11206294 had already COMPLETED (17:09–17:29), before the 22:02 fed.py edit, so their results are valid.
  - Verified on GPU (Qwen3-4B): 252 per-layer shapes (36 layers × 7 modules); in fed-shared-off the server holds no o and the clients' o_i differ.
  - The local-off smoke is still to run once the user lifts the molecule hold.
- **NR-60b code landed:** `fed.shared_offset` (aligned + offset + fedavg only); 76/76 tests; defaults bit-identical on 5 paths; Anvil synced. The pending cc jobs never snapshotted mid-edit. Grant closed; code syncs released for molecule and math.

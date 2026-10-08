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

**Code constraints for the next design:** the offset sits in the `gain` group, so `calibration=shared` cannot keep it private; `coverage`/`consensus`/`aligned` reject `lora.offset=true`. *(Superseded: `fed.private_offset` (NR-60) for shared, and aligned accepts `lora.offset` (NR-62, private o_i by default; `fed.shared_offset` shares it). coverage/consensus still reject it.)*

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
- **NR-60c `fed-shared-soff`, resubmitted at the user's request** as an independent run beside NR-60b: g_{i,l}(α) = o + h_l(α), on the default shared path. cc 11209750, Anvil 21179756, watcher pid 2967789.
  - Contrasts: vs fed-shared-off, vs fed-shared (repeat of the entry-20 check), vs fed-aligned-soff.
  - Newsroom's Anvil exposure: ≈ 20 SU across NR-62, NR-60b and NR-60c if Anvil wins each race. NR-60 itself used 0. Within the 45 SU cap.
- NR-60b `fed-aligned-soff`: cc 11209468, Anvil 21179559, watcher pid 2807952, both PENDING. The 2 × 3 offset table (rows: shared h_l / aligned h̄_l; columns: none / private o_i / shared o) is adopted for the report.

## CAL-9. NR-63 (NR-60 fed arms, test): a private offset ties fed-shared on the mean and repairs private shapes; with s = 1 it shifts the range instead of extending it (2026-10-08)

- **Results** (cc, single seed, round 100), test pct err (worst client):
  - fed-shared 0.150 (0.204); fed-shared-off 0.150 (0.205); fed-private 0.158 (0.241); fed-private-off 0.150 (**0.192**); fed-aligned 0.153 (0.213).
  - shared-off − shared: better 1/8 (nypost −0.015), worse 1/8 (people +0.008); out-of-support worst client 0.184 vs 0.201.
  - private-off − private: better 2/8 (nypost −0.049, reuters −0.023), worse 1/8.
  - Reach 0.393 vs 0.407 and the α = 1 cell 0.121 vs 0.110 are both worse with the offset. The α = 0 gain seen on dev (round 50) is gone by round 70.
- **Learned o_i** are similar under shared and private shapes: nypost −0.66 / −0.51, reuters −0.38 / −0.44, mean −0.22 / −0.21. The level is identified independently of how the shape is learned.
- **Level vs shape:** in-support shape distance to the other clients falls 41% for nypost and 5% for reuters. Under fed-shared-off reuters still overshoots by +0.26 / +0.22 at α = 0.25 / 0.5 and only +0.07 at 0.75: a slope difference.
- **Coordinator reading:**
  1. With s = 1, g_{i,l}(α) = o_i + h(α) spans [o_i, o_i + 1]. A negative o_i lowers the top as much as the bottom, so it trades α = 1 for α = 0.
  2. A shared gain s cannot fix this, because it is absorbed into D: (o_i + s·h)D = (o_i/s + h)(sD).
  3. The copy-heavy clients need a lower coefficient at low α and the same at high α, i.e. a smaller slope, not a shift. A top-anchored form g_{i,l}(α) = o_i + (1 − o_i)·h(α) expresses that with the same single private parameter: g(0) = o_i, g(1) = 1.
  4. The general form with two private endpoints is a private gain, s_i = 1 − o_i → free. A private gain fitted only in support has failed before (entry 41, MOL-15), but always together with private shapes, never under a shared shape.
- Mean differences among the top designs are ≤ 0.003 at one seed, so worst-client and endpoint metrics carry the comparison. Seeds are needed before the design is fixed.
- Still pending: the local arms (local-off running, local queued; these are the "better than local" bar), NR-62, NR-60b, NR-60c.

## CAL-10. NR-63a needed-coefficient check: clients need a wider range (b ≈ 1.4–1.9), mostly common to all clients (2026-10-08)

- **Method (newsroom-iter1, fed-shared-off r100):** invert each client's response p(c) at α = 0.25 / 0.5 / 0.75, then fit the needed c* with three forms:
  - (a) shift, o_i + h;
  - (b) top-anchored, o_i + (1 − o_i)·h;
  - (c) a + b·h.
- **Result:**
  - (c) fits every client with rmse 0.02–0.03.
  - (a) fits only forbes (b = 1.03).
  - The other clients need b = 1.38–1.91: reuters a = −1.12, b = 1.77; nypost a = −1.08, b = 1.38.
  - (b) and (c) are not separable with this grid, because α = 1 is never reached.
- **Coordinator reading:**
  - Most of the needed slope is **common**: median b ≈ 1.6. The client-specific part is a factor of about 0.65–1.2 around it.
  - A common factor is in principle absorbable into D. But D at s = 1 has not grown that far by round 100, while the old learned shared gain reached s = 1.93 and had the best reuters (0.199). So in practice a learned shared gain matters (NR-52 point 5 again).
  - **Caveat:** c* is a post-hoc inversion at a frozen P_i (an oracle remap in coefficient space). Joint training could settle elsewhere, and the top end is extrapolated. A direct coefficient sweep (GPU eval only) would measure reach beyond the trained range.
- **Candidate forms** (all with a shared per-layer shape):
  - (A) learned shared s plus private o_i: runnable with existing flags (`fed.calibration=shared fed.fix_gain=false lora.offset=true fed.private_offset=true`).
  - (B) private s_i plus private o_i: needs a new flag to keep s private under a shared shape.
  - (C) s_i = s·exp(δ_i), with the shared s and a private δ_i pulled toward 0: the consensus principle applied to the gain. Needs code.
- Asked the user to choose.
- **User decision (2026-10-08): the gain stays fixed at 1.** None of the CAL-10 gain designs (A/B/C) will be tested for now.

## CAL-11. NR-65: fed-aligned-off is the best Newsroom design so far; Anvil environment broken; NR-64 τ sweep (2026-10-08)

- **NR-65 (newsroom-iter1, all cc, single seed).** 2 × 3 grid, test pct err (worst client):

| Shape | no offset | private o_i | shared o |
|---|---|---|---|
| shared h_l | 0.150 (0.204) | 0.150 (0.205) | running (NR-60c) |
| aligned h̄_l | 0.153 (0.213) | **0.146 (0.199)** | 0.150 (0.210) |

  - Local baselines: local 0.157 (0.244); local-off 0.178 (0.266).
  - **fed-aligned-off**, i.e. g_{i,l}(α) = o_i + h̄_l(α):
    - paired vs fed-shared: better 1/8 (nypost −0.012), worse 0/8;
    - vs fed-aligned: 2 better / 0 worse;
    - vs fed-shared-off: 3 better / 0 worse;
    - vs local: 4 better / 0 worse;
    - best worst client, best out-of-support worst client (0.184), best reach (0.425).
    - Mechanism: o_i fixes aligned's low end on the copy-heavy clients (nypost's α = 0 output percentile 0.244 → 0.168), while h̄_l keeps aligned's top-end range.
  - o_i is the same in all three fed private-offset arms (nypost ≈ −0.62 to −0.66, reuters ≈ −0.38).
  - A shared o settles at −0.037 (private mean −0.21). aligned-soff vs aligned-off is worse on nypost (+0.019) and reuters (+0.011). **The level must be private.**
  - local-off is worse than local on 7/8 clients: o_i drifts upward to a mean of +0.78, because with a private D_i it is redundant. **The offset is a fed-only component; the local reference is `local`.**
  - **On Newsroom this meets the user's bar** (beats local and plain fed-shared, worse on no client), with the CAL-5 design. Single seed: seed replicates of fed-aligned-off, fed-shared and local are needed before claiming it.
- **NR-64 (user's direct request):** pooling-weight sweep on fed-aligned *without* offset, τ_pool ∈ {300, 1000, 3000} plus count pooling. Diagnostic for the saturating-weight loss; now that o_i fixes that loss, its bearing on the final design is indirect.
- **Anvil environment broken:**
  - `/anvil/scratch/x-zchen17/lucmon/envs/rein` has lost stdlib files (probably a scratch purge). Every Anvil job since 03:09 fails in seconds, at 0 SU.
  - ~~race_watch counts a FAILED copy as started~~ **Corrected:** `STARTED` (race_watch.py:31) is unused. A copy wins only if it has been RUNNING for at least `--min_running_s` or is COMPLETED (lines 103–111: "a copy that FAILED early must not win"), so a failed Anvil copy cannot cancel cc copies. Residual risk: the watcher exits early once only one copy is live.
  - Molecule and Newsroom cannot use Anvil until the environment is rebuilt (user decision; preferably outside scratch).
- Spec note for fed-aligned-off on any dataset: pass `lora.offset=true` only. Under aligned, o_i is private by default, and `fed.private_offset` would be a silent no-op (it applies only to calibration=shared).

## CAL-12. MATH-18 (per-layer shapes, single client): per-layer helps reach, α = 0 level unmoved, gain clamp not binding (2026-10-08)

- Single client `math`, answer in prompt, learned gain, warp_reg 0.01, no offset. Test pct err:
  - local-gain-1shape 0.253 (s clamped at 4);
  - **local-gain (per-layer) 0.213**, Spearman 0.870, reach 0.345;
  - local-gain-gm16 0.225 (s plateaus near 5.1, so the clamp was not binding).
- Per-layer shapes raise the α = 1 mean percentile from 0.75 to 0.87. The α = 0 output stays at about 0.22, because g_l(0) = 0: the MATH-13 level problem.
- Host caveat: G2 on Delta (torch 2.5.1), P/PG on dtai (torch 2.13).
- E1 dev at round 100: fed-shared-gain-1shape 0.256 vs local-gain-1shape 0.253; test analysis pending (MATH-20).
- **Tension with the cross-dataset design.** The user fixed s = 1, but math learns s ≈ 3–5 (shared 3.17 in E1-fed). At s = 1, D must grow about 3–4× by itself.
  - On Newsroom, fixing s cost little only once per-layer shapes were added (NR-57).
  - Whether math tolerates s = 1 is untested.
  - It is the first thing to check before porting fed-aligned-off to math: single-client `local` (per-layer shapes, s = 1) vs MATH-14 P (s learned).

## CAL-13. NR-66: τ_pool sweep on fed-aligned-off, launched at the user's request (2026-10-08)

- cc only: tp300 11215856, tp1000 11215857, tp3000 11215858, cnt 11215859. τ = 100 is NR-62.
- Report: a 2 × 5 grid (fed-aligned / fed-aligned-off × τ ∈ {100, 300, 1000, 3000, count}), with nypost/reuters α = 0 / 0.25 cells and o_i per τ.
- Runs land on different GPU types (ccc0284; ccc0465 H200; ccc0390 A100-80GB). With one seed, GPU-type nondeterminism adds noise comparable to seed noise, so the τ choice (on dev) should be confirmed with a second seed before it enters the design.

## CAL-14. MATH-20: E1 (answer in prompt, learned gain, one shape) — fed ties local (2026-10-08)

- Test pct err: fed-shared-gain-1shape 0.255 (worst 0.320) vs local-gain-1shape 0.257 (0.323).
  - Fed better on 5/8 clients, all gaps within ±0.05.
  - Spearman 0.789 vs 0.763; reach 0.169 vs 0.201; accuracy 0.942 vs 0.945.
- Error sits at the endpoints: |pct − α| ≈ 0.31 at α = 0 and 0.32–0.37 at α = 1, against 0.13–0.20 in the interior. The α = 0 → 1 span is only ≈ 0.37 of the scale.
- The α = 0 level varies 0.03–0.62 by client (olympiads 0.60), so the offset o_i is the expected lever, as on Newsroom.
- Proposed s = 1 check (not launched, user decides), single client on dtai so it is host-matched with MATH-14 P:
  - (i) s = 1 with warp_reg 0.01, to isolate s;
  - (ii) s = 1 with warp_reg 0, the reference design.

## CAL-15. NR-67: higher τ_pool fixes most of aligned's loss; a shared offset slightly hurts under a FedAvg shape; quality is flat (2026-10-08)

- **NR-64 τ sweep** (fed-aligned, no offset), test pct err (worst client): τ = 100 0.153 (0.213); 300 0.150 (0.210); 1000 0.151 (0.213); **3000 0.147 (0.202)** (H200 node); count still running.
  - τ = 3000 vs τ = 100: better 3/8 (nypost −0.025, reuters −0.011, people −0.009), worse 0/8.
  - Best α = 1 cell (0.091) and the highest reach of any run (0.435).
- **Re-weighting vs a private level, α = 0 output percentile** (τ = 100 → re-weighted → private o_i):
  - nypost: 0.244 → 0.171–0.206 with re-weighting → 0.168 with o_i.
  - reuters: 0.275 → 0.254 → 0.229.
  - They are partly substitutes for nypost and complements for reuters. NR-66 tests the combination.
- **NR-60c fed-shared-soff** 0.153 (0.222), shared o = −0.026: worse than fed-shared on 2/8 (reuters +0.018), better on none.
  - Offset grid, test pct err:
    - FedAvg shape: no offset 0.150 / private 0.150 / shared 0.153;
    - aligned τ = 100: no offset 0.153 / private 0.146 / shared 0.150.
  - **A private level is the only offset worth keeping.**
- **Quality** is flat across the fed arms: AlignScore, BERTScore, judge and repetition all within noise.
  - The only cost that scales with range is the out-of-support length gap: aligned without offset +3.8 to +5.1 words, fed-aligned-off +3.4, fed-shared-off +2.2, local +1.4.
  - Local has the weakest ordering (worst-client Spearman 0.805) and a negative out-of-support relevance gap.
- **Reading:** pooling weights should move toward data-proportional (the saturating τ = 100 under-weights the clients that own a region, NR-59). The FedAvg shape gets this implicitly.
  - Leading design candidates, all single seed: fed-aligned-off (0.146) and fed-aligned-tp3000 (0.147); NR-66 will show tp3000 + o_i.
  - Seeds are required: seed-1 replicates of fed-aligned-off, fed-aligned-tp3000 (on ccc0284), fed-shared and local; not launched.

## CAL-16. NR-68 assigned: house style and held-out clients (E2) for the leading Newsroom designs (2026-10-08)

**User request:** complete house-style and held-out-client results, only for `fed-aligned-tp3000` (NR-64), `fed-aligned-off` (NR-62), `local-off` (NR-60) and `fed-shared` (NR-54 A_L).

- **House style** (`scripts/style_eval.py`, CPU, on saved test evals): runs as is on all four, with the earlier ceiling (real summaries) and floor (A2) for reference.
- **E2:** `e2_heldout.py` refuses aligned runs, and it trains the gain whenever it trains calibration (the offset sits in the `gain` group; `fix_gain` is ignored). Shared-code grant to newsroom-iter1 (`e2_heldout.py`, `fedsteer/adapt.py`, tests). Protocol for a new client:
  - `fed-aligned-*`, frozen_D: frozen D and frozen h̄_l (the run's inference table). Train P; with `lora.offset`, also train the client's own o_i from 0 (u fixed).
  - `fed-shared`, frozen_D: unchanged path (frozen shared calibration), checked with per-layer shapes.
  - `local-off` reference = the run-independent `local_D` setting made to match the local-off arm: gain fixed at 1, per-layer shapes, offset trained, D_i from zero. Run once per rotation. Plain `local_D` without offset is optional.
  - plugin: with an offset, fit only o_i on the n pairs. prompt: reuse the existing rotation-0 results if their settings match.
  - Same n grid and settings as entry 30 (n ∈ {16, 64, 256, 1024, all}, rotation 0, 200 test articles per held-out client), plus the LLM judge on frozen_D / local_D.
- **User correction to NR-68 (2026-10-08):**
  - Replace `fed-shared` with the two offset variants under a FedAvg shape: `fed-shared-off` (private o_i, NR-60, 11206103) and `fed-shared-soff` (shared o, NR-60c, 11209750).
  - `local-off` means: held-out clients train *local* models with the offset (local_D with offset, gain 1, per-layer shapes), as already planned.
  - Final arm set: fed-aligned-tp3000, fed-aligned-off, fed-shared-off, fed-shared-soff, plus the local-off reference. Plain local_D (no offset) is dropped.
  - E2 needs the `private_offset` path as well: frozen h_l from the server; the new client trains P and its own o_i from 0, with u fixed. `fed-shared-soff` uses the existing shared path, with the server's o frozen.

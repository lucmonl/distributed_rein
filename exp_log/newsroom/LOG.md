# Experiment log: newsroom

Updated after every code change and every job; newest entries at the bottom. Entry IDs: `NR-<n>`, starting at NR-60.
Conventions (paths, terminology, job names): `../../CONVENTIONS.md`. History before 2026-10-07: `../EXPERIMENT_LOG.md` (archive, read-only).


## NR-60. Private offset under a shared shape, fed and local, plus a local Adam-reset control (2026-10-07)

Assigned by the coordinator (CAL-2 + addendum). Matched to NR-54.

### Shared code change (grant: `fedsteer/fed.py`, `fedsteer/lora.py`, `e2_heldout.py`, `e3_drift.py`, `tests/test_fedsteer.py`)

Two opt-in flags in `FedConfig`; with both at their default (`false`), behaviour is unchanged.

- **`fed.private_offset`** (default `false`). With `calibration=shared` and `lora.offset=true`, the offset o stays per client while the gain and shape stay shared:
  - o is removed from `shared_params`, so its Adam state is not reset each round, and it is removed from the server state, so it is never averaged.
  - Each client stores only `{steer_control.o}` in `clients[c]["gain"]` (snapshots and `state.pt` carry it). `load_client` loads it on top of the server's shape and gain.
  - `load_snapshot_into` restores the client's own o after the shared calibration when the snapshot's `fed_config.private_offset` is true.
  - No-op with `calibration=private` and in local mode. Raises an error if `lora.offset=false`. coverage/consensus/aligned still reject offsets.
- **`fed.local_reset_opt_state`** (default `false`). In local mode, `_load_opt` drops the Adam state of `shared_params` at every round start, as fedavg does with `reset_shared_opt_state`. In local mode with a private adapter, `shared_params` is exactly the client's own direction D_i. The adapter, shape and offset states persist. No-op in fedavg.
- `e2_heldout.py` refuses `private_offset` runs with shared calibration, because a new client would need its own offset. `e3_drift.py --refit_k` refuses them too, because a refit would also refit the shared calibration.
- `fedsteer/lora.py`: no change needed (`OFFSET_KEY` already existed).
- **Tests, 3 new** (`tests/test_fedsteer.py`):
  - `test_private_offset_with_shared_shape`:
    - o is not in `shared_params` or the server, so it is never averaged.
    - Clients hold only o; the offsets are trained after the gain warm-up and differ across clients.
    - o's Adam state survives the round reset, while the shape's state is reset.
    - Loaded clients share shape and gain but keep their own o.
    - The snapshot → `load_snapshot_into` round trip restores the shared shape plus each client's o.
    - Resume equals an uninterrupted run, offsets included.
  - `test_private_offset_is_a_noop_where_the_offset_is_already_private`: fedavg and local with `calibration=private` give bit-identical results with the flag on or off. The flag without `lora.offset` raises. The default shared path still shares o.
  - `test_local_reset_opt_state_resets_only_the_direction`: with the flag, local mode starts every round with D_i's state empty, while the adapter, shape and offset states persist. Without the flag D_i's state is kept. In fedavg the flag changes nothing (bit-identical); in local mode it changes D_i.
  - **Full suite: 74/74 pass.**
- **The defaults reproduce the old code exactly.** The pre-edit `fed.py` was rebuilt in a scratch copy. Four configs were run for 3 rounds under the old and the new code: fed-shared-off (offset shared), fed-shared, fed-private-off and local-off, all with per-layer shapes. Server, client states, per-round logs and snapshot-loaded models are **bit-identical**.
- **CPU smoke** (`train_fed.py`, Newsroom, Llama-3.2-1B, 3 rounds × 1 step, 8 examples/client, monitor off):
  - `fed.calibration=shared lora.offset=true fed.private_offset=true`: the server has the shape and u but no o; the 8 client offsets differ.
  - `fed.mode=local ... fed.local_reset_opt_state=true`: runs. The config.yaml of both runs records the flags.
- **Code synced to Anvil** with `scripts/setup_anvil.sh --code-only`. Anvil's `fedsteer/*.py` md5s equal cc's, and a follow-up rsync dry run lists no file differences.

### Log paths (CONVENTIONS §2)

The newsroom sbatch files now write to `sbatch/logs/newsroom/%x.o%j`:
- cc: `train_eval`, `eval_settings`, `eval_baselines`, `judge_evals`, `eval_pilot`, `newsroom_pilot`, `sweep_pilot`.
- Anvil: `train_eval_anvil`.

`GPU_LOG` now writes to `sbatch/logs/newsroom/gpumem.o<jobid>`. `sbatch/logs/newsroom/` exists on both hosts.

### Arms and jobs

Launch: [NR-60.sh](launch/NR-60.sh). Common to all arms: `model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0`, rotation 0, configs/newsroom_fedavg.yaml. Pipeline: train → dev selection → test (200 prompts, density) → quality → judge. The offset trains from the gain warm-up (round 2) although the gain is fixed; `offset_max` = 2.

| Arm (= job name = `runs/<name>_…`) | Overrides | cc | Anvil |
|---|---|---|---|
| `nr60-fed-shared-off` | `fed.calibration=shared lora.offset=true fed.private_offset=true` | 11206103 | 21174317 |
| `nr60-fed-private-off` | `fed.calibration=private lora.offset=true` | 11206104 | 21174318 |
| `nr60-local-off` | `fed.mode=local fed.calibration=private lora.offset=true` | 11206105 | 21174319 |
| `nr60-local` | `fed.mode=local fed.calibration=private lora.offset=false` | 11206106 | 21174320 |
| `nr60-local-off-reset` | `fed.mode=local fed.calibration=private lora.offset=true fed.local_reset_opt_state=true` | 11206107 **cancelled** | 21174353 **cancelled** |
| `nr60-fed-shared-soff` (added 21:52, see NR-60a) | `fed.calibration=shared lora.offset=true fed.private_offset=false` | 11209326 **cancelled** | 21179398 **cancelled** |
| `nr60-fed-shared-soff` (resubmitted 22:55, see NR-60c) | `fed.calibration=shared lora.offset=true fed.private_offset=false` | 11209750 | 21179756 |
| `nr60-fed-aligned-soff` (added 22:10, see NR-60b) | `fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01 lora.offset=true fed.shared_offset=true` | 11209468 | 21179559 |

- **Race:** one group `NR60`. Spec [NR-60_race.json](launch/NR-60_race.json); watcher pid 1157389; log `sbatch/logs/newsroom/race-NR-60.log`.
- **cc partitions:** `dali,IllinoisComputes-GPU,scavenger`, excluding `ccc0387,ccc0089,ccc0090,ccc0232-0236`. This matches NR-54 except that the race spec does not mark scavenger as preemptible.
- **Anvil budget:** the balance was 473.1 / 500 SU before submission. NR-54's arms took about 6.1 h each on cc's A100s, so if Anvil wins, five arms cost about **31–35 SU**, within the 45 SU the coordinator allowed. Losing copies are cancelled while still PENDING.
- **Cancelled by user decision, 2026-10-07 16:58 CDT: `nr60-local-off-reset`** (cc 11206107 and Anvil 21174353, both while PENDING; done by the coordinator, CAL-6).
  - The user's ruling: the per-round Adam reset is part of FedAvg, not a confound. FedAvg overwrites the shared parameters every round, so resetting their moments is the protocol.
  - The reset control is dropped on all datasets. `fed.local_reset_opt_state` stays in the code (default false) but is unused.
  - NR-60 continues with 4 arms. In tables this arm is "cancelled", not missing.
- **Race decided for cc (16:58–16:59).** `fed-shared-off` and `fed-private-off` started on ccc0284, and the watcher then cancelled all four remaining Anvil copies (21174317–20) while they were PENDING.
  - It skipped the already-cancelled pair ("state is CANCELLED"), as in NR-55.
  - `local-off` and `local` are pending on cc. Anvil cost: 0 SU.
- **Archive names cited below:** `fed-shared` = NR-54 A_L (`nr54_layer_shared`, 11190971; test 0.150, worst 0.204); `fed-private` = NR-54 B_L (`nr54_layer_private`, 11190972; 0.158, worst 0.241).

### Analysis plan (rewritten after the cancellation; supersedes the one written at launch)

Notation (CONVENTIONS §3): g_{i,l}(α) = o_i + s · h_{i,l}(α), with s = 1 throughout. `fed-shared(-off)` has one h_l per layer shared by all clients. The private arms have h_{i,l}. `-off` adds a private o_i (one scalar per client).

**Contrasts:**
1. **`fed-shared-off` vs `fed-shared`** (primary): does a private level o_i help on top of a shared shape h_l?
2. `fed-private-off` vs `fed-private`.
3. `local-off` vs `local`.
4. `fed-shared(-off)` vs `local(-off)`: C1 with a matched local baseline (gain 1, per-layer shapes) for the first time.
5. `fed-private-off` vs `local-off`: same calibration, only the direction's training differs (FedAvg D vs. D_i); the same contrast as molecule's MOL-22.

**Fed vs. local is the intended comparison, with fed as FedAvg is defined.** FedAvg's per-round reset of the direction's Adam state is part of the protocol (user, CAL-6). If fed reaches farther than local, the reset can be named as one mechanism of that reach advantage (MOL-23), alongside the averaging itself.

**Per client, report:**
- pct err in and out of support;
- per-α-cell error, with the endpoints α = 0 and α = 1 shown separately (NR-53: endpoints were 43% of `fed-shared`'s error);
- the output percentile at α = 0;
- the learned o_i (from the snapshot and the train log);
- quality (AlignScore, BERTScore) and the judge.
- Call out nypost.com, reuters.com and theguardian.com.

**Checks after round 1:** each run's `config.yaml` should show `warp_scope: module`, `warp_reg: 0`, `fix_gain: true`, and per arm `offset`, `private_offset`, `mode` and `local_reset_opt_state` (false in all four remaining arms) as in the table.

**Round-1 config check (17:10, the two started arms):** `nr60-fed-shared-off_20261007-165434_j11206103` and `nr60-fed-private-off_20261007-165435_j11206104` both have `warp_scope: module`, `warp_reg: 0`, `fix_gain: true`, `offset: true`, `local_reset_opt_state: false`, rounds 100 and 4000 per client.
- `private_offset` is true for fed-shared-off (calibration shared) and false for fed-private-off (calibration private), as intended.
- o_i = 0 for all clients in rounds 0–1, as expected: the offset starts training at round 2, after the gain warm-up.
- `local-off` and `local` are still pending; check them when they start.

## NR-61. `fed-aligned` (NR-58) test results: closes consensus's train/inference gap but does not beat `fed-shared`; NR-60 interim at dev round 50 (2026-10-07, 20:30)

Analysis only; no code or jobs.

**Names** (CONVENTIONS §3–4; notation g_{i,l}(α) = o_i + s · h_{i,l}(α), s = 1 and no offset in all five runs, so the coefficient at α = 0 is exactly 0):

| Arm | Archive name | Job | Calibration at inference |
|---|---|---|---|
| `fed-shared` (reference) | NR-54 A_L | 11190971 | h_l, FedAvg of the shape parameters |
| `fed-private` | NR-54 B_L | 11190972 | h_{i,l} |
| `fed-consensus-l0p1` | NR-56 consensus λ = 0.1 | 11191900 | ḡ_l (training used h_{i,l}) |
| `fed-consensus-l1` | NR-56 consensus λ = 1 | 11191901 | ḡ_l (training used h_{i,l}) |
| `fed-aligned` | NR-58 `nr58_aligned_l0p01` | 11203892 | ḡ_l, which training also used |

Reports:
- `reports/nr61_aligned_test.txt` (`compare_runs.py`, 200 test articles per client, paired bootstrap over articles);
- `reports/nr61_aligned_cells_test.txt` (per-α-cell error and output percentile per client; computed from the eval grids with a scratch script, cell means reproduce `pct_calib_err`).

All runs are dev-selected at round 100, single seed.

### Test results

| Arm | Pct err (worst) | In-support | Out-of-support (worst) | Reach | α = 0 cell | α = 1 cell | nypost.com | reuters.com | theguardian.com |
|---|---|---|---|---|---|---|---|---|---|
| **`fed-shared`** | **0.150 (0.204)** | **0.134** | 0.157 (0.201) | 0.407 | **0.198** | 0.110 | **0.187** | **0.204** | 0.129 |
| `fed-private` | 0.158 (0.241) | 0.136 | 0.168 (0.269) | 0.406 | 0.236 | 0.103 | 0.241 | 0.213 | 0.125 |
| `fed-consensus-l0p1` | 0.158 (0.225) | 0.141 | 0.165 (0.244) | 0.389 | — | — | 0.225 | 0.210 | 0.135 |
| `fed-consensus-l1` | 0.156 (0.226) | 0.141 | 0.164 (0.217) | 0.400 | 0.216 | 0.103 | 0.202 | 0.226 | 0.129 |
| `fed-aligned` | 0.153 (0.213) | 0.138 | 0.157 (0.228) | **0.421** | 0.212 | **0.094** | 0.209 | 0.213 | **0.124** |

Paired per-client contrasts (95% bootstrap CI excludes 0):

| Contrast | Better on | Worse on | Detail |
|---|---|---|---|
| `fed-aligned` − `fed-shared` | 1/8 (aol.com −0.007) | **3/8** (nypost.com **+0.022**, reuters.com +0.009, forbes.com +0.006) | Out of support: better on aol.com (−0.021), wsj.com (−0.011) and theguardian.com (−0.007); worse on nypost.com (+0.027) |
| `fed-aligned` − `fed-consensus-l1` | **4/8** (reuters.com −0.013, aol.com −0.009, cbc.ca −0.007, wsj.com −0.007) | 0/8 | |
| `fed-aligned` − `fed-consensus-l0p1` | 4/8 | 0/8 | |
| `fed-aligned` − `fed-private` | 3/8 (nypost.com −0.032) | 1/8 (forbes.com +0.005) | |

**Quality.**
- AlignScore in support is 0.791 vs. 0.781 for `fed-shared`; BERTScore is the same.
- Out-of-support summaries are longer than the reference (gap +4.9 words vs. +3.2).
- The judge's out-of-support faithfulness is 0.815 vs. 0.779 (subsample, no CI). Everything else is within noise.

### What the results say

**1. Aligned does what it was built for: it removes consensus's train/inference mismatch.**
- `fed-aligned` beats both consensus arms on 4/8 clients and loses on none.
- The largest gains are on reuters.com and on the broad clients' out-of-support cells.
- `fed-consensus-l1` still paid for the mismatch plus a tight tie; aligned pays for neither.

**2. It still does not beat `fed-shared`, and it loses where NR-59 predicted: the copy-heavy clients.**
- nypost.com +0.022 and reuters.com +0.009 are the same two clients that `fed-consensus-l1` lost on (+0.015, +0.022).
- Diagnostics from the training log:
  - **The isotonic projection never acted** (`proj_layers` = 0 and `proj_adjust_max` = 0 in all 100 rounds). In practice, then, ḡ_l was simply the saturating-weighted average of the clients' h_{i,l}.
  - The tie is negligible (≤ 8·10⁻⁵).
  - The clients' curves drifted apart (disagreement 0.007 at round 5, 0.036 at round 60, 0.028 at round 100). nypost.com and theguardian.com are farthest from ḡ in their own support (`support_gap` 0.051 and 0.045; the others 0.014–0.017). nypost.com's own curve is the lowest: mean over layers 0.48 / 0.62 / 0.77 at α = 0.4 / 0.6 / 0.8, against 0.53–0.60 / 0.67–0.74 / 0.81–0.87 for the others.
- So `fed-aligned` and `fed-shared` differ mainly in **how the clients' curves are pooled**:
  - `fed-shared`: FedAvg of the shape parameters, where each client pulls by the size of its gradient, i.e. implicitly in proportion to where it actually generates.
  - `fed-aligned`: saturating weights c/(τ + c), which cap the copy-heavy clients' say at high α (NR-59: about 30% of the weight against 54–66% of the nearby data).

**3. Where the error moves (per-α cells, test):**

| α cell | `fed-shared` | `fed-aligned` | nypost.com (shared → aligned) | reuters.com (shared → aligned) |
|---|---|---|---|---|
| 0 | 0.198 | 0.212 | 0.190 → 0.244 | 0.247 → 0.275 |
| 0.25 | 0.165 | 0.175 | 0.272 → 0.352 | 0.299 → 0.317 |
| 0.5 | 0.148 | 0.155 | 0.231 → 0.236 | 0.233 → 0.243 |
| 0.75 | 0.127 | 0.127 | 0.129 → 0.134 | 0.134 → 0.136 |
| 1 | 0.110 | **0.094** | 0.110 → 0.078 | 0.106 → 0.093 |

- Aligned **gains at the top**: the α = 1 cell drops by 0.016, with aol.com reaching 0.932 instead of 0.891, and reach is 0.421 vs. 0.407.
- It **loses at the bottom**: the α = 0 and α = 0.25 cells, almost entirely on nypost.com and reuters.com, which sit outside their support there (nypost.com [0.51, 0.95], reuters.com [0.10, 0.97]).
- The coefficient at α = 0 is 0 in both arms, so **the α = 0 cell is the adapter P_i alone**. P_nypost trained under ḡ_l writes summaries that sit higher at α = 0 than under h_l: output percentile 0.244 vs. 0.190 (target 0). That is much less than under its own h_{i,l} in `fed-private` (0.440), but it is the same failure in the same direction: the adapter absorbs part of the copy-heavy style when the calibration that client trains under is not its own.
- The endpoints remain about 41% of the error in every arm (NR-53). Neither aligned nor consensus moves the α = 0 cell, which no shape can reach. That is NR-60's question.

**4. Conclusion for the "how the shared function is learned" row of CAL-1.** Test pct err on Newsroom (per-layer shapes, gain 1, no offset):

| Design | Pct err |
|---|---|
| `fed-shared` | **0.150** |
| `fed-borrow-l1` | 0.151 |
| `fed-aligned` | 0.153 |
| `fed-borrow-l0p1` | 0.154 |
| `fed-consensus-l1` | 0.156 |
| `fed-consensus-l0p1` | 0.158 |
| `fed-private` | 0.158 |
| `fed-borrow-l0p01` | 0.159 |

(The borrow arms are NR-54's C_L arms.)
- None of the pooled-function designs beats FedAvg of the shape parameters.
- Aligned is the best design that infers with a pooled ḡ_l.
- The remaining gap to `fed-shared` is consistent with the pooling weights, not with the train/inference coupling.
- **The decisive next arm, if this line continues, is `fed-aligned-cnt`** (count pooling, `fed.cov_pool=count`), as NR-59 proposed: it gives nypost.com and reuters.com their data share of ḡ_l at high α. Risk: nypost.com would dominate ḡ_l there and could flatten it for everyone.
- Not launched; this is a coordinator decision.
- The dev trajectory adds a caution: `fed-aligned` was at or below `fed-shared` on dev at rounds 50–90 (e.g. 0.156 vs. 0.160 at 70, 0.163 vs. 0.169 at 80) and behind only at round 100 (0.152 vs. 0.147). Selection is on one checkpoint and one seed, so a 0.003 test difference is within what a second seed could move.

### NR-60 interim (dev, still training; no conclusions yet)

State at 20:05: `fed-shared-off` and `fed-private-off` are at about round 55, `local-off` about round 35, and `local` is still pending.

Dev pct err (mean over clients; nypost.com, reuters.com in brackets), at matched rounds:

| Round | `fed-shared` | `fed-shared-off` | `fed-private` | `fed-private-off` | `local-off` |
|---|---|---|---|---|---|
| 10 | 0.245 | 0.234 | 0.242 | 0.237 | 0.251 |
| 20 | 0.202 | 0.194 | 0.206 | 0.199 | 0.208 |
| 30 | 0.195 | 0.188 | 0.204 | 0.195 | 0.199 |
| 40 | 0.191 | 0.187 | 0.200 | 0.193 | — |
| 50 | 0.176 (0.189, 0.254) | **0.173** (0.192, 0.233) | 0.181 (0.240, 0.252) | 0.175 (0.210, 0.240) | — |

- **Both offset arms lead their no-offset counterparts at every round so far.** `fed-shared-off`'s lead shrinks (−0.011 at round 10, −0.003 at round 50); `fed-private-off`'s holds (−0.005 to −0.009).
- **The offsets do what NR-53 asked for: they lower the α = 0 level.**
  - At round 50 every o_i is negative.
  - The copy-heavy clients have the largest: under `fed-shared-off`, nypost.com −0.63, reuters.com −0.42, theguardian.com −0.10, the others −0.16 to −0.19. `fed-private-off` is similar (−0.51, −0.42).
  - On dev the α = 0 cell drops from 0.260 to 0.240 (`fed-shared` → `fed-shared-off`), almost all of it on reuters.com (0.399 → 0.304). With `fed-private-off` the cell drops from 0.284 to 0.258, mostly on nypost.com (0.378 → 0.272) and reuters.com (0.417 → 0.345).
  - So the private level does reach the cell that no shape can.
- **`local-off` vs. `fed-private-off`** (round 30) is level overall: 0.199 vs. 0.195, with the α = 0 cell 0.296 vs. 0.297.
  - `local-off`'s offsets are mostly positive (theguardian.com +0.27, aol.com +0.14), with nypost.com +0.01 and reuters.com −0.01.
  - Each local direction D_i has its own scale and sign, so local and fed offsets are not comparable in size.
- The final comparison follows the NR-60 plan once all four arms finish. The federated arms should finish around 23:00; `local` has not started.

**Coordinator follow-up (CAL-7).**
- Do **not** launch `fed-aligned-cnt`: the user is designing the next aligned variant. The coordinator's candidate is `fed-aligned-off` (pooled h_{i,l} with a private o_i), which needs an extension of aligned mode that has not been started.
- The NR-60 report must add:
  1. the test table and paired contrasts;
  2. o_i at the selected round;
  3. whether nypost.com's and reuters.com's h_{i,l} at α = 0.25 / 0.5 / 0.75 move toward the other clients' in `fed-private-off`, compared with `fed-private`;
  4. per-α cells, plus quality and judge in and out of support at α = 0.

**Preview of item 3 at matched round 50** (`fed-private-off` still training; a scratch script reads h_{i,l} from the snapshots). The figure is the mean over the 112 layers of |h_{i,l}(α) − mean of the six other clients' h_{·,l}(α)| at α = 0.25 / 0.5 / 0.75:

| Client | `fed-private` r50 | `fed-private-off` r50 | o_i (`-off`) |
|---|---|---|---|
| nypost.com | 0.257 / 0.328 / 0.313 | **0.201 / 0.234 / 0.194** | −0.51 |
| reuters.com | 0.118 / 0.121 / 0.097 | **0.106 / 0.098 / 0.065** | −0.42 |
| the other six clients (range) | 0.060–0.133 at α = 0.5 | 0.048–0.101 at α = 0.5 | −0.13 to −0.22 |

- With a private level, nypost.com's shapes move about a third of the way toward the others', and reuters.com's move closer at α = 0.75. Mean h at 0.5: nypost.com 0.268 → 0.365, reuters.com 0.480 → 0.530; the others 0.57–0.68.
- **Consistent with "mostly level, partly shape"**: nypost.com's shape is still the most distinct by a wide margin.
- To be redone at the selected round.

**Item 3, revised as the coordinator asked (CAL-7): in-support distances only.** nypost.com's support is [0.51, 0.95], so the α = 0.25 and 0.5 values in the table above are unconstrained extrapolations; that table is superseded by this one. Same measure at round 50: the mean over 112 layers of |h_{i,l}(α) − mean of the non-focus clients' h_{·,l}(α)|. "Pooled" averages over the in-support points of the grid 0.05, 0.10, …, 0.95:

| Client (support) | `fed-private` r50: d(0.25) / d(0.5) / d(0.75); pooled | `fed-private-off` r50: d(0.25) / d(0.5) / d(0.75); pooled | o_i |
|---|---|---|---|
| nypost.com [0.51, 0.95] | out / out / 0.313; **0.289** | out / out / 0.194; **0.180** | −0.51 |
| reuters.com [0.10, 0.97] | 0.118 / 0.121 / 0.097; **0.104** | 0.106 / 0.098 / 0.065; **0.081** | −0.42 |
| the six others, pooled | 0.046–0.106 | 0.046–0.104 | −0.13 to −0.22 |

- In support, a private level reduces the shape distance by about 38% for nypost.com and 22% for reuters.com, and leaves the others unchanged.
- nypost.com is still about twice as far from the others as the most distinct of them (0.180 vs. ≤ 0.104).
- **Reading:** partly level, still largely shape (the coordinator's wording).
- Final version at the selected round. `fed-shared-off` will be reported as the response gap: each client's output percentile at the same α, which is what a shared h_l plus o_i leaves unexplained.

## NR-62. `fed-aligned-off`: aligned calibration with a private offset, launched (2026-10-07, 21:45)

**Request (user, directly):** "kick off the run on aligned calibration for nr60".
- Plain `fed-aligned` in the NR-60 setting already exists: NR-58 is matched to NR-54 and NR-60.
- So the run is the aligned counterpart of NR-60's private offset, **`fed-aligned-off`**: the coordinator's CAL-7 candidate, which it had described as not yet started.
- Not `fed-aligned-cnt`, which stays on hold.

**Model.** g_{i,l}(α) = o_i + h̄_l(α). With the offset outside the pool, the pooled object is a shape, so it is written h̄_l (CONVENTIONS §3; coordinator, CAL-8).
- h̄_l is built exactly as NR-58 built its pooled curve (pooled over `_warp_on_grid`, i.e. shape values only, no o_i). Client i's forward pass pools its own live h_{i,l} with the other clients' values frozen from the last round, using saturating weights c/(τ + c), followed by the differentiable isotonic projection. Inference applies the same operator to the final values.
- o_i is private per client, never pooled into h̄_l and never averaged, trained from the gain warm-up (round 2), |o_i| ≤ 2.
- Only the shapes are pooled. s = 1.

### Shared code change (`fedsteer/fed.py`, `tests/test_fedsteer.py`)

- `_check_coverage_config` no longer rejects `lora.offset=true` when `calibration=aligned`; coverage and consensus still reject it. Setting `lora.offset=true` is the opt-in: the same config used to raise an error, so every existing configuration behaves exactly as before.
- No other code was needed:
  - `coef_for` already returns offset + gain · table(α) when a table is set;
  - in aligned mode the offset already lives in each client's `gain` state, as the calibration is not `shared`;
  - `load_client` and `load_snapshot_into` already restore it alongside the table;
  - `e2_heldout.py` and `e3_drift.py` already refuse aligned runs.
- **New test** `test_aligned_with_private_offset`:
  - coverage and consensus still reject an offset;
  - aligned + offset trains; nothing goes through FedAvg; the o_i are nonzero and differ between clients;
  - at inference the coefficient is exactly o_i + h̄_l(0.5), and o_i at α = 0;
  - snapshot → `load_snapshot_into` reproduces it;
  - resume restores o_i.
- **Full suite 75/75 pass.**
- **CPU smoke** (`train_fed.py`, Newsroom, 3 rounds): runs cleanly; the server holds no calibration parameters; 8 distinct offsets; 3 table updates over 112 shapes.
- Code synced to Anvil (`setup_anvil.sh --code-only`; a follow-up dry run shows no differences). The coordinator has been told about the edit to the shared file.

### Job

- Launch [NR-62.sh](launch/NR-62.sh). NR-58's settings plus `lora.offset=true`:
  - `fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01`, with b = 0.2 and τ = 100 at their defaults;
  - NR-60's common settings: Llama-3.2-1B, rotation 0, 4k per client, 100 rounds, `fix_gain=true`, `warp_scope=module`, `warp_reg=0`;
  - same train → dev → test → quality → judge pipeline.
- **cc 11209264, Anvil 21179348**, job name `nr62-fed-aligned-off`. Race watcher pid 2741896, spec [NR-62_race.json](launch/NR-62_race.json), log `sbatch/logs/newsroom/race-NR-62.log`.
- Anvil balance before submission: 473.1 SU. About 6–7 SU if Anvil wins.

### Analysis plan

- **Primary: `fed-aligned-off` vs. `fed-aligned`** (NR-58, 11203892): does a private level fix aligned's losses on nypost.com and reuters.com at α = 0 and 0.25 (NR-61 §3)?
- **`fed-aligned-off` vs. `fed-shared-off`** (NR-60): with private levels on both sides, does pooling shape values into h̄_l still trail FedAvg of shape parameters?
- **CAL-7's hypothesis:** if o_i absorbs the copy-heavy clients' level difference, the pooling weights should matter less.
  - Check `support_gap` and disagreement against NR-58: nypost.com 0.051 and theguardian.com 0.045 at round 100; disagreement 0.028.
  - Check whether the projection ever acts.
- **Report, per client:**
  - pct err in and out of support;
  - per-α cells with α = 0 and 1 separately, and the output percentile at α = 0;
  - o_i compared per client with NR-60 `fed-shared-off`'s o_i (coordinator, CAL-8): similar values would mean the level is identified independently of how the shape is learned;
  - quality and judge, including at α = 0.

**Coordinator note (CAL-8).** NR-62 accepted; the edit to `fed.py` is recorded as a closed grant in CONVENTIONS. Process: before touching shared code, send the coordinator a one-line note first, even when the user asks directly, so the molecule and math sessions do not sync mid-edit.

## NR-60a. Added arm `fed-shared-soff`: shared shape AND shared offset (2026-10-07, 21:52)

**Request (user):** "only for this nr60, add a new variant that makes clients share the offset o_i".
- Added as an NR-60 arm, not a new experiment ID.
- Label `fed-shared-soff` (CONVENTIONS §4: `soff` = shared offset).

**Model.** g_{i,l}(α) = o + h_l(α).
- A single offset o is shared by all clients, together with the shared per-layer shape h_l, s = 1.
- The server averages o every round like the shape and the direction (the code's default shared path, `fed.private_offset=false`), so o's Adam state is reset every round as for every shared parameter.
- No code change.

**Contrasts:**
- `fed-shared-soff` vs. `fed-shared-off`: the only difference is whether o is shared or private. Does the level need to be per client, or is a global level shift enough?
- `fed-shared-soff` vs. `fed-shared` (NR-54 A_L): does a global offset help at all with gain 1 and per-layer shapes? In the archive, a shared offset hurt (0.158 vs. 0.151, learned gain with a single shape); it has not been tested in this setting.
- **Expected if NR-60's interim reading holds** (o_i from −0.10 to −0.63, the copy-heavy clients largest): the shared o settles near the clients' average and fixes the α = 0 cell less for nypost.com and reuters.com than private o_i does.
- **Report:** the shared o over rounds, per-client α = 0 and α = 1 cells, and the output percentile at α = 0.

**Job.**
- Launch [NR-60_soff.sh](launch/NR-60_soff.sh): NR-60's common settings plus `fed.calibration=shared lora.offset=true fed.private_offset=false`.
- **cc 11209326, Anvil 21179398**, job name `nr60-fed-shared-soff`.
- Its own race (the NR60 group was already decided for cc): spec [NR-60_soff_race.json](launch/NR-60_soff_race.json), watcher pid 2762675, log `sbatch/logs/newsroom/race-NR-60-soff.log`.
- Anvil: no NR jobs have run there since the 473.1 SU balance check. About 6–7 SU if Anvil wins this race; together with NR-62, at most about 14 SU.

**Coordinator notes on NR-60a (CAL-8):**
- `fed-shared-soff` races on its own, so it may run on Anvil while the rest of NR-60 ran on cc. If it does, every table names its host next to it, and the contrast with `fed-shared-off` is flagged "different host". The torch and transformers versions are pinned to match, but it is still a difference.
- The report puts the learned shared o next to the mean of `fed-shared-off`'s private o_i. It also checks whether entry 20's result repeats at gain 1 with per-layer shapes: there, a shared offset was worse than none (0.158 vs. 0.151, learned gain, one shape).

## NR-60b. `fed-shared-soff` cancelled; the shared-offset arm is `fed-aligned-soff` instead (2026-10-07, 22:10)

**User correction:** "I actually meant the we build on top of the aligned calibration for the shared offset arm."
- NR-60a's `fed-shared-soff` (shared h_l + shared o) was **cancelled while PENDING** on both hosts at about 21:58 (cc 11209326, Anvil 21179398). Its watcher (pid 2762675) was stopped by hand; this is noted in `race-NR-60-soff.log`.
- NR-60a's plan and the coordinator's NR-60a notes now apply to `fed-aligned-soff`, with `fed-aligned-off` (NR-62) as its private-offset counterpart.

**Model.** g_{i,l}(α) = o + h̄_l(α).
- h̄_l is pooled exactly as in NR-58 and NR-62, in training and in inference: saturating weights, K = 21, b = 0.2, τ = 100, λ_max = 0.01.
- **One offset o for all clients**, averaged by the server every round like the direction, trained from round 2, |o| ≤ 2. Its Adam state is reset every round, as for every shared parameter. s = 1.

### Shared code change (`fedsteer/fed.py`, `tests/test_fedsteer.py`; the coordinator was told before the edit)

- New `FedConfig.shared_offset` (default `false`), accepted only with `lora.offset=true`, `calibration=aligned` and `mode=fedavg`; anything else raises. With `calibration=shared` the offset is already shared by default.
- When on:
  - o is appended to `shared_params` (round reset) and put into the server state (FedAvg);
  - clients' gain states no longer hold o, so `load_client` takes it from the server;
  - `load_snapshot_into` restores the server's o when the snapshot's `fed_config.shared_offset` is set.
- **The defaults are unchanged, bit for bit.** Old vs. new code over 3 rounds gives identical results for aligned without offset (the NR-58 path), `fed-shared-off`, `fed-shared`, `fed-private-off` and `local-off`.
- **New test** `test_aligned_with_shared_offset`:
  - invalid combinations raise;
  - the server's o equals the mean of the last round's client offsets;
  - every client's coefficient is o + table;
  - snapshot → `load_snapshot_into` and resume both preserve o.
  - **Full suite 76/76 pass.**
- **CPU smoke** of `train_fed.py` with the arm's exact settings: the server holds only o among the calibration parameters, no client holds o, and the table updated 3 times.
- **Pending jobs during the edit** (`nr60-local` 11206106, `nr62-fed-aligned-off` 11209264, Anvil 21179348): all PENDING throughout, so none snapshotted the tree mid-edit. `fed.py` was written in one step at 22:02:27. NR-62 runs identical code either way.
- Synced to Anvil after the tests passed; a follow-up dry run shows no differences. Coordinator informed (76/76); molecule and math may resume code syncs.

### Job

- Launch [NR-60b.sh](launch/NR-60b.sh), job name `nr60-fed-aligned-soff`: **cc 11209468, Anvil 21179559**.
- Its own race: spec [NR-60b_race.json](launch/NR-60b_race.json), watcher pid 2807952, log `sbatch/logs/newsroom/race-NR-60b.log`.
- About 6–7 SU if Anvil wins.
- If it lands on a different host from NR-62, every table names the host, and the contrast is flagged "different host" (coordinator, CAL-8).

### Analysis plan

- **`fed-aligned-soff` vs. `fed-aligned-off`** (NR-62): only whether o is shared or private differs. Does the level need to be per client?
- **`fed-aligned-soff` vs. `fed-aligned`** (NR-58): does one global level help at all? Entry 20 found a shared offset worse than none (0.158 vs. 0.151, learned gain, one shape).
- **Report:**
  - the learned shared o over rounds, next to the mean of `fed-aligned-off`'s and `fed-shared-off`'s private o_i;
  - per-client α = 0 and α = 1 cells and the output percentile at α = 0. Expected: the shared o fixes less for nypost.com and reuters.com, which had the largest private o_i in NR-60;
  - quality and judge;
  - the coverage diagnostics (`support_gap`, disagreement, projection activity) against NR-58 and NR-62.

## Paper draft note (no experiment ID): aligned calibration added to the method section (2026-10-07)

No code or jobs. `paper/Distributed_Steering/newsroom_experiments.tex` gets a new subsection "Aligned calibration" in the setup (the NR-58 design, `fed-aligned`): per-layer shapes, gain 1, no offset; evidence counts c_ik and saturating weights w_ik = c_ik/(τ + c_ik); pooled mean of shape values and the monotone projection Π (weighted isotonic regression, interpolation of uncovered points, endpoints 0 and 1); training through ḡ with own live values and others' last-round values (leave-one-out sums), differentiable projection; tie penalty λ_max = 0.01; K = 21, b = 0.2, τ = 100. Formulas checked against `fedsteer/coverage.py` and `fedsteer/fed.py`. Results tables unchanged; the text states they use the shared calibration. Offset variants (NR-60b, NR-62) not described.
- Follow-up (2026-10-08): reworded the uncovered-grid-point interpolation as part of the definition of Π (not an approximation), with the edge case at the ends of the axis. Checked nr58 run: no grid point is uncovered (min total count ≈ 3,190).
- Follow-up (2026-10-08): defined Ω_k (total evidence weight, constant), the covered set C, the isotonic-regression objective, and blocks / Ω_B / μ(B) / B(k) via the pool-adjacent-violators algorithm with strict merging as in fedsteer/coverage.py:isotonic_blocks.
- Follow-up (2026-10-08): stated that the block partition B = B(m) is piecewise constant (zero derivative a.e., no gradient needed), the projection is linear A_B m on each region, continuous and 1-Lipschitz at partition switches, and the strict-merge Jacobian is a Clarke generalized Jacobian element there.

## NR-60c. `fed-shared-soff` resubmitted as an independent run, alongside `fed-aligned-soff` (2026-10-07, 22:55)

**User:** "also do an independent run of using plain federated (11206103) with shared offset".
- 11206103 is `fed-shared-off` (shared h_l, private o_i). Its plain-federated counterpart with a shared offset is `fed-shared-soff`: g_{i,l}(α) = o + h_l(α), with one o averaged by the server like h_l and the direction.
- This is the arm cancelled in NR-60b, which never started (no run directory). It is now resubmitted with identical settings, in addition to `fed-aligned-soff`.
- No code change: `calibration=shared` with `lora.offset=true` and the default `private_offset=false` shares the offset. That path was verified bit-identical to the pre-NR-60 code.

**Job.**
- Launch [NR-60_soff.sh](launch/NR-60_soff.sh), run unchanged. NR-60 common settings plus `fed.calibration=shared lora.offset=true fed.private_offset=false`.
- **cc 11209750, Anvil 21179756**, job name `nr60-fed-shared-soff`.
- Its own race: spec [NR-60c_race.json](launch/NR-60c_race.json), watcher pid 2967789, log `sbatch/logs/newsroom/race-NR-60c.log`.
- About 6–7 SU if Anvil wins. Possible Anvil total for NR-62, NR-60b and NR-60c: about 20 SU.

**Plan (as in NR-60a, with the coordinator's CAL-8 notes):**
- **`fed-shared-soff` vs. `fed-shared-off`** (11206103): the only difference is whether o is shared or private.
- **`fed-shared-soff` vs. `fed-shared`** (NR-54 A_L): does a global offset help at gain 1 with per-layer shapes, or does entry 20 repeat (shared offset worse than none: 0.158 vs. 0.151)?
- **`fed-shared-soff` vs. `fed-aligned-soff`** (NR-60b): the same shared o, with the shape from FedAvg of shape parameters vs. the pooled h̄_l.
- **Report:**
  - the learned shared o next to the mean of `fed-shared-off`'s private o_i;
  - per-client α = 0 and α = 1 cells and the output percentile at α = 0;
  - quality and judge;
  - the host next to each arm, flagging "different host" in any contrast that crosses cc and Anvil.

**Report layout for the offset family (coordinator, CAL-8):** one 2 × 3 table.
- Rows: shape **shared** (FedAvg of the shape parameters, h_l) vs. **aligned** (pooled h̄_l).
- Columns: **no offset** / **private o_i** / **shared o**.
- Cells: `fed-shared` (NR-54) | `fed-shared-off` (NR-60) | `fed-shared-soff` (NR-60c); `fed-aligned` (NR-58) | `fed-aligned-off` (NR-62) | `fed-aligned-soff` (NR-60b).
- Each cell gives the host and the selected round.

## NR-63. NR-60 first results: a private offset repairs private shapes but does not improve on the shared-shape reference (2026-10-07, 23:40)

Analysis only; no code or jobs.

**State (23:17).**
- **Done**, both on cc (ccc0284), both dev-selected at round 100, full pipeline: `fed-shared-off` 11206103 and `fed-private-off` 11206104 (finished 23:15).
- **Running on cc:** `local-off` (about round 85), `local`, `fed-aligned-off` (NR-62; the race was decided for cc at 23:16 and the Anvil copy cancelled).
- **Pending:** `fed-aligned-soff` (NR-60b) and `fed-shared-soff` (NR-60c), each on cc and Anvil.
- Anvil cost so far: 0 SU (balance 473.1).

**Reports:**
- `reports/nr63_offset_test.txt`: `compare_runs.py`, 200 test articles per client, paired bootstrap;
- `reports/nr63_offset_cells_test.txt`: per-α cells, output percentiles, o_i;
- `reports/nr63_private_shape_distance.txt`: coordinator item 3;
- `reports/nr63_quality_alpha0.txt`: quality and judge at α = 0 and 1;
- `reports/nr63_response_gap.txt`: signed output-percentile error.

Single seed. All runs in this entry are on cc.

### The offset grid so far (CAL-8 layout; test pct err, worst client in brackets; host, selected round)

| Shape \ offset | none | private o_i | shared o |
|---|---|---|---|
| shared h_l (FedAvg of parameters) | `fed-shared` 0.150 (0.204); cc, r100 | `fed-shared-off` **0.150 (0.205)**; cc, r100 | `fed-shared-soff`: pending (NR-60c) |
| aligned h̄_l (pooled) | `fed-aligned` 0.153 (0.213); cc, r100 | `fed-aligned-off`: running (NR-62) | `fed-aligned-soff`: pending (NR-60b) |
| *(private h_{i,l}, for reference)* | *`fed-private` 0.158 (0.241); cc, r100* | *`fed-private-off` **0.150 (0.192)**; cc, r100* | — |

| Arm | In-support | Out-of-support (worst) | Reach | α = 0 cell | α = 1 cell | nypost.com | reuters.com | theguardian.com |
|---|---|---|---|---|---|---|---|---|
| `fed-shared` | 0.134 | 0.157 (0.201) | **0.407** | 0.198 | **0.110** | 0.187 | 0.204 | 0.129 |
| `fed-shared-off` | 0.134 | 0.158 (**0.184**) | 0.393 | **0.196** | 0.121 | **0.172** | 0.205 | **0.124** |
| `fed-private` | 0.136 | 0.168 (0.269) | 0.406 | 0.236 | 0.103 | 0.241 | 0.213 | 0.125 |
| `fed-private-off` | **0.133** | 0.160 (0.208) | 0.393 | 0.209 | 0.116 | 0.192 | **0.190** | **0.123** |

### Paired contrasts (95% bootstrap CI excludes 0)

| Contrast | Mean | Better on | Worse on | Out of support |
|---|---|---|---|---|
| **`fed-shared-off` − `fed-shared`** (primary) | 0.150 vs. 0.150 | 1/8: nypost.com −0.015 | 1/8: people.com +0.008 | better on theguardian.com (−0.009), nypost.com (−0.017); worse on aol.com (+0.021), people.com (+0.013) |
| `fed-private-off` − `fed-private` | 0.150 vs. 0.158 | 2/8: **nypost.com −0.049**, reuters.com −0.023 | 1/8: forbes.com +0.007 | nypost.com −0.061, reuters.com −0.029 |
| `fed-shared-off` − `fed-private-off` | 0.150 vs. 0.150 | 2/8: nypost.com −0.020, cbc.ca −0.007 | 2/8: reuters.com +0.015, people.com +0.006 | |
| `fed-shared-off` − `fed-aligned` | 0.150 vs. 0.153 | 2/8: nypost.com −0.037, reuters.com −0.008 | 2/8: aol.com +0.012, wsj.com +0.007 | aol.com +0.042 |

### Learned offsets at the selected round (r100)

| Client | o_i, `fed-shared-off` | o_i, `fed-private-off` |
|---|---|---|
| nypost.com | **−0.663** | **−0.510** |
| reuters.com | **−0.376** | **−0.437** |
| forbes.com | −0.166 | −0.202 |
| cbc.ca | −0.173 | −0.136 |
| people.com | −0.139 | −0.147 |
| aol.com | −0.143 | −0.104 |
| wsj.com | −0.097 | −0.087 |
| theguardian.com | −0.020 | −0.041 |
| **mean** | **−0.222** | **−0.208** |

- The two arms give nearly the same o_i (same order, nypost.com and reuters.com far below the rest), so the level is identified independently of whether the shape is shared or private.
- The mean is −0.22 from round 30 on (dev: −0.24 / −0.25 / −0.23 / −0.22 / −0.22 at rounds 30 / 50 / 70 / 90 / 100). The offsets settle early and stay.

### What the results say

**1. Primary: a private level on top of a shared shape does not improve `fed-shared`.**
- The means tie (0.150), and the per-client contrast is 1 better / 1 worse.
- The worst-client out-of-support error is lower (0.184 vs. 0.201), and nypost.com improves (−0.015).
- The cost is at the top: reach 0.393 vs. 0.407, and the α = 1 cell 0.121 vs. 0.110 (aol.com 0.891 → 0.855, people.com 0.815 → 0.783).
- **Mechanism:** with s fixed at 1 the coefficient spans [o_i, o_i + 1]. A negative offset *shifts* that range down rather than stretching it, so what the low end gains the top end gives back.

**2. The α = 0 advantage on dev was transient.**
- The α = 0 cell with and without the offset (dev): 0.266 vs. 0.283 at round 30 (−0.017), 0.240 vs. 0.260 at round 50 (−0.020), then −0.001 at round 70, −0.007 at round 90, −0.001 at round 100. On test: 0.196 vs. 0.198.
- The offsets stay at about −0.22 throughout, so it is `fed-shared` that catches up: without an offset, P_i and D reach the same α = 0 level later in training.
- So with a shared shape, the private level mainly **speeds up** the low end. The interim reading in NR-61 (round 50) overstated it.

**3. The offset matters where the shape is private: it repairs `fed-private`'s copy-heavy failure.**
- nypost.com 0.241 → 0.192: its α = 0 output percentile falls from 0.440 to 0.275, and the α = 0.25 cell from 0.358 to 0.244.
- reuters.com 0.213 → 0.190, the best reuters.com of any Newsroom arm so far.
- `fed-private-off` has the **best worst-client error of all runs (0.192)** and the best in-support error (0.133), and ties `fed-shared` on the mean. This is the first private-shape design that matches the shared one; borrow and consensus only repaired toward it (NR-57, NR-59).

**4. "Level, not shape" (CAL-7), coordinator item 3 at the selected round.** Mean over 112 layers of |h_{i,l}(α) − mean of the six non-focus clients' h_{·,l}(α)|, in-support α only; "pooled" is over the in-support points of 0.05, …, 0.95:

| Client (support) | `fed-private` r100: d(0.25) / d(0.5) / d(0.75); pooled | `fed-private-off` r100: d(0.25) / d(0.5) / d(0.75); pooled |
|---|---|---|
| nypost.com [0.51, 0.95] | out / out / 0.356; **0.334** | out / out / 0.210; **0.197** (−41%) |
| reuters.com [0.10, 0.97] | 0.143 / 0.145 / 0.125; **0.131** | 0.174 / 0.144 / 0.092; **0.125** (−5%) |
| the six others, pooled | 0.054–0.125 | 0.055–0.119 |

- For **nypost.com the level explains about 40%** of its distinct shape. Its mean h(0.5) goes 0.283 → 0.368, against 0.60–0.71 for the others. It is still about 1.7 times farther from the others than the most distinct of them.
- For **reuters.com the level explains almost none of it** at round 100. The round-50 reduction (−22%, NR-61) did not hold: its shape moves closer at 0.75 but farther at 0.25.
- **Response gap under `fed-shared-off`** (signed mean output percentile − α, in support): reuters.com still overshoots by **+0.26 at α = 0.25 and +0.22 at α = 0.5**, the same as `fed-shared` (+0.26 / +0.21). The others are within ±0.10.
  - So o_i = −0.38 does not remove reuters.com's in-support overshoot.
  - The overshoot falls toward α = 0.75 (+0.07), which is a shape (slope) gap, not a level gap.
- **Verdict:** the level is a real, consistently identified per-client quantity, largest for the copy-heavy clients, but **most of the copy-heavy clients' difference is shape**: about 60% for nypost.com and essentially all of it for reuters.com.

**5. Quality at α = 0** (outside every client's support; 25 judged summaries per client per α):

| Arm | AlignScore at α = 0 | judge faithful at α = 0 | length |
|---|---|---|---|
| `fed-shared` | 0.511 | 0.782 | 24.8 |
| `fed-shared-off` | 0.519 | 0.810 | 24.4 |
| `fed-private` | 0.537 | 0.794 | 24.8 |
| `fed-private-off` | 0.520 | 0.780 | 23.8 |

- The uniform offset does not hurt quality at α = 0 under a shared shape: slightly better, within noise.
- Under private shapes, nypost.com's AlignScore at α = 0 falls (0.651 → 0.524), and so does its judge faithfulness (0.807 → 0.700, n = 25). This comes with copying much less (output percentile 0.440 → 0.275). AlignScore and faithfulness rise with copying, so this is the expected trade, not damage.
- At α = 1, all four arms are within 0.01 AlignScore.
- Out-of-support length gaps shrink with an offset (+2.2 vs. +3.2 words for `fed-shared`), so long out-of-support summaries are slightly less overlong.

### Consequences

- For a shared shape, a private offset is neutral on Newsroom: it trades the top end for faster convergence at the bottom.
- With fixed gain, an offset can only shift the coefficient range. The arm that would separate level from range would let the gain or the range adapt, e.g. o_i with s_i learned, or a top-anchored offset. That is a design question for the user and coordinator; nothing is launched.
- For CAL-7: `fed-aligned-off` (NR-62) can at best remove the level part of aligned's nypost.com loss (about 40% of nypost.com's shape distance) and little of reuters.com's.
- **Still to come:**
  - `local-off` and `local`: contrasts 3–5 of the NR-60 plan, the first matched local baseline at gain 1 with per-layer shapes;
  - `fed-aligned-off` (NR-62);
  - `fed-shared-soff` and `fed-aligned-soff` (NR-60b/c);
  - the full 2 × 3 table once all six cells exist.

### NR-63a. Which calibration form would the clients need? Inversion of their response curves (coordinator request, CAL-9; no new runs)

**Method** (`reports/nr63_needed_coefficient.txt`):
- Summarise each client's coefficient at the grid α_k by c_k = o_i + s · mean_l h_l(α_k). This is a scalar approximation: the per-layer shapes spread by SD 0.26 / 0.19 / 0.10 at α = 0.25 / 0.5 / 0.75.
- Take its mean test output percentile p_k at each grid α_k.
- Invert the piecewise-linear response p(c) at the target p = α_k to get the **needed coefficient c*_k**.
- At α = 0 and α = 1 the target lies outside every client's observed response, so c* there is an extrapolation and is not used. That leaves **3 points per client** (α = 0.25, 0.5, 0.75).
- Fit three forms against h̄ = mean_l h_l: (a) c* = o + h; (b) c* = o + (1 − o) h (top-anchored); (c) c* = a + b · h.
- Done on `fed-shared-off` and, as a check, on `fed-shared`. The two runs have different adapters, so the numbers differ, but the conclusion is the same.

**`fed-shared-off`, r100** (h̄ = 0.437 / 0.643 / 0.826 at α = 0.25 / 0.5 / 0.75):

| Client | o_i | needed c* at α = 0.25 / 0.5 / 0.75 | (a) shift: rmse | (b) top-anchored: rmse | (c) a, b: rmse |
|---|---|---|---|---|---|
| reuters.com | −0.376 | −0.360 / 0.051 / 0.326 | 0.124 | 0.141 | a = −1.12, **b = 1.77**: 0.023 |
| nypost.com | −0.663 | −0.498 / −0.151 / 0.037 | 0.068 | 0.281 | a = −1.08, **b = 1.38**: 0.030 |
| forbes.com | −0.166 | 0.206 / 0.458 / 0.604 | **0.020** | 0.083 | a = −0.23, b = 1.03: 0.020 |
| the five others | −0.02 to −0.17 | | 0.077–0.159 | 0.023–0.066 | b = 1.43–1.91: 0.003–0.066 |

`fed-shared` (no offset) gives b = 1.10 (forbes.com) to 2.09 (cbc.ca), with reuters.com 2.01 and nypost.com 1.70.

**Reading (rough: 3 points, 2 parameters for form (c)):**

1. **The needed coefficient curve is steeper than h̄ for every client except forbes.com (b ≈ 1.4–2.0). Clients need a *wider* coefficient range, not a smaller slope.**
   - reuters.com's output overshoots at α = 0.25 (+0.26) much more than at 0.75 (+0.07). Its output curve rises too fast early, so the *coefficient* must start much lower and then rise faster to catch up: needed c* − used c = −0.42, −0.22, −0.12 at α = 0.25 / 0.5 / 0.75.
   - In coefficient terms that is a larger slope, i.e. a gain s_i > 1 together with a negative level.
   - This is the coordinator's "smaller slope" read in output space; in coefficient space it is the opposite.
2. **(a) a pure shift fits only forbes.com.** For reuters.com it is the worst of the three forms (rmse 0.12). This is the NR-63 finding again: with s = 1 an offset cannot supply the missing range.
3. **(b) top-anchored fits the middle clients** (aol.com, people.com, wsj.com: rmse 0.02–0.04) **but not the copy-heavy ones under `fed-shared-off`** (reuters.com 0.14, nypost.com 0.28), whose needed curve sits well below h̄ at α = 0.75.
   - Under `fed-shared` (no offset) (b) fits them acceptably (0.04–0.06).
   - **The 5-point grid cannot identify the top end** (α = 1 is always extrapolated), so (b) and (c) are not separable at the top.
4. **(c) a + b · h with both free fits everyone.** The levels a are well below the learned o_i for the copy-heavy clients (reuters.com −1.12 vs. o_i = −0.38; nypost.com −1.08 vs. −0.66), and the slopes are b ≈ 1.4–1.8.
   - That is the parametrization g_{i,l}(α) = o_i + s_i · h_l(α) with a **learned private gain**. With a shared shape it is the old method (learned gain, offset; archive 11063910, 0.151 at test with one shape).
   - So the next form the data points to is **o_i + s_i · h_l(α) with per-layer shapes**. This is untested on Newsroom: the learned-gain runs all used one shape.
5. **Caveats:**
   - The endpoint targets (percentile 0 and 1) are never reached by any client within the coefficients used; the extrapolated c*(0) is −0.5 to −1.1 and c*(1) is 0.9–1.3.
   - |o_i| ≤ 2 and the current gain clamp (0.25–4) would allow these values.
   - The scalar coefficient ignores how the per-layer shapes differ.
   - A direct coefficient sweep per client (as `scripts/coeff_sweep.py` does for ChEMBL) would measure the response at more than 5 points. It needs GPU evals and is not launched.

## NR-64. Pooling-weight sweep for `fed-aligned`: τ_pool ∈ {300, 1000, 3000} and count pooling, launched (2026-10-07, 23:38)

**Request (user).** Tune τ in the pooling weight, to test NR-61's leading explanation for `fed-aligned`'s nypost.com/reuters.com loss: the saturating weights under-weight the clients that own the high-α region.
- `fed-aligned-cnt` had been held by the coordinator (CAL-7); the user has now decided. The coordinator was told before launch.
- No code change.

**Why this grid** (from NR-58's `coverage.json`).
- The in-support evidence counts c_ik are in the hundreds to thousands. For example, at α = 0.9: nypost.com 2331, reuters.com 1361, cbc.ca 589, aol.com 338, people.com 79, theguardian.com 11.
- So at τ_pool = 100 every client's weight c/(τ + c) is close to 1. nypost.com and reuters.com share of the pooled weight, by setting:

| α | τ = 100 (NR-58) | τ = 300 | τ = 1000 | τ = 3000 | τ = 10000 | count |
|---|---|---|---|---|---|---|
| 0.5 | 0.23 | 0.21 | 0.18 | 0.17 | 0.16 | 0.15 |
| 0.8 | 0.30 | 0.34 | 0.41 | 0.47 | 0.51 | 0.54 |
| 0.9 | 0.33 | 0.39 | 0.49 | 0.58 | 0.63 | 0.66 |

- τ = 10000 is within 0.03 of count everywhere, so the grid is {300, 1000, 3000, count}. Together with NR-58 (τ = 100) it spans the weighting from equal-per-client to proportional-to-data in roughly even steps.
- Larger τ also shifts weight at mid α toward the broad clients (0.23 → 0.15 at α = 0.5), not only toward the copy-heavy ones at high α.

**Arms.** NR-58's setting with only the pooling changed:
- `fed-aligned`, no offset, λ_max = 0.01, K = 21, b = 0.2, τ_local = 100 (the tie weights are unchanged);
- NR-60's common settings (Llama-3.2-1B, rotation 0, 4k per client, 100 rounds, s = 1, per-layer shapes, warp_reg 0).

Launch [NR-64.sh](launch/NR-64.sh); all four arms are one race group, so the sweep runs on one host:

| Arm | Pooling | cc | Anvil |
|---|---|---|---|
| `fed-aligned-tp300` | saturating, τ_pool = 300 | 11210164 | 21180024 |
| `fed-aligned-tp1000` | saturating, τ_pool = 1000 | 11210167 | 21180030 |
| `fed-aligned-tp3000` | saturating, τ_pool = 3000 | 11210170 | 21180031 |
| `fed-aligned-cnt` | count | 11210173 | 21180180 |

- Watcher pid 3156912, spec [NR-64_race.json](launch/NR-64_race.json), log `sbatch/logs/newsroom/race-NR-64.log`.
- **Pre-check:** a CPU run of the trainer with `cov_pool=count` and with τ_pool = 3000 (aligned, per-layer shapes, 3 rounds) gives finite, monotone tables.
- **Anvil:** at most about 26 SU if the group lands there. Total possible exposure with NR-60b/c pending is about 39 SU, within the 45 SU cap.

**Selection and analysis plan.**
- **Choose τ on dev:** mean pct err of each arm's dev-selected checkpoint, with τ = 100 (NR-58) as the fifth point. Then report test for all five; the test number of the dev-chosen τ is the headline, so test is not used to pick τ.
- **If the group lands on Anvil**, the τ = 100 point (cc) is "different host" in every comparison.
- **Prediction (NR-61):** larger τ moves nypost.com and reuters.com toward their `fed-shared` errors (0.187 / 0.204, against 0.209 / 0.213 for NR-58).
- **Risk:** nypost.com dominates h̄_l at high α, flattening it for the broad clients and costing `fed-aligned`'s α = 1 advantage (0.094 vs. 0.110). Watch aol.com, wsj.com and cbc.ca at α = 1, and reach.
- **Report:**
  - per-client pct err and paired contrasts vs. `fed-aligned` (τ = 100) and `fed-shared`;
  - per-α cells with α = 0 and 1 separately;
  - the coverage diagnostics (`support_gap`, disagreement, whether the projection ever acts — never at τ = 100);
  - the high-α pooled-weight shares above next to the outcome.
- Follow-up (same day): added a "Gradient through the pooled curve" paragraph. It covers interpolation → projection Jacobian (block averaging with the partition held fixed; exact except where adjacent blocks tie) → pooling (∂ḡ(a_k)/∂h_i(a_k') = 1[k' ∈ B(k)] ω_ik'/Ω_B(k)) → shape parameters, plus the tie-penalty gradient. The pooling weight was renamed ω_ik to avoid a clash with the shape's mixing weight w. New citation: Blondel et al. 2020, in `paper/Distributed_Steering/newsroom.bib`.

## NR-65. NR-60 local arms, NR-62 and NR-60b: `fed-aligned-off` is the best Newsroom design so far; the level must be private; Anvil's environment is broken (2026-10-08, 09:45)

Analysis only; no code or jobs.

**State (09:20).**
- **Done**, all on cc (ccc0284), full pipeline:
  - `local-off` 11206105 (dev-selected r60);
  - `local` 11206106 (r100);
  - `fed-aligned-off` 11209264 (NR-62, r100);
  - `fed-aligned-soff` 11209468 (NR-60b, r100).
- **Running on cc:** `fed-shared-soff` 11209750 (NR-60c), `fed-aligned-tp300` 11210164, `fed-aligned-tp1000` 11210167.
- **Pending on cc:** `fed-aligned-tp3000` 11210170, `fed-aligned-cnt` 11210173.

**Reports:**
- `reports/nr65_test.txt` (`compare_runs.py`; 200 test articles per client; paired bootstrap);
- `reports/nr65_extra_pairs.txt`;
- `reports/nr65_cells_test.txt` (per-α cells, output percentiles, o_i).

Single seed. Every run in this entry is on cc, so there are no cross-host contrasts.

### Anvil: every copy since 03:09 failed in seconds; the conda environment is damaged

- The Anvil copies of NR-60c (21179756) and all four NR-64 arms (21180024/30/31, 21180180) **FAILED 4–8 s after starting**, with `ImportError: cannot import name '_parser' from partially initialized module 're'`.
- In `/anvil/scratch/x-zchen17/lucmon/envs/rein/lib/python3.11`, the `re/` package has **only `__pycache__` left**, and **3 top-level stdlib `.py` files remain**. Many stdlib directories (`collections`, `ctypes`, `xml`, `unittest`, `html`, …) were modified after 05:00 EDT today. `site-packages` (142 entries, torch and transformers among them) is still listed.
- Nothing in our workflow writes to the environment. **Probable cause (unconfirmed): Anvil's scratch purge.** Files unpacked from conda packages keep old timestamps, so an age-based purge would select them. The environment was built on 10-04.
- **Cost: 0 SU**; the balance is still 473.1, since the failed jobs ran for seconds.
- **Side effect:** the race watchers count a FAILED copy as "started", so they recorded the races as decided without cancelling anything. The cc copies are unaffected and will run there.
- **Consequences:**
  - **Anvil cannot run any job** of either workstream (Newsroom or molecule) until the environment is rebuilt.
  - Rebuilding it on scratch would run into the purge again. `/home` has only 25 GB, and projects (5 TB) is an alternative.
  - This is the user's decision, on someone else's account; not done.

### The offset grid (CAL-8 layout): test pct err (worst client); host, selected round

| Shape \ offset | none | private o_i | shared o |
|---|---|---|---|
| shared h_l (FedAvg of parameters) | `fed-shared` 0.150 (0.204); cc, r100 | `fed-shared-off` 0.150 (0.205); cc, r100 | `fed-shared-soff`: running (NR-60c) |
| aligned h̄_l (pooled, τ_pool = 100) | `fed-aligned` 0.153 (0.213); cc, r100 | **`fed-aligned-off` 0.146 (0.199); cc, r100** | `fed-aligned-soff` 0.150 (0.210); cc, r100 |
| *local (private h_{i,l}, own D_i)* | *`local` 0.157 (0.244); cc, r100* | *`local-off` 0.178 (0.266); cc, **r60*** | — |

| Arm | In-support | Out-of-support (worst) | Reach | α = 0 cell | α = 1 cell | nypost.com | reuters.com | Spearman (worst) |
|---|---|---|---|---|---|---|---|---|
| `fed-shared` | 0.134 | 0.157 (0.201) | 0.407 | 0.198 | 0.110 | 0.187 | 0.204 | 0.931 (0.881) |
| `fed-aligned` | 0.138 | 0.157 (0.228) | 0.421 | 0.212 | **0.094** | 0.209 | 0.213 | 0.927 (0.873) |
| **`fed-aligned-off`** | 0.135 | **0.151 (0.184)** | **0.425** | **0.195** | 0.104 | **0.174** | **0.199** | 0.929 (0.892) |
| `fed-aligned-soff` | 0.136 | 0.155 (0.209) | 0.405 | 0.201 | 0.102 | 0.194 | 0.210 | 0.931 (0.884) |
| `local` | 0.139 | 0.159 (0.274) | 0.396 | **0.189** | 0.130 | 0.244 | 0.194 | 0.912 (0.805) |
| `local-off` | 0.154 | 0.186 (0.300) | 0.365 | 0.246 | 0.134 | 0.266 | 0.213 | 0.886 (0.730) |

### 1. `fed-aligned-off` (NR-62): a private level fixes aligned's copy-heavy loss, and it is the best design so far

| Contrast | Mean | Better on | Worse on |
|---|---|---|---|
| `fed-aligned-off` − `fed-aligned` | 0.146 vs. 0.153 | **2/8**: nypost.com −0.034, reuters.com −0.014 | **0/8** |
| `fed-aligned-off` − `fed-shared-off` | 0.146 vs. 0.150 | **3/8**: people.com −0.011, aol.com −0.010, wsj.com −0.006 | **0/8** |
| `fed-aligned-off` − `fed-shared` (reference) | 0.146 vs. 0.150 | 1/8: nypost.com −0.012 (the other seven lean negative, CIs touch 0) | **0/8** |

- It is the first arm to improve on the reference without losing on any client. Its worst client (0.199) and worst out-of-support client (0.184) are the best of any run, as are its reach (0.425) and its α = 0 cell among the federated arms (0.195).
- **Mechanism, matching NR-61/63:**
  - Aligned's loss was the low end on the copy-heavy clients. P_nypost trained under h̄_l sat at an output percentile of 0.244 at α = 0. With o_i = −0.62 it sits at **0.168**; reuters.com goes from 0.275 to 0.229.
  - The private level lets each client set its own floor, while the pooled shape keeps aligned's top-end advantage (α = 1 cell 0.104 vs. `fed-shared`'s 0.110).
  - Under a FedAvg shape (`fed-shared-off`) the same offsets bought nothing (NR-63). The difference is that aligned's h̄_l reaches higher (0.425 reach): there is range at the top to give back.
- **Offsets are the same as in the other federated arms:** nypost.com −0.618, reuters.com −0.381, the others −0.02 to −0.15. `fed-shared-off` gave −0.663 / −0.376. This is the third arm with the same per-client levels; **the level is identified independently of how the shape is learned** (CAL-8's question).
- Dev agrees: `fed-aligned-off` 0.148 at r100 against `fed-shared` 0.147. Its dev error was below `fed-shared`'s at r60 (0.156 vs. 0.161) and r90 (0.150 vs. 0.156). Single seed: the test margin over `fed-shared` (0.004) is significant per client only on nypost.com.

### 2. The level must be private: `fed-aligned-soff` (NR-60b)

- The shared o settles at **−0.037**, against −0.21 for the mean of the private o_i. Averaging every round keeps it near the clients' consensus, and most clients want little shift.
- `fed-aligned-soff` − `fed-aligned`: better on 1/8 (nypost.com −0.015), worse on 0/8; mean 0.150 vs. 0.153.
- `fed-aligned-soff` − `fed-aligned-off`: worse on 2/8 (**nypost.com +0.019, reuters.com +0.011**), better on 1/8 (wsj.com −0.005).
- So a global shift does a little; the per-client level does the work. Entry 20's result (a shared offset worse than none) does **not** repeat here; it is about neutral.
- The FedAvg-shape version (`fed-shared-soff`, NR-60c) is still running.

### 3. The matched local baseline at last (gain 1, per-layer shapes): fed beats local, mostly through nypost.com

| Contrast | Mean | Better on | Worse on |
|---|---|---|---|
| `fed-shared` − `local` (C1) | 0.150 vs. 0.157 | 2/8: **nypost.com −0.058**, cbc.ca −0.013 | 2/8: wsj.com +0.010, reuters.com +0.010 |
| `fed-aligned-off` − `local` | 0.146 vs. 0.157 | **4/8**: **nypost.com −0.070**, theguardian.com −0.009, people.com −0.009, cbc.ca −0.008 | **0/8** |
| `fed-private-off` − `local` | 0.150 vs. 0.157 | 2/8: nypost.com −0.053, theguardian.com −0.012 | 2/8: wsj.com +0.012, forbes.com +0.008 |

- Local is competitive on the broad clients. It even has the best α = 0 cell (0.189): each client's own direction reaches its own low end.
- Local fails the copy-heavy, narrow-support client: nypost.com 0.244, against 0.172–0.192 federated. Its ranking quality is also lower (Spearman worst client 0.805 vs. 0.88–0.89; near-tie rate 0.219 vs. 0.19–0.20).
- **`fed-aligned-off` is the first federated arm that beats the matched local baseline without losing on any client.**
- The per-round Adam reset is part of FedAvg (CAL-6) and is not a confound here.

### 4. A private offset hurts local training: the offset drifts when the direction is private

- `local-off` − `local`: **worse on 7/8 clients**, better on none; 0.178 vs. 0.157. It was dev-selected at r60.
- **Mean o_i on dev drifts steadily upward**: −0.12 at r20, +0.08 at r30, +0.21 at r50, +0.52 at r60, +0.73 at r80, +0.78 at r100. Dev error stalls after r60: 0.179, 0.189, 0.192, 0.189, 0.185. `local`'s dev error keeps falling to 0.156.
- Final o_i are large and positive (theguardian.com **+1.10**, aol.com +0.62, people.com +0.52, nypost.com +0.24).
  - So at α = 0 the coefficient is far from 0: theguardian.com's α = 0 output percentile is 0.249 vs. 0.139 for `local`, and the α = 0 cell is 0.246 vs. 0.189.
- **Likely cause** (consistent with the data, not proven): in local mode the direction D_i is private too, so o_i · D_i is one more private term that the adapter P_i and the scale of D_i can trade against. Nothing pins o_i down, and it random-walks.
  - In fed, D is shared and anchors o_i: all three federated private-offset arms give the same o_i, which stay flat from r30 on.
- **Consequence:** the offset is a federated-only feature. `fed-private-off` − `local-off` (6/8 better, 0 worse) and `fed-shared-off` − `local-off` (5/8 better, 0 worse) compare against a degraded baseline. **Use `local` as the local reference.**

### 5. Quality

- AlignScore, BERTScore and the judge are within noise across the federated arms. `fed-aligned-off`: AlignScore 0.786 / 0.731 in / out of support, against 0.781 / 0.730 for `fed-shared`.
- `local-off` has the highest AlignScore (0.806 in support), consistent with its outputs being more extractive throughout (positive o_i). Its out-of-support length gap is the largest (+4.3 words).

### Conclusions

- **New best calibration on Newsroom: `fed-aligned-off`**: g_{i,l}(α) = o_i + h̄_l(α), with a pooled per-layer shape (aligned, saturating weights τ = 100) and a private offset. It scores 0.146 (worst 0.199); it is better than `fed-shared` on 1/8 clients, `fed-aligned` on 2/8, `fed-shared-off` on 3/8 and `local` on 4/8, and worse than none of them. Single seed: **a second seed of `fed-aligned-off` and `fed-shared` is the next confirmation needed.**
- **The level is per client and identified:** the same o_i across three federated designs. A shared level is close to useless (o → −0.04).
- **NR-64 (τ_pool sweep) now has a sharper question.** Larger τ weights the copy-heavy clients more in h̄_l. But `fed-aligned-off` already repairs those clients through o_i, so the sweep on plain `fed-aligned` may matter less than the same sweep with o_i, if a follow-up is wanted. NR-63a's remaining slope mismatch (needed b ≈ 1.4–1.8) is untouched by either.
- **Still open:** `fed-shared-soff` (NR-60c); the NR-64 sweep (2 running, 2 pending, all on cc now); the Anvil environment.

**Coordinator decisions after NR-65 (CAL-11), 2026-10-08:**
1. **The gain stays fixed at 1** (user decision). No gain designs; CAL-10's A/B/C are off, and NR-63a's learned-gain direction is not pursued.
2. **NR-64 report adds**, per τ_pool, nypost.com's and reuters.com's α = 0 and α = 0.25 cells next to `fed-aligned-off`'s (NR-62: nypost.com 0.168 output percentile at α = 0). This shows whether re-weighting the pool reaches the same place without a private level.
3. **No Anvil submissions** until the environment is rebuilt (the user's decision, raised by the coordinator). Race on cc only for now.
4. **Next, not launched, waiting on the user:**
   - seed-1 replicates of `fed-aligned-off`, `fed-shared` and `local`;
   - an E2 protocol for `fed-aligned-off`, where a new client fits only its o_i (`e2_heldout.py` refuses aligned runs today).

## NR-66. Pooling-weight sweep for `fed-aligned-off` (the offset version of NR-64), launched on cc (2026-10-08, 10:05)

**Request (user):** "an offset version of the experiments of sweeping tau in aligned calibration."
- Model: g_{i,l}(α) = o_i + h̄_l(α), with o_i private. Only the pooling weight of h̄_l varies: c/(τ_pool + c), or count.
- Everything else is NR-62's setting. **τ_pool = 100 is NR-62** (`fed-aligned-off`, 0.146), so this sweep completes the τ axis for the best design.
- No code change; the coordinator was told.

| Arm | Pooling | cc job |
|---|---|---|
| `fed-aligned-off-tp300` | saturating, τ_pool = 300 | 11215856 |
| `fed-aligned-off-tp1000` | saturating, τ_pool = 1000 | 11215857 |
| `fed-aligned-off-tp3000` | saturating, τ_pool = 3000 | 11215858 |
| `fed-aligned-off-cnt` | count | 11215859 |

- Launch [NR-66.sh](launch/NR-66.sh). **cc only, no race:** the Anvil environment is broken (NR-65; CAL-11 rule).
- nypost.com + reuters.com share of the pooled weight at α = 0.9: 0.33 (τ = 100), 0.39, 0.49, 0.58, 0.66 (count) (NR-64 table).

**Host note (applies to NR-64 too).**
- Two NR-64 arms landed outside ccc0284 (dali, A100-40GB), on IllinoisComputes-GPU: `fed-aligned-tp3000` on ccc0465 (**H200**) and `fed-aligned-cnt` on ccc0390 (A100-80GB).
- Same cluster, environment and code; different GPU types, so bf16 kernels differ slightly. Tables will list the node per arm.
- The same may happen for NR-66.

**Analysis plan.**
- Choose τ on dev (each arm's dev-selected checkpoint, with τ = 100 from NR-62), then report test for all five.
- **The 2 × 5 grid with NR-64:** rows `fed-aligned` / `fed-aligned-off`, columns τ = 100, 300, 1000, 3000, count. The question is whether the private level and the re-weighting are substitutes or complements:
  - substitutes: the τ effect seen without an offset (NR-64) vanishes with one;
  - complements: both help.
- **Per τ:** nypost.com and reuters.com cells at α = 0 and α = 0.25, and their α = 0 output percentile, next to NR-62 (CAL-11 request). Also the o_i: do they shrink as h̄_l moves toward the copy-heavy clients?
- **The risk from NR-64 applies here too:** at large τ, nypost.com dominates h̄_l at high α. Watch the α = 1 cell and reach for aol.com, wsj.com and cbc.ca (`fed-aligned-off` holds 0.104 and 0.425 at τ = 100).

**Coordinator caveat for NR-64/66 (CAL-13).**
- Arms on different GPU types (H200, A100-80GB, ccc0284's A100-40GB) are not bit-reproducible against each other.
- With one seed, **differences of ≤ 0.005 in mean pct err between τ values count as unresolved**, and the write-up says so.
- Any τ chosen on dev needs a seed-1 replicate before it enters the design.

## NR-67. NR-64 τ_pool sweep (3 of 4 arms) and NR-60c `fed-shared-soff`, with metrics beyond pct err (2026-10-08, 15:30)

Analysis only; no code or jobs.

**State (14:46).**
- **Done** (full pipeline, all dev-selected at r100):
  - `fed-shared-soff` 11209750 (cc ccc0284, dali A100-40GB);
  - `fed-aligned-tp300` 11210164 and `fed-aligned-tp1000` 11210167 (ccc0284);
  - `fed-aligned-tp3000` 11210170 (**ccc0465, H200**).
- **Running:** `fed-aligned-cnt` 11210173 (ccc0390, A100-80GB); NR-66 τ = 300, 1000, 3000 (ccc0284).
- **Pending:** NR-66 count (11215859).

**Reports:**
- `reports/nr67_test.txt` (`compare_runs.py`; pct err, quality, judge, paired bootstrap);
- `reports/nr67_other_metrics.txt` (steering-shape, quality, judge per run);
- `reports/nr67_cells_test.txt` (per-α cells and output percentiles).

Single seed. Per CAL-13, mean differences of ≤ 0.005 between arms on different GPU types are **unresolved**.

### 1. NR-64: re-weighting the pool helps `fed-aligned`, mostly on nypost.com

| τ_pool (node) | nypost.com + reuters.com share at α = 0.9 | Dev pct err r100 | **Test pct err (worst)** | Out-of-support (worst) | Reach | α = 1 cell | nypost.com | reuters.com |
|---|---|---|---|---|---|---|---|---|
| 100 = `fed-aligned` (ccc0284) | 0.33 | 0.152 | 0.153 (0.213) | 0.157 (0.228) | 0.421 | 0.094 | 0.209 | 0.213 |
| 300 (ccc0284) | 0.39 | 0.150 | 0.150 (0.210) | 0.152 (0.194) | 0.417 | 0.100 | 0.180 | 0.210 |
| 1000 (ccc0284) | 0.49 | 0.151 | 0.151 (0.213) | 0.156 (0.210) | 0.418 | 0.098 | 0.192 | 0.213 |
| **3000 (ccc0465, H200)** | 0.58 | **0.148** | **0.147 (0.202)** | **0.149 (0.198)** | **0.435** | **0.091** | 0.184 | **0.202** |
| count | 0.66 | running | | | | | | |
| *ref. `fed-aligned-off` (τ = 100, private o_i)* | *0.33* | *0.148* | ***0.146 (0.199)*** | *0.151 (0.184)* | *0.425* | *0.104* | ***0.174*** | ***0.199*** |
| *ref. `fed-shared`* | — | *0.147* | *0.150 (0.204)* | *0.157 (0.201)* | *0.407* | *0.110* | *0.187* | *0.204* |

Paired contrasts vs. `fed-aligned` (τ = 100):

| Arm | Better on | Worse on |
|---|---|---|
| τ = 300 | 2/8: **nypost.com −0.029**, people.com −0.008 | 1/8: aol.com +0.008 |
| τ = 1000 | 2/8: **nypost.com −0.017**, forbes.com −0.008 | 0/8 |
| τ = 3000 | **3/8: nypost.com −0.025, reuters.com −0.011, people.com −0.009** | **0/8** |

- **All three larger τ improve nypost.com**, as predicted (NR-61). Only τ = 3000 also helps reuters.com.
- The test means are **not monotone in τ** (0.150, 0.151, 0.147). The three differ by ≤ 0.004, which is unresolved.
- **Dev picks τ = 3000** (0.148, the lowest of the four). Its test result (0.147, worst 0.202) is the best no-offset aligned run and essentially ties `fed-aligned-off` (0.146) and `fed-shared` (0.150).
- **Caveat:** it ran on the H200, unlike every other arm in the comparison. Per CAL-13 it needs a seed-1 replicate, ideally on ccc0284, before it enters the design.
- **The risk did not appear.** Weighting the copy-heavy clients more did not flatten h̄_l for the broad clients: τ = 3000 has the best α = 1 cell (0.091) and the highest reach (0.435) of all runs, and aol.com's α = 1 output percentile holds (0.927 vs. 0.932 at τ = 100).

**Re-weighting vs. a private level (CAL-11 request).** nypost.com and reuters.com, test:

| Cell | `fed-aligned` τ = 100 | τ = 300 | τ = 1000 | τ = 3000 | `fed-aligned-off` (o_i, τ = 100) |
|---|---|---|---|---|---|
| nypost.com, output percentile at α = 0 (target 0) | 0.244 | **0.171** | 0.206 | 0.191 | **0.168** |
| nypost.com, α = 0.25 cell | 0.352 | 0.271 | 0.288 | 0.257 | **0.240** |
| reuters.com, output percentile at α = 0 | 0.275 | 0.262 | 0.282 | 0.254 | **0.229** |
| reuters.com, α = 0.25 cell | 0.317 | 0.303 | 0.314 | 0.297 | **0.283** |

- Re-weighting gets nypost.com **most of the way** to what o_i achieves: 0.171–0.206 vs. 0.168 at α = 0.
- It gets reuters.com **less than half the way** (0.254 at best vs. 0.229).
- So the two act **partly as substitutes for nypost.com and as complements for reuters.com**. NR-66 (the τ sweep with o_i) will show whether combining them adds anything.

### 2. NR-60c `fed-shared-soff`: a shared offset slightly hurts under a FedAvg shape

| Arm | Test pct err (worst) | Shared o (r100) | Contrast |
|---|---|---|---|
| `fed-shared-soff` | 0.153 (0.222) | **−0.026** | vs. `fed-shared`: better 0/8, **worse 2/8** (reuters.com +0.018, forbes.com +0.008) |
| | | | vs. `fed-shared-off`: better 1/8 (aol.com −0.005), **worse 3/8** (reuters.com +0.017, nypost.com +0.010, theguardian.com +0.006) |

- As with `fed-aligned-soff` (o = −0.037, NR-65), averaging keeps the shared offset near zero. Under a FedAvg shape it costs reuters.com (+0.018).
- So entry 20's result (shared offset worse than none) **partly repeats** here: it is slightly worse under a FedAvg shape and neutral under aligned.
- **Completed shared/private-offset grid (test pct err):**

| Shape | no offset | private o_i | shared o |
|---|---|---|---|
| FedAvg shape `fed-shared-*` | 0.150 | 0.150 | 0.153 |
| aligned (τ = 100) `fed-aligned-*` | 0.153 | **0.146** | 0.150 |

  The level must be private.

### 3. Metrics beyond pct err (test, selected checkpoint, mean over clients)

**Steering behaviour.**

| Arm | Spearman (worst) | Concordance (near-tie adj.) | Adjacent increase (near-tie adj.) | Near-no-effect rate | Reach | Output range (pct at α = 1 − α = 0) |
|---|---|---|---|---|---|---|
| `fed-shared` | 0.931 (0.881) | 0.909 | 0.719 | 0.000 | 0.407 | 0.692 |
| `fed-shared-off` | 0.928 (0.890) | 0.909 | 0.710 | 0.001 | 0.393 | 0.683 |
| `fed-shared-soff` | 0.926 (0.859) | 0.904 | 0.707 | 0.003 | 0.397 | 0.679 |
| `fed-aligned` | 0.927 (0.873) | 0.907 | 0.718 | 0.005 | 0.421 | 0.694 |
| `fed-aligned-tp300` | 0.931 (0.870) | 0.909 | 0.720 | 0.001 | 0.417 | 0.700 |
| `fed-aligned-tp1000` | 0.930 (0.896) | 0.908 | 0.722 | 0.001 | 0.418 | 0.694 |
| **`fed-aligned-tp3000`** | **0.933 (0.902)** | **0.912** | **0.731** | 0.001 | **0.435** | **0.708** |
| `fed-aligned-off` | 0.929 (0.892) | 0.910 | 0.720 | 0.001 | 0.425 | 0.701 |
| `fed-aligned-soff` | 0.931 (0.884) | 0.909 | 0.717 | 0.003 | 0.405 | 0.698 |
| `local` | 0.912 (**0.805**) | 0.897 | 0.692 | 0.010 | 0.396 | 0.681 |
| `local-off` | 0.886 (0.730) | 0.874 | 0.639 | 0.019 | 0.365 | 0.620 |

- **Ordering, monotonicity and range agree with pct err.**
  - All federated arms rank outputs alike (Spearman 0.926–0.933; concordance 0.904–0.912).
  - The aligned arms with larger τ or a private offset have the widest output range (0.70–0.71) and the most monotone adjacent steps (0.72–0.73).
- **Local is clearly worse at ordering:**
  - worst-client Spearman 0.805 against 0.86–0.90 for federated;
  - adjacent increase 0.692;
  - near-no-effect 1.0%, i.e. more knob changes with no visible effect.
  - `local-off` is worse still.
- No run produced unscorable outputs, and every run's α = 1 output is more extractive than its α = 0 output for at least 99.1% of articles.

**Summary quality** (gap = generated − real summaries at the same α):

| Arm | AlignScore in / out | gap in / out | BERTScore F1 in / out | Length gap in / out (words) | Trigram repetition (out) |
|---|---|---|---|---|---|
| `fed-shared` | 0.781 / 0.730 | −0.023 / +0.046 | 0.900 / 0.889 | +1.8 / +3.2 | 0.006 |
| `fed-shared-off` | 0.787 / 0.734 | −0.017 / +0.050 | 0.901 / 0.889 | +1.8 / +2.2 | 0.006 |
| `fed-shared-soff` | 0.782 / 0.730 | −0.023 / +0.046 | 0.902 / 0.890 | +2.2 / +2.8 | 0.005 |
| `fed-aligned` | 0.791 / 0.737 | −0.013 / +0.053 | 0.901 / 0.890 | +2.4 / +4.9 | 0.007 |
| `fed-aligned-tp300` | 0.785 / 0.735 | −0.019 / +0.051 | 0.902 / 0.889 | +1.6 / +3.8 | 0.006 |
| `fed-aligned-tp1000` | 0.795 / 0.740 | −0.009 / +0.056 | 0.902 / 0.890 | +1.6 / +3.9 | 0.006 |
| `fed-aligned-tp3000` | 0.788 / 0.729 | −0.016 / +0.045 | 0.901 / 0.889 | +2.3 / **+5.1** | 0.006 |
| `fed-aligned-off` | 0.786 / 0.731 | −0.018 / +0.047 | 0.900 / 0.888 | +2.4 / +3.4 | 0.005 |
| `fed-aligned-soff` | 0.778 / 0.736 | −0.027 / +0.053 | 0.901 / 0.890 | +2.3 / +3.8 | 0.006 |
| `local` | 0.786 / 0.722 | −0.018 / +0.039 | 0.901 / 0.889 | +2.4 / **+1.4** | 0.005 |

- **No arm trades quality for steering.**
  - AlignScore spans 0.778–0.795 in support and 0.722–0.740 out of support, with no pattern by design. BERTScore spans 0.900–0.902 / 0.888–0.890. Repetition is 0.4–0.7%, with no empty outputs.
  - The best steering arms (`fed-aligned-off`, `fed-aligned-tp3000`) sit mid-pack on every quality metric.
- **Two systematic patterns:**
  - Generated summaries are slightly *less* consistent with the article than the real summaries in support (gap −0.01 to −0.03), and slightly *more* consistent out of support (+0.04 to +0.06), at the same α.
  - **Out of support, the summaries are longer than the references.** The aligned arms without an offset are longest (+3.8 to +5.1 words); a private offset shortens them (`fed-aligned-off` +3.4, `fed-shared-off` +2.2). `local` is closest (+1.4), but it also reaches least far.
  - Because extractiveness and length are correlated on Newsroom, the extra length out of support is consistent with reaching further (reach 0.42–0.44). The length gap is the one quality cost that scales with steering range.

**LLM judge** (Qwen2.5-7B; 25 summaries per client per α):

| Arm | Faithful in / out | Relevance in / out (1–5) | Coherence in / out (1–5) |
|---|---|---|---|
| `fed-shared` | 0.827 / 0.779 | 3.65 / 3.66 | 4.23 / 4.17 |
| `fed-shared-off` | 0.835 / 0.802 | 3.61 / 3.69 | 4.22 / 4.22 |
| `fed-shared-soff` | 0.837 / 0.792 | 3.62 / 3.66 | 4.21 / 4.16 |
| `fed-aligned` | 0.823 / 0.815 | 3.64 / 3.71 | 4.20 / 4.20 |
| `fed-aligned-tp300` | 0.840 / 0.805 | 3.61 / 3.68 | 4.20 / 4.20 |
| `fed-aligned-tp1000` | 0.828 / 0.785 | 3.62 / 3.67 | 4.19 / 4.21 |
| `fed-aligned-tp3000` | 0.834 / 0.776 | 3.62 / 3.66 | 4.21 / 4.22 |
| `fed-aligned-off` | 0.828 / 0.788 | 3.62 / 3.67 | 4.19 / 4.17 |
| `fed-aligned-soff` | 0.818 / 0.792 | 3.63 / 3.66 | 4.22 / 4.21 |
| `local` | 0.832 / 0.779 | 3.64 / **3.59** | 4.25 / 4.15 |

- All federated arms are within 0.82–0.84 (faithful in), 0.78–0.82 (faithful out), 3.61–3.71 (relevance) and 4.19–4.23 (coherence).
- With about 500 judged summaries per half-split, the binomial SE of faithfulness is about 0.018, so **every federated difference is within noise**.
- `local` has the lowest out-of-support relevance (3.59, gap −0.04 vs. the references, where every federated arm is ≥ 0), a mild signal that local steering beyond support drifts off-topic.

### Conclusions

1. **Two equally good ways to fix aligned's copy-heavy loss:** a private level (`fed-aligned-off`, 0.146) or pool weights that follow the data (`fed-aligned-tp3000`, 0.147, H200). They are unresolved against each other (CAL-13).
   - The private level does more for reuters.com.
   - Re-weighting does more for the top end: reach 0.435 vs. 0.425, α = 1 cell 0.091 vs. 0.104.
   - NR-66 tests the combination.
2. **The level must be private.** A shared o goes to about −0.03 under FedAvg and is neutral (aligned) or slightly harmful (FedAvg shape).
3. **No quality cost** on any metric, apart from longer out-of-support summaries, which grow with steering range.
4. **Federated steers more reliably than local on every behavioural metric.** That holds beyond pct err: worst-client ordering 0.80 vs. 0.86–0.90, monotone steps, no-effect rate and out-of-support relevance.
5. **Before anything enters the design:**
   - seed-1 replicates of `fed-aligned-off`, `fed-aligned-tp3000` (on ccc0284), `fed-shared` and `local` (not launched; waiting on the user, CAL-11/13);
   - NR-64 count and NR-66 to finish.

## NR-68. House style and held-out clients (E2) for the offset/aligned arms: style results; E2 code change; E2 launched (2026-10-08, 16:20)

**Assigned by the coordinator (user request; shared-code grant for `e2_heldout.py`, `fedsteer/adapt.py`, `tests/test_fedsteer.py`).** Arm set as corrected by the user mid-task: `fed-shared` was replaced by its two offset variants, and plain local without an offset was dropped.

| Arm | Run dir (selected round) | Training node |
|---|---|---|
| `fed-aligned-tp3000` | `runs/nr64-fed-aligned-tp3000_20261008-095918_j11210170` (r100) | ccc0465 H200 |
| `fed-aligned-off` | `runs/nr62-fed-aligned-off_20261007-231639_j11209264` (r100) | ccc0284 |
| `fed-shared-off` | `runs/nr60-fed-shared-off_20261007-165434_j11206103` (r100) | ccc0284 |
| `fed-shared-soff` | `runs/nr60-fed-shared-soff_20261008-055340_j11209750` (r100) | ccc0284 |
| `local-off` (house style: the trained arm; E2: the matched local_D reference, below) | `runs/nr60-local-off_20261007-181308_j11206105` (r60) | ccc0284 |
| *context:* `fed-shared` | `runs/nr54_layer_shared_20261006-233527_j11190971` (r100) | ccc0284 |
| *floor:* A2 shared adapter | `runs/exp25_fed_adaptershared_calshared_nooff_cap4k_20261001-213114_j11094904` (r70) | |

### Part 1: house style (`scripts/style_eval.py`, as in archive entry 31; CPU; report `reports/nr68_style_test.txt`)

- **Attribution:** a density-matched publication classifier (cached `runs/_style/style_clf_58467d166c.pkl`, the same classifier as entry 31). It is scored on in-support generated test summaries; chance is 0.125.
- **Feature gap:** the mean |standardized difference| of 11 style features against the client's real summaries at the same α (lower = closer).

| Run | Attribution acc | P(own) | Feature gap |
|---|---|---|---|
| Real test summaries (ceiling) | 0.587 | 0.388 | — |
| `fed-aligned-tp3000` | 0.562 | 0.380 | **0.103** |
| `fed-aligned-off` | 0.562 | 0.381 | 0.110 |
| `fed-shared-off` | 0.572 | 0.387 | 0.105 |
| `fed-shared-soff` | **0.578** | 0.387 | 0.105 |
| `local-off` | 0.570 | 0.378 | 0.120 |
| *`fed-shared` (context)* | *0.566* | *0.382* | *0.107* |
| A2 shared adapter (floor) | 0.409 | 0.295 | 0.190 |

Per client: attribution accuracy / feature gap (in support):

| Client | real | `tp3000` | `aligned-off` | `shared-off` | `shared-soff` | `local-off` | A2 |
|---|---|---|---|---|---|---|---|
| **nypost.com** | 0.705 / — | 0.695 / 0.106 | 0.635 / **0.158** | 0.675 / 0.104 | 0.675 / 0.099 | 0.665 / 0.116 | 0.250 / 0.352 |
| **reuters.com** | 0.705 / — | 0.613 / 0.228 | 0.613 / 0.236 | 0.657 / 0.237 | 0.688 / 0.238 | **0.722** / 0.206 | 0.538 / 0.251 |
| **theguardian.com** | 0.578 / — | 0.569 / 0.045 | 0.569 / 0.054 | 0.569 / 0.058 | 0.556 / 0.060 | 0.525 / 0.099 | 0.294 / 0.133 |
| forbes.com | 0.520 | 0.510 | 0.510 | 0.507 | 0.535 | 0.448 | 0.352 |
| people.com | 0.595 | 0.607 | 0.637 | 0.635 | 0.615 | 0.645 | 0.490 |
| wsj.com | 0.405 | 0.395 | 0.405 | 0.408 | 0.395 | 0.413 | 0.345 |
| aol.com | 0.545 | 0.443 | 0.453 | 0.482 | 0.477 | 0.477 | 0.383 |
| cbc.ca | 0.640 | 0.663 | 0.670 | 0.642 | 0.680 | 0.662 | 0.622 |

**Reading.**
- **Every arm keeps the house style.** Attribution is 0.562–0.578, within 0.01–0.025 of the real-summary ceiling (0.587) and far above A2's 0.409. The feature gap is 0.103–0.120 vs. 0.190 for A2.
- Differences between the five arms are within noise. With about 600–1000 in-support summaries per client, the SE of the 8-client mean is about 0.007, so the spread (0.562–0.578) is about 2 SE and is not interpreted.
- **Neither the calibration design nor the offset affects style**, the entry-31 result again: the attribute goes through D and the calibration, the style through P.
- **Attribution does not track α:** it is flat over α = 0 → 1 in every arm (e.g. `fed-aligned-off` 0.54 / 0.53 / 0.54 / 0.55 / 0.57).
- **Two per-client notes:**
  - `fed-aligned-off` has the largest nypost.com feature gap (0.158 vs. about 0.10 for the others) and lower nypost.com attribution (0.635). Its o_i (−0.62) makes nypost.com's low-α summaries much less extractive (NR-65), into a range where its real summaries are sparse. **A possible cost of reaching nypost.com's low end, not established** (one seed, in-support α only).
  - `local-off` has the weakest theguardian.com (0.525 / 0.099) and forbes.com (0.448), where its steering was also worst (NR-65), and the best reuters.com (0.722).

### Part 2: E2 code change (shared code; grant held; the coordinator was told before the edit)

Developed in a scratch copy and installed in one step at 15:52, after checking that the repo's `e2_heldout.py` and `fedsteer/adapt.py` were unchanged since the copy. No job had started from the tree in between.

- **`fedsteer/adapt.py`:**
  - New `component_groups(model)`: `trainable_parameter_groups` plus two opt-in keys that split `gain` into **`u`** (the gain parameter) and **`offset`** (o).
  - `train_steered` accepts them (learning rate falls back to `lrs["gain"]`).
  - `fit_calibration` gets `components=` (default `("gain", "warp")`, as before).
- **`e2_heldout.py`:**
  - New `e2_plan(fc, lora, snap, init_control, respect_fix_gain=False)`, which decides per setting what a new client trains and which frozen calibration it starts from.
  - Accepts `calibration=aligned` (coverage/consensus still refused) and `fed.private_offset` runs (no longer refused).
  - New opt-in flag `--respect_fix_gain` for private/shared runs.
  - `reset()` also sets or clears the alpha table.
  - The output `e2` block records the components per setting.
- **What a new client trains** (u is never trained wherever the run has `fix_gain`, so s = 1):

| Run type | frozen_D | plugin (after the SFT adapter) | local_D |
|---|---|---|---|
| aligned (+ offset) | P (+ own o_i from 0); D frozen; the run's table h̄_l set exactly as `load_snapshot_into` does; warps unused | fit o_i only (nothing without offset) | P + D_i from 0 + own per-layer warps (+ o_i) |
| shared + `private_offset` | P + own o_i from 0; the server's shape and gain frozen (the server holds no o) | fit o_i only | as above |
| shared (incl. shared offset) | P only; the server's `steer_control.*` frozen (shared o included) | nothing | unchanged (`private, shared, gain, warp`) |
| private (existing) | unchanged: `private, gain, warp` | unchanged: `gain, warp` | unchanged |

- **Tests, 3 new; full suite 79/79 pass in the repo:**
  - `test_e2_aligned_new_client_gets_table_and_own_offset`:
    - the table equals the snapshot's (values and grid);
    - u = 0, s = 1; o_i ≠ 0; warps, D and the shape untouched; P trained;
    - the coefficient at α = 0.5 equals table + o_i;
    - plugin changes only `steer_control.o`.
  - `test_e2_private_offset_and_shared_offset_runs`:
    - `fed-shared-off`-type: o starts at 0 and is trained; the shape and gain stay equal to the server's; u = 0; plugin changes only o.
    - `fed-shared-soff`-type: the loaded o equals the server's and stays frozen; only P trains.
  - `test_e2_plan_keeps_existing_runs_and_gain_rule`:
    - existing private/shared runs get exactly the components the old code trained;
    - u + offset partition the gain group;
    - `--respect_fix_gain` keeps u at 0.
- **The existing paths are bit-identical.** Pristine vs. new code, CPU, 3 training steps, 2 test prompts, telegraph.co.uk n = 16, same scratch config:
  - Shared run (NR-54 A_L): frozen_D, local_D and plugin give identical grids, generated texts, controls and summaries. The plugin `train` field differs only because the second run loaded the cached SFT adapter.
  - Private-calibration run (11041282): frozen_D and local_D are identical. The old-code plugin process was killed twice by the login node's limits, so private plugin was not compared end to end. It calls the same `fit_calibration` on the same parameter list, and the unit test pins `plan["plugin"] == ("gain", "warp")`.
- **Real-snapshot check** (CPU, no training):
  - `fed-aligned-off`: frozen table 112 × 21 equals the snapshot; o = 0, u = 0; trains `private, offset`.
  - `fed-aligned-tp3000`: same table check; trains `private`.
  - `fed-shared-off`: 336 shape tensors equal the server's; o = 0; trains `private, offset`.
  - `fed-shared-soff`: o loaded = **−0.0260** = the server's; trains `private`.
- **Housekeeping.**
  - The CPU comparison wrote one 3-step SFT adapter into `runs/_e2_adapters/`. I removed it and, through a too-broad glob, also four older 3-step smoke-test adapters from 10-01 (`*_n16_w0-0.4_steps3_*`). These are cache files used only by 3-step smoke runs; any such run rebuilds them. No real adapter was touched.
  - `e3_drift.py` was not touched.

### Part 2: E2 jobs (cc only; launch [NR-68_e2.sh](launch/NR-68_e2.sh))

Protocol as archive entries 23/30:
- rotation 0, held-out clients telegraph.co.uk, bbc.com, mashable.com, latimes.com;
- n ∈ {16, 64, 256, 1024, all}, nested subsets;
- steps clip(⌈4n/8⌉, 200, 2000), warm-up 20;
- 200 test articles per client, α grid 0 / 0.25 / 0.5 / 0.75 / 1, density scorer;
- quality and the LLM judge (frozen_D and local_D files) run inside each job (`sbatch/eval_settings.sbatch`).

Resources: `--gres=gpu:A100:1` on dali/IllinoisComputes-GPU (A100-40GB ccc0284 or A100-80GB ccc0388–0390; no H200), `--exclude=ccc0387`, 12 h each.

| Job | Arm | ns | Settings |
|---|---|---|---|
| 11222319 / 20 / 21 / 22 / 23 | `fed-aligned-off` | 16 / 64 / 256 / 1024 / all | frozen_D + **local_D = the `local-off` reference** + plugin (o_i only) |
| 11222324 / 25 | `fed-shared-off` | 16+64+256+1024 / all | frozen_D + plugin (o_i only) |
| 11222326 / 27 | `fed-aligned-tp3000` | 16+64+256+1024 / all | frozen_D |
| 11222329 / 30 | `fed-shared-soff` | 16+64+256+1024 / all | frozen_D |

- **local-off reference:**
  - Run inside the `fed-aligned-off` jobs: D_i from zero, own per-layer warps, o_i trained, u fixed.
  - Its lora and fed settings are identical to the `local-off` arm's config (checked key by key: lora, learning rates, batch, rounds, local steps, warm-up, `fix_gain`, `warp_reg`). It does not depend on the federated run.
  - Plain local_D without offset: not run (dropped by the user).
- **prompt:** reused from the archive rotation-0 E2 (`runs/exp17_fed_calshared_nooff_cap4k_*/evals/*e2_prompt_n{16,64,256,1024}*`, jobs 11088050–53). Same model, max_new_tokens 128, 3 shots, seed 0, 200 prompts, held-out clients, and an identical `alpha_reference.json`. There is no prompt result at n = all, as in the archive.

### Report plan

- For each n: frozen_D of each federated arm vs. the `local-off` reference, per held-out client, with paired bootstrap, mean and worst.
- In- and out-of-support error; quality (AlignScore, BERTScore, length gap) and judge.
- The new clients' learned o_i (frozen_D, plugin, local_D) vs. the participants' o_i (`fed-aligned-off` −0.02 to −0.62, `fed-shared-off` −0.02 to −0.66).
- The house-style table above.
- E2 nodes are named per job.

**NR-68 follow-up 1: nypost.com style gap by α (coordinator, CAL-16).** Report `reports/nr68_style_nypost_by_alpha.txt`, from the per-α style results. nypost.com's support is [0.51, 0.95], so **on the 5-point grid only α = 0.75 is in support**, and the "in-support" gap *is* the α = 0.75 cell.

| Arm | Gap (attribution acc) α = 0 | α = 0.25 | α = 0.5 | **α = 0.75 (in support)** | α = 1 |
|---|---|---|---|---|---|
| `fed-aligned-tp3000` | 0.205 (0.34) | 0.198 (0.40) | 0.221 (0.54) | 0.106 (0.70) | 0.132 (0.64) |
| **`fed-aligned-off`** | 0.268 (0.44) | 0.234 (0.40) | 0.234 (0.56) | **0.158 (0.64)** | **0.171** (0.61) |
| `fed-shared-off` | 0.239 (0.38) | 0.161 (0.37) | 0.197 (0.53) | 0.104 (0.68) | 0.107 (0.68) |
| `fed-shared-soff` | 0.264 (0.42) | 0.227 (0.47) | 0.248 (0.62) | 0.099 (0.68) | 0.113 (0.66) |
| `local-off` | 0.220 (0.38) | 0.180 (0.47) | 0.213 (0.53) | 0.116 (0.67) | 0.134 (0.67) |
| *`fed-shared` (context)* | 0.240 (0.38) | 0.160 (0.40) | 0.226 (0.54) | 0.108 (0.71) | 0.118 (0.67) |

- **Not the low-α explanation.** Every arm's gap is about twice as large below nypost.com's support (0.16–0.27 at α ≤ 0.5) as inside it (0.10–0.16). That is the expected effect of steering below the house range, and it is similar in all arms. The in-support comparison is the fair one, and there `fed-aligned-off` stands out at α = 0.75 (0.158 vs. 0.099–0.116) and α = 1 (0.171 vs. 0.107–0.134), with somewhat lower attribution (0.64 vs. 0.67–0.71).
- **The offset alone does not explain it:** `fed-shared-off` has nearly the same nypost.com o_i (−0.66 vs. −0.62) and a normal gap (0.104). So it comes from the combination of the pooled shape h̄_l and o_i; plausibly P_nypost compensates differently under aligned.
- One seed, about 200 summaries per cell, and no CI on the feature gap: **flagged for the seed-1 replicate of `fed-aligned-off`**, not established.
- The arm-level house-style conclusion stands (every arm keeps the style; attribution is within noise). The one per-client cost candidate is `fed-aligned-off` on nypost.com inside its own range.

**Pooling test for the nypost.com style gap (coordinator, CAL-16); set up 15:58, no new training.**
- **Hypothesis:** at τ = 100 nypost.com holds only about 1/3 of the pooled weight in its own region, so h̄_l follows the broad clients' curve there and P_nypost compensates. If the α = 0.75 / 1 gap returns to about 0.10 at τ = 3000 or count, the pooling weights explain it, which argues for higher τ in the final design.
- **Test:** `style_eval.py` on the finished NR-66 arms (`fed-aligned-off-tp*` / `-cnt`), with `fed-aligned-off` (τ = 100) for reference. Report nypost.com's feature gap and attribution at α = 0.75 and 1.
- A background wait runs it once `fed-aligned-off-tp3000` (11215858) completes; output in `reports/nr68_style_nr66.txt`. The count arm (11215859) is still pending and will be added when it finishes.

**Pooling test result (19:35; three of four NR-66 arms; count still running).** Reports `reports/nr68_style_nr66.txt` and `reports/nr68_style_nypost_by_alpha.txt` (appended). All three NR-66 arms were trained on ccc0284 and dev-selected at r100.

| Arm | nypost.com gap (attribution) α = 0.75 | α = 1 | nypost.com o_i | Test pct err (mean) | All-client attribution / feature gap |
|---|---|---|---|---|---|
| `fed-aligned-off` (τ = 100) | 0.158 (0.635) | 0.171 (0.610) | −0.618 | 0.146 | 0.562 / 0.110 |
| `fed-aligned-off-tp300` | 0.121 (0.655) | 0.156 (0.590) | −0.637 | 0.146 | 0.569 / 0.103 |
| `fed-aligned-off-tp1000` | **0.092 (0.680)** | 0.128 (0.655) | −0.627 | 0.145 | 0.568 / 0.100 |
| `fed-aligned-off-tp3000` | 0.133 (0.630) | 0.189 (0.575) | −0.623 | 0.146 | 0.558 / 0.103 |
| *`fed-shared-off` (reference)* | *0.104 (0.675)* | *0.107 (0.675)* | *−0.663* | *0.150* | *0.572 / 0.105* |

- **The pooling hypothesis is not confirmed.**
  - The gap is **not monotone in τ**: 0.158, 0.121, 0.092, 0.133. It falls to `fed-shared-off`'s level at τ = 1000 and rises again at τ = 3000, where nypost.com's pooled weight is highest of the three. α = 1 behaves the same way (0.171, 0.156, 0.128, 0.189).
  - nypost.com's o_i is unchanged across τ (−0.62 to −0.64).
- **Most likely reading:** the cell-level nypost.com feature gap varies by about ±0.03 between near-identical models (one α cell, about 200 summaries, one seed). The τ = 100 value (0.158) sits at the top of that spread rather than marking a mechanism.
  - The all-client numbers stay flat (attribution 0.558–0.569, gap 0.100–0.110).
  - **Do not use the nypost.com style gap to choose τ.** The seed-1 replicate of `fed-aligned-off` will show whether the τ = 100 value repeats.
- **Side result for NR-66** (full write-up when the count arm is done): **with a private offset, τ does not move steering**. Test pct err is 0.146 / 0.146 / 0.145 / 0.146 for τ = 100 / 300 / 1000 / 3000, against 0.153 → 0.147 without an offset (NR-64). The two fixes are **substitutes**: once o_i sets each client's level, re-weighting h̄_l adds nothing to the mean.

**NR-67 addendum: per-client `fed-aligned-off` vs. `fed-shared-off`** (user question; report `reports/nr67_perclient_alignedoff_vs_sharedoff.txt`; both runs on ccc0284, r100; paired bootstrap from NR-65).
- **Pct err:** `fed-aligned-off` is significantly better on 3/8 clients (people.com −0.011, aol.com −0.010, wsj.com −0.006) and worse on none. The other five are within ±0.003.
- **Out of support:** better on aol.com (−0.024*) and people.com (−0.014*); reuters.com −0.011 (CI touches 0).
- **In support:** within ±0.007, except nypost.com +0.015 (no CI computed for this split).
- **Reach** is higher on 7/8 clients (aol.com +0.10, forbes.com +0.06, nypost.com +0.04). Adjacent monotone steps are better on 5/8 (aol.com +0.04, wsj.com +0.03). Spearman is mixed (±0.01).
- **Quality and judge per client are noisy** (judge about 25 summaries per client per α).
  - Candidates, not established: reuters.com faithfulness out of support 0.745 vs. 0.857; nypost.com out-of-support length gap +7.9 vs. +3.3 words; theguardian.com AlignScore in support 0.705 vs. 0.748.
  - In the other direction: cbc.ca faithfulness 0.860 / 0.883 vs. 0.827 / 0.850.

## NR-69. Complete τ_pool grid for aligned calibration, with and without a private offset (NR-64 + NR-66): with o_i, τ does not matter (2026-10-08, 23:55)

Analysis only; no code or jobs.

**State.**
- All ten arms done: `fed-aligned` and `fed-aligned-off` at τ_pool = 100 (NR-58, NR-62), 300, 1000, 3000 and count (NR-64, NR-66). All dev-selected at **r100**.
- Nodes: ccc0284 (A100-40GB), except `fed-aligned-tp3000` (ccc0465, H200) and both count arms (ccc0390, A100-80GB).
- Reports: `reports/nr69_tau_grid_test.txt` (`compare_runs.py`), `reports/nr69_cells_test.txt` (cells, output percentiles, o_i).
- Single seed. Per CAL-13, mean differences ≤ 0.005 across GPU types are unresolved.

### The grid (test pct err, worst client in brackets; dev pct err at r100 below)

| | τ = 100 | τ = 300 | τ = 1000 | τ = 3000 | count |
|---|---|---|---|---|---|
| nypost.com + reuters.com share of the pool at α = 0.9 | 0.33 | 0.39 | 0.49 | 0.58 | 0.66 |
| **`fed-aligned`** (no offset) | 0.153 (0.213) | 0.150 (0.210) | 0.151 (0.213) | 0.147 (0.202) *H200* | 0.147 (0.207) *A100-80* |
| dev | 0.152 | 0.150 | 0.151 | 0.148 | 0.148 |
| **`fed-aligned-off`** (private o_i) | **0.146 (0.199)** | 0.146 (0.205) | 0.145 (0.204) | 0.146 (0.191) | **0.143 (0.191)** *A100-80* |
| dev | 0.148 | 0.148 | 0.147 | 0.144 | 0.143 |
| o_i: nypost.com / reuters.com / mean | −0.618 / −0.381 / −0.209 | −0.637 / −0.368 / −0.213 | −0.627 / −0.360 / −0.211 | −0.623 / −0.367 / −0.211 | −0.624 / −0.372 / −0.210 |
| *ref. `fed-shared`* | *0.150 (0.204)* | | | | |

### 1. With a private offset, τ does not matter

Paired per-client contrasts against τ = 100, within the offset row:

| Contrast | Better on | Worse on |
|---|---|---|
| τ = 300 − τ = 100 | 1/8 (wsj.com −0.006) | 0/8 |
| τ = 1000 − τ = 100 | 1/8 (nypost.com −0.009) | 0/8 |
| τ = 3000 − τ = 100 | 0/8 | 0/8 |
| count − τ = 100 | 2/8 (cbc.ca −0.007, nypost.com −0.007) | 0/8 |
| count − τ = 3000 | 2/8 (cbc.ca −0.010, nypost.com −0.008) | 0/8 |

- **No τ is worse than τ = 100 on any client.** All means are within 0.003 (0.143–0.146), which is unresolved.
- **The level is identified independently of the pooling:** nypost.com −0.62 to −0.64, reuters.com −0.36 to −0.38, mean −0.21 at every τ.
- Count pooling is nominally the best cell of the grid: test 0.143 (worst 0.191), dev 0.143. The margin is ≤ 0.003 on the mean, small per-client gains, and a different GPU type (A100-80GB vs. τ = 100's A100-40GB; same architecture). It is not a resolved winner.
- **Robustness result: with o_i, the design is insensitive to the pooling weight.** No τ needs tuning.

### 2. Without an offset, τ matters, mostly through nypost.com

- count − τ = 100: better on 2/8 (**nypost.com −0.029**, people.com −0.009), worse on 0/8.
- τ = 3000 (NR-67) − τ = 100: better on 3/8 (nypost.com −0.025, reuters.com −0.011, people.com −0.009), worse on 0/8.
- The mean goes 0.153 → 0.147 at τ = 3000 and at count.

### 3. Offset vs. pool re-weighting: substitutes on the mean, but o_i still adds on the copy-heavy clients

Offset minus no offset, at each τ:

| τ | Mean | Better on | Worse on |
|---|---|---|---|
| 100 (NR-65) | −0.007 | nypost.com −0.034, reuters.com −0.014 | — |
| 300 | −0.004 | wsj.com −0.010, nypost.com −0.008 | — |
| 1000 | −0.006 | nypost.com −0.027, reuters.com −0.009 | forbes.com +0.006 |
| 3000 | −0.001 (GPU types differ) | reuters.com −0.011, nypost.com −0.009 | aol.com +0.006 (out of support +0.016), cbc.ca +0.005 |
| count | −0.004 | **reuters.com −0.016 (out of support −0.034)**, nypost.com −0.012 | — (aol.com out of support +0.012) |

- Re-weighting alone gets **nypost.com** to the offset's level. Output percentile at α = 0 (target 0): 0.172 with count pooling vs. 0.159–0.168 with o_i, against 0.244 at τ = 100 without.
- It does **not** fix **reuters.com**: 0.282 with count pooling vs. 0.214–0.257 with o_i.
- So the private level is still needed for reuters.com at every τ. On the 8-client mean the gap closes (substitutes), because the costs on aol.com and cbc.ca at τ = 3000 offset the reuters.com and nypost.com gains.

### 4. Other metrics

- All ten arms are within noise on AlignScore (0.784–0.796 in support, 0.725–0.740 out), BERTScore (0.900–0.902), judge faithfulness (0.81–0.84 in support, 0.78–0.82 out) and coherence (4.17–4.23).
- Out-of-support length gaps: the offset arms +3.2 to +3.9 words, the no-offset arms +3.8 to +5.1. The offset shortens out-of-support summaries at every τ except 300.
- Spearman (0.927–0.937) and near-tie concordance (0.907–0.914) are flat. Reach is 0.418–0.435 everywhere (`fed-shared` 0.407).

### Conclusions

1. **Design recommendation:** keep `fed-aligned-off` at **τ = 100** (no reason to change; insensitive to τ), or count pooling if one setting is to be fixed. Both need the seed-1 replicate before the choice is final (CAL-13).
2. **The private offset is the more robust fix.** It handles both copy-heavy clients and makes the pooling weight irrelevant. Re-weighting handles nypost.com but not reuters.com.
3. The coordinator's NR-68 style check found no τ effect on house style either (all-client attribution 0.558–0.569).

## NR-70. E2 interim: `fed-aligned-off`, held-out clients at n = 16–1024 (2026-10-08, 23:45)

Analysis only.

**State (23:34).**
- **Done**, all on ccc0284 (A100-40GB): `fed-aligned-off` E2 at n = 16 / 64 / 256 / 1024 (11222319–22). Each covers frozen_D, local_D (= the `local-off` reference) and plugin on the 4 rotation-0 held-out clients, plus quality and judge (frozen_D and local_D).
- **Running:**
  - `fed-aligned-off` n = all (ccc0284);
  - `fed-shared-off` n ≤ 1024 and n = all (ccc0284);
  - `fed-aligned-tp3000` n ≤ 1024 (ccc0388) and n = all (ccc0390), both A100-80GB.
- **Pending:** `fed-shared-soff` (both).
- Reports: `reports/nr70_e2_alignedoff_interim.txt` (`scripts/e2_report.py`), `reports/nr70_e2_offsets.txt`.

### The E2 curve (test, 200 articles per held-out client; mean over the 4 clients)

| n | frozen_D (in / out) | local-off reference (in / out) | plugin (in / out) | frozen_D better than local-off on | *archive (entry 30): method frozen_D / local_D / prompt* |
|---|---|---|---|---|---|
| 16 | **0.174** (0.197 / 0.150) | 0.315 (0.288 / 0.339) | 0.198 (0.245 / 0.146) | **4/4** (−0.13 to −0.16) | *0.179 / 0.295 / 0.411* |
| 64 | **0.167** (0.177 / 0.152) | 0.294 (0.269 / 0.331) | 0.183 (0.219 / 0.128) | **4/4** (−0.11 to −0.14) | *0.170 / 0.260 / 0.388* |
| 256 | **0.145** (0.147 / 0.141) | 0.277 (0.266 / 0.292) | 0.178 (0.216 / 0.121) | **4/4** (−0.13 to −0.14) | *0.146 / 0.225 / 0.411* |
| 1024 | **0.141** (0.146 / 0.134) | 0.231 (0.225 / 0.239) | 0.196 (0.229 / 0.146) | **4/4** (−0.04 to −0.14) | *0.139 / 0.192 / 0.387* |

- Per client, frozen_D − local-off is significant on every client at every n, overall and out of support (95% bootstrap CI excludes 0).
- The smallest gap is mashable.com at n = 1024 (−0.037), the largest bbc.com at n = 16 (−0.162).
- **Prompt baseline:** the archive results (0.39–0.41) apply unchanged, since the setting is identical (NR-68).

### What the results say

**1. Same data efficiency as the archive method, from the new design.** A new client of `fed-aligned-off` fits only its adapter and one scalar o_i, yet scores 0.174 / 0.167 / 0.145 / 0.141. The archive method (one shared shape, learned gain, no offset) scored 0.179 / 0.170 / 0.146 / 0.139. These are different runs and no paired test is possible, but the curves are the same within about 0.005. So the calibration changes (gain fixed at 1, per-layer pooled shapes, private offset) cost new clients nothing. With 16 pairs a new client already beats the local reference trained on 1024 pairs (0.174 vs. 0.231).

**2. One scalar o_i is learnable from 16 pairs** (coordinator follow-up 2). Held-out o_i under frozen_D:

| Client | n = 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| telegraph.co.uk | −0.015 | +0.020 | +0.013 | +0.058 |
| bbc.com | −0.044 | −0.032 | −0.019 | +0.015 |
| mashable.com | −0.046 | −0.033 | −0.030 | −0.004 |
| latimes.com | −0.063 | −0.059 | −0.062 | −0.025 |

- It is stable across n (within about ±0.07 from n = 16 to n = 1024) and drifts slightly upward with more data.
- It sits **near 0, inside the upper end of the participants' range** (−0.62 to −0.02, mean −0.21).
- This fits the rotation-0 caveat of archive entry 23: the held-out clients are broad (supports 0.02–0.99), like the broad participants, so they need little level shift. **Whether a copy-heavy new client (nypost.com-like, o_i about −0.6) can learn its large offset from 16 pairs is untested here.** It would need rotation 2 (nypost.com and theguardian.com held out) or `--alpha_window`.

**3. plugin: the offset repairs most of the anchor shift but not all of it.**
- plugin scores 0.18–0.20, against 0.34–0.37 for archive plugin with frozen calibration and 0.23–0.28 with a refitted private calibration (entry 30). A single fitted offset now does most of the repair.
- **The learned offsets show the mechanism:** plugin's o_i is −0.32 to −0.56, against about 0 for frozen_D. An adapter trained without the knob absorbs the client's typical extractiveness, and o_i must pull the level back down.
- plugin is still worse than frozen_D in support (0.22–0.25 vs. 0.15–0.20) and overall at every n. Out of support it is *better* than frozen_D (0.121–0.146 vs. 0.134–0.152). A displaced anchor plus a negative offset happens to land the extremes well while miscalibrating the middle.
- Training the adapter with the knob in the loop (frozen_D) remains the right protocol.

**4. The local-off reference is weak, for the reason found in NR-65.**
- Its o_i drifts upward with n: +0.23 → +0.36 → +0.65 → +1.05 for latimes.com, and up to +1.63 for telegraph.co.uk at n = 1024, the same non-identifiability with a private D_i.
- It is worse than the archive's local_D (learned gain, no offset, one shape: 0.295 / 0.260 / 0.225 / 0.192).
- Against the *archive* local_D, frozen_D still wins by 0.05–0.12 at every n.
- Local at small n barely steers:
  - reach 0.18–0.28, against 0.33–0.54 for frozen_D;
  - Spearman 0.43–0.77, against 0.90–0.94;
  - a near-no-effect rate at the endpoints of 0.04–0.21, against 0.000.

**5. Quality.**

| Setting | AlignScore gap in support | Out of support | Out-of-support length gap | Judge faithfulness in / out | Judge relevance gap out |
|---|---|---|---|---|---|
| frozen_D | −0.03 to +0.03 | +0.05 to +0.09 | **+10.2 words at n = 16**, falling to +4.6 at n = 1024 (the small-n length cost seen in entry 30) | 0.78–0.83 / 0.73–0.78 | −0.02 to 0.00 |
| local-off reference | | | | 0.68 at n = 16, 0.79–0.83 / 0.70–0.79 after | **−0.12 at n = 16**, −0.07 at n = 64 |
| plugin | | | **+9 to +15 words at every n** | | |

Remaining: n = all for `fed-aligned-off`; `fed-shared-off`, `fed-aligned-tp3000` and `fed-shared-soff` at every n; then the cross-arm frozen_D comparison per n.

**Coordinator request for the final E2 table (CAL-16).**
- Add a second local reference, **`local-gain-1shape`**: the archive's local_D from E2 on 11063910 (`runs/exp17_fed_calshared_nooff_cap4k_*/evals/*e2_local_D_n*`). It is the strongest local baseline measured.
  - Values: 0.295 / 0.260 / 0.225 / 0.192 / **0.151** at n = 16 / 64 / 256 / 1024 / all.
  - Its lora settings (checked in that run's config): learned gain (s ∈ [0.25, 4]), **one** kumaraswamy_mix shape per client, no offset. Training used warp_reg 0.01; E2's local_D has no warp penalty.
- **Protocol caveat:** same rotation (0), held-out clients, nested-n rule, steps, α reference (identical `alpha_reference.json`), 200 test articles and density scorer. But a different run family, code version and date (10-01): numbers only, no paired test.
- The copy-heavy newcomer test (rotation 2 / `--alpha_window`) is **not** launched; the coordinator is raising it with the user.

## NR-71. Baselines B1 / B3 / B4 for `fed-aligned-off`, launched (2026-10-09, 00:20)

**Assigned by the coordinator (user request).** The method is `fed-aligned-off` (NR-62, τ = 100, `runs/nr62-fed-aligned-off_20261007-231639_j11209264`, snapshot r100). Archive protocol of entries 19/22 via `sbatch/eval_baselines.sbatch`: test split, 200 articles per client, density scorer, `max_new_tokens` 128, quality scoring after each step.

| Baseline | What | Job |
|---|---|---|
| smoke | `SMOKE=1`, steps b1 + b4 + b3 on the `fed-aligned-off` and `local-off` snapshots | **11236750** |
| **B1 k = 0** | prompting, the client's model (adapter + calibration, direction removed) | 11236751 (`nr71-b1-k0`) |
| **B1 k = 3** | the same with 3 own-publication examples nearest the target α | 11236752 (`nr71-b1-k3`) |
| B1 k = 3, base model | **reused from the archive**: `runs/exp11_cap4k_base_*/evals/eval_round_0100_b1_prompt_k3_base__20261001-104506_j11065061.json` | — |
| **B4 federated CAA** | middle layer, 200 examples per side, 30 dev articles to fit the gain; **widened grid** 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.4, 3.2 | 11236753 (`nr71-b4-caa`) |
| **B3 from `local-off`** | merge `runs/nr60-local-off_20261007-181308_j11206105` at its dev-selected **r60**; each client keeps its adapter, per-layer shapes and offset | 11236754 (`nr71-b3-local-off`) |
| **B3 from `local`** | merge `runs/nr60-local_20261007-231639_j11206106` at **r100** (secondary) | 11236755 (`nr71-b3-local`) |

- The five real jobs depend on the smoke job (`afterok`). All are cc only, with `--gres=gpu:A100:1`, `--exclude=ccc0387`, 12 h. Launch [NR-71_baselines.sh](launch/NR-71_baselines.sh).

**Checks before launch:**
- **Reuse of B1 k = 3 base is valid.**
  - The archive file's arguments: test, density, shots 3, base model, 200 prompts, `max_new_tokens` 128.
  - Same model, data, `clients_file`, rotation, `max_train_per_client` (4000), seed 0 (so the identical training subset the shots come from), participants, prompt and target lengths, and an **identical `alpha_reference.json`**.
  - `fedsteer/baselines.py` changed since 10-01 only by adding molecule and math templates (the news template is still selected for news records) and passing the empty `chat_template_kwargs`. The base model does not depend on the run.
- **Aligned snapshot under `load_snapshot_into`** (CPU, Llama-3.2-1B, clients nypost.com and aol.com):
  - the table equals the snapshot's;
  - o_i loaded = snapshot (−0.618, −0.125);
  - with D, the logits move with α (max |Δ| 4.6–5.3);
  - **with B_d = 0, the logits are bit-identical at α = 0 and 1 and with the offset removed** (max |Δ| = 0). So B1 and B4 see the adapter alone, as intended.
- **No shared-code change.**
  - `sbatch/eval_baselines.sbatch` (newsroom-owned) gets two opt-in variables. `B1_SETS` is a ";"-separated list of B1 variants, defaulting to the original three. `CAA_ARGS` is extra B4 arguments, outside smoke mode, default empty.
  - Both are passed through the environment with `--export=ALL`, because `sbatch --export` splits values on commas (entry 19a).

**Report plan:**
- A table in the archive entry-22 format: `fed-aligned-off`, `local-off`, `local`, B3 × 2, B4, B1 × 3. Columns: pct err (worst), in / out of support, reach, Spearman, near-tie, endpoint near-tie, AlignScore gap in / out, length gap.
- Paired per-client contrasts of `fed-aligned-off` vs. each baseline.
- B4: the gain each client picks, flagging any client still at 3.2.
- B3: which source is stronger. `local-off`'s offsets drift (mean +0.78), which may weaken its merged direction.

**Pending, low priority (coordinator, cross-dataset):** a restricted aligned-calibration disagreement for NR-62 `fed-aligned-off` (r100 snapshot) and NR-58 `fed-aligned`.
- The logged `disagreement` (`fed.py` `_update_aligned`) is a weighted SD averaged over every (layer, grid) point, which understates disagreement where only one client has weight.
- The new measure is the SD over grid points where ≥ 2 clients each hold ≥ 5% of the point's weight, plus the number of qualifying points.
- molecule-iter1 is writing the post-hoc script under `scripts/`; the coordinator will pass on its name. Run it when it exists (CPU, no code change) so Newsroom and ChEMBL use the same measure.

**Restricted disagreement, NR-62 vs. NR-58** (coordinator, cross-dataset; `scripts/disagreement_report.py` by molecule-iter1, read before running; CPU; r100; report `reports/nr71_disagreement_restricted.txt`). Restriction: grid points where ≥ 2 clients each hold ≥ 5% of the point's weight.

| Run | Unrestricted (logged, r99) | Restricted | Qualifying grid points | nypost.com support gap | theguardian.com support gap | Other six |
|---|---|---|---|---|---|---|
| `fed-aligned` (NR-58) | 0.02841 (0.02841, match) | 0.02841 | **21/21** | 0.051 | 0.045 | 0.014–0.017 |
| `fed-aligned-off` (NR-62) | 0.02336 (0.02336, match) | 0.02336 | **21/21** | **0.033** | **0.035** | 0.012–0.018 |

- **No dilution on Newsroom at τ = 100 either** (factor 1.00×). Every grid point keeps 7–8 clients above the 5% floor. Only nypost.com (17/21 points) and theguardian.com (16/21) drop out at the ends of the scale. That is the same as the coordinator found at τ = 3000. The logged scalar is a fair measure here.
- **With a private offset, the copy-heavy and narrow clients' own shapes sit closer to the pooled table.** nypost.com's support gap falls from 0.051 to 0.033 (−35%) and theguardian.com's from 0.045 to 0.035 (−22%); the others are unchanged. Overall disagreement falls from 0.028 to 0.023.
- This is consistent with NR-63's shape-distance result for private shapes (nypost.com −41% with an offset): the level absorbs part of what would otherwise show up as a shape difference. One seed.

**o_i vs. support floor (coordinator, cross-dataset, CAL-21; CPU).** Support floor = the client's 5th percentile on the global α scale (`support[0]` of the test eval; `fedsteer/data.client_support`). Plain test eval at the selected round. Report `reports/nr71_offset_vs_floor.txt`.

| Arm (round) | Spearman (n = 8) | Pearson | Without nypost.com (n = 7): Spearman / Pearson | o_i vs. **median α**: Spearman / Pearson |
|---|---|---|---|---|
| `fed-aligned-off` NR-62 (r100) | −0.571 | −0.894 | −0.357 / −0.609 | **−0.786 / −0.854** |
| `fed-shared-off` NR-60 (r100) | −0.643 | −0.913 | −0.464 / −0.631 | **−0.833 / −0.851** |
| `local-off` NR-60 (r60) | −0.500 | −0.481 | −0.250 / −0.339 | — |
| *`fed-aligned-off-tp3000` NR-66 (cross-check)* | *−0.429* | *−0.908* | *−0.143 / −0.613* | — |

(support floor, o_i) pairs, sorted by floor:

| Client | Floor | Median α | o_i `fed-aligned-off` | o_i `fed-shared-off` | o_i `local-off` |
|---|---|---|---|---|---|
| forbes.com | 0.029 | 0.28 | −0.149 | −0.166 | +0.471 |
| people.com | 0.033 | 0.32 | −0.133 | −0.139 | +0.523 |
| wsj.com | 0.036 | 0.47 | −0.097 | −0.097 | +0.457 |
| theguardian.com | 0.045 | 0.27 | −0.021 | −0.020 | +1.101 |
| aol.com | 0.062 | 0.55 | −0.125 | −0.143 | +0.624 |
| reuters.com | 0.100 | 0.72 | −0.381 | −0.376 | +0.302 |
| cbc.ca | 0.106 | 0.57 | −0.151 | −0.173 | +0.463 |
| nypost.com | 0.506 | 0.85 | −0.618 | −0.663 | +0.242 |

- **On Newsroom the support floor is a poor descriptor of the level; the median is a good one.**
  - The floors bunch at 0.03–0.11 except nypost.com (0.51). The large |Pearson| with a weak Spearman shows that nypost.com carries the floor fit; without it, Spearman is −0.36 / −0.46.
  - reuters.com has the second-largest offset (−0.38) with an unremarkable floor (0.10): it is copy-heavy through its median (0.72), not its floor.
  - Against the client's **median α**, the ordering is near-monotone in both federated arms (Spearman −0.79 / −0.83; −0.68 / −0.75 without nypost.com). o_i tracks *where a client's data sits*, which on ChEMBL coincides with the floor.
- **This matches ChEMBL's shared vs. private split:**
  - the federated arms (shared or pooled shape) are strongly correlated;
  - `local-off` is weak and of opposite sign in level (all o_i positive and drifting, NR-65). Its −0.50 Spearman mostly reflects nypost.com and reuters.com having the smallest positive offsets.
- The τ = 3000 cross-check reproduces molecule-iter1's numbers exactly (−0.43 / −0.91).

**Bug in `scripts/offset_support_report.py` (molecule-iter1's; reported, not edited).**
- Its pattern `eval_round_[0-9]*__*.json` also matches E2 and baseline evals, and it reads the alphabetically last file.
- For NR-62 and `fed-shared-off` that is `eval_round_0100_e2_plugin_nall__…`: the 4 *held-out* clients' plugin offsets. It printed n = 4 and Spearman −1.000.
- The `local-off` row is correct (no E2 files in that run). The numbers above use the strict pattern `eval_round_[0-9][0-9][0-9][0-9]__*.json` (the archive entry-30 fix in `compare_runs.py`).

**Correction to the `local-off` reading above (molecule-iter1 via coordinator, CAL-21; checked here).**
- Rerun with the fixed `scripts/offset_support_report.py` (strict 4-digit glob; refuses to guess between rounds); report `reports/nr71_offset_support_fixed_script.txt`.
- It reproduces my numbers for the federated arms:
  - `fed-aligned-off` (r100): floor −0.571 / −0.894; median α −0.786 / −0.854.
  - `fed-shared-off` (r100): floor −0.643 / −0.913; median α −0.833 / −0.851.
- **`local-off` is evaluated at its dev-selected round 60, not 100.** Its o_i vs. median α is **Spearman −0.786, Pearson −0.681**: the same ordering as the federated arms. I had not computed this column for `local-off`.
- On dev across rounds the ordering holds or strengthens while the level drifts upward:

| Round | Spearman | o_i range | Mean o_i |
|---|---|---|---|
| r30 | −0.667 | −0.01 to +0.27 | +0.08 |
| r60 | −0.786 | +0.24 to +1.10 | +0.52 |
| r80 | −0.857 | +0.32 to +1.47 | +0.73 |
| r100 | −0.857 | +0.35 to +1.58 | +0.78 |

- **So `local-off` learns the right relative structure at an unidentified absolute level**, a common upward drift. This is not "no structure without sharing".
- **The refined claim, consistent on both datasets:** sharing or pooling the shape makes o_i *correctly levelled* (negative, bounded, stable from about round 30, monotone in median α). A private shape leaves the absolute level unidentified: it drifts with D_i and P_i (NR-65), even though the ordering across clients is still learned.

## NR-72. E2 final: held-out clients join every offset/aligned arm equally well, far ahead of local (2026-10-09, 11:40)

Analysis only.

**State.**
- All 11 NR-68 E2 jobs are done, with no errors.
- Nodes: `fed-aligned-off` and `fed-shared-off` on ccc0284 (A100-40GB); `fed-aligned-tp3000` on ccc0388 / ccc0390 (A100-80GB); `fed-shared-soff` on ccc0284.
- Reports:
  - `reports/nr72_e2_n{16,64,256,1024,all}.txt`: `compare_runs.py` with explicit E2 file patterns; paired bootstrap over articles. Every file holds the same 4 held-out clients, the same 200 test articles each and an identical α reference.
  - `reports/nr72_e2_offsets_quality.txt`.
- Protocol: rotation 0, held-out telegraph.co.uk, bbc.com, mashable.com, latimes.com; nested n; steps clip(⌈4n/8⌉, 200, 2000).
- **`local-gain-1shape`** = the archive's local_D (learned gain, one shape, no offset; 11088050–54; same protocol, different code date). Shown for its means only, without paired tests (CAL-16).

### Mean pct err over the 4 held-out clients (worst in brackets)

| n | `fed-aligned-off` | `fed-shared-off` | `fed-aligned-tp3000` | `fed-shared-soff` | **local-off** (reference) | *`local-gain-1shape`* | plugin, aligned-off | plugin, shared-off | *prompt (archive)* |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.174 (0.195) | 0.173 (0.187) | 0.177 (0.197) | 0.174 (0.188) | 0.315 (0.332) | *0.295 (0.325)* | 0.198 | 0.200 | *0.411* |
| 64 | 0.167 (0.191) | 0.168 (0.200) | 0.165 (0.180) | 0.163 (0.181) | 0.294 (0.322) | *0.260 (0.287)* | 0.183 | 0.183 | *0.388* |
| 256 | 0.145 (0.187) | 0.148 (0.187) | 0.141 (0.182) | 0.148 (0.201) | 0.277 (0.319) | *0.225 (0.247)* | 0.178 | 0.179 | *0.411* |
| 1024 | 0.141 (0.177) | 0.139 (0.167) | 0.138 (0.184) | 0.138 (0.172) | 0.231 (0.319) | *0.192 (0.207)* | 0.196 | 0.198 | *0.387* |
| all | **0.132 (0.159)** | 0.132 (0.167) | 0.130 (0.164) | 0.134 (0.168) | 0.167 (0.230) | *0.151 (0.176)* | 0.212 | 0.212 | — |

### What the results say

**1. Every federated arm beats local on every held-out client at every n.**
- frozen_D − `local-off` is significant on **4/4 clients at every n for all four arms** (20 of 20 arm × n cells).
- Against the strongest local baseline measured (`local-gain-1shape`), every arm is ahead in mean at every n: by 0.12 at n = 16, 0.09–0.10 at n = 64, about 0.08 at n = 256, about 0.05 at n = 1024 and about 0.02 at n = all.
- **16 pairs with the frozen direction (0.173–0.177) beat local trained on 1024 pairs (0.192–0.231).**
- At full data, every new client (0.130–0.134) is better than the participants' own test error (0.143–0.150).

**2. E2 does not separate the calibration designs.**
- The four arms are within 0.005 of each other at every n (unresolved, CAL-13).
- Pairwise contrasts are mixed:
  - `fed-aligned-off` − `fed-shared-off`: 1 better / 1–2 worse;
  - `fed-aligned-off` − `fed-aligned-tp3000`: 0–1 better / 0–2 worse;
  - `fed-shared-off` − `fed-shared-soff`: 0–1 / 0–1.
- **For broad new clients, how the shared calibration was learned does not matter**, and neither does the client's own o_i. `fed-shared-soff`, which inherits the frozen shared o = −0.026 and trains only its adapter, does as well as the arms where the client fits its own o_i.
- This follows from rotation 0's held-out clients being broad (next point).

**3. Held-out o_i stays near 0, the participants' broad-client level.** Mean (range) over the 4 clients under frozen_D:

| Arm | n = 16 | 256 | all |
|---|---|---|---|
| `fed-aligned-off` | −0.042 (−0.063 to −0.015) | −0.025 | +0.024 (−0.009 to +0.076) |
| `fed-shared-off` | −0.045 (−0.066 to −0.017) | −0.027 | +0.018 |

Participants: −0.02 to −0.66. One scalar is stable from n = 16 on. Whether a copy-heavy newcomer can learn o_i ≈ −0.6 is untested (the coordinator is raising it with the user).

**4. plugin stays well behind frozen_D, and gets *worse* with more data.**
- 0.198 → 0.183 → 0.178 → 0.196 → **0.212** at n = all, while frozen_D improves to 0.132.
- frozen_D − plugin is significant on 3/4 clients at n ≤ 64 and on 4/4 from n = 256.
- With more SFT data the knob-unaware adapter absorbs more of the client's level, and a single offset cannot undo it.
- **Training the adapter with the knob in the loop is essential**, more so the more data a client has.

**5. Local with an offset is the weaker local reference** (NR-65's drift again). `local-off` is below `local-gain-1shape` at every n (0.315 vs. 0.295 … 0.167 vs. 0.151). The conclusion holds against either.

**6. Quality and judge.**
- AlignScore gap to same-α references:
  - federated: −0.04 to +0.03 in support, +0.05 to +0.09 out of support;
  - `local-off`: up to +0.14 out of support.
- **Out-of-support length gap:**
  - federated, small n: +10 to +13 words at n = 16, falling to +4 to +6 at n = all;
  - local: +1 to +6 words throughout.
  - This is the one quality cost of joining with the frozen direction at very small n (seen in archive entry 30 too).
- **Judge:**
  - faithfulness in / out of support: federated 0.77–0.78 / 0.71–0.73 at n = 16, rising to 0.82–0.86 / 0.76–0.79 at n = all; `local-off` 0.68 / 0.70 at n = 16, 0.84 / 0.80 at n = all;
  - out-of-support relevance gap: federated −0.05 to +0.03 at every n; `local-off` −0.12 at n = 16; `local-gain-1shape` −0.14 at n = 256;
  - coherence: 4.17–4.30 everywhere.

**Conclusion (C2, portability).** A new client joins any of the four offset/aligned designs with a frozen direction and a frozen shared calibration, fitting only its adapter (plus one scalar), and gets the same data-efficiency curve as the archive method. It beats local training on every client at every data size; 16 pairs beat 1024 local pairs. **E2 neither favours nor penalizes any of the four calibration designs** on these broad held-out clients.

## NR-73. Baselines B1 / B3 / B4 for `fed-aligned-off`: results (2026-10-09, 11:50)

Analysis only.

**State.**
- NR-71 jobs all done, with no errors:
  - B1 k = 0: 11236751, ccc0284;
  - B1 k = 3: 11236752, ccc0388;
  - B4: 11236753, ccc0284;
  - B3 from `local-off`: 11236754, ccc0284;
  - B3 from `local`: 11236755, ccc0390.
- The smoke job 11236750 (ccc0388) passed.
- Report `reports/nr73_baselines_test.txt` (`compare_runs.py`; real-job files selected by job ID, because the smoke files sit in the same directories; paired bootstrap over articles).
- Test split, 200 articles per client.

### Table (archive entry-22 format)

| | Pct err (worst) | In-support | Out-of-support (worst) | Reach | Spearman (worst) | Near-tie | Endpoint near-tie | AlignScore in / out (gap) | Length gap in / out |
|---|---|---|---|---|---|---|---|---|---|
| **`fed-aligned-off`** | **0.146 (0.199)** | **0.135** | **0.151 (0.184)** | **0.425** | **0.929 (0.892)** | 0.188 | 0.001 | 0.786 / 0.731 (−0.02 / +0.05) | +2.4 / +3.4 |
| `local` | 0.157 (0.244) | 0.139 | 0.159 (0.274) | 0.396 | 0.912 (0.805) | 0.219 | 0.010 | 0.786 / 0.722 (−0.02 / +0.04) | +2.4 / +1.4 |
| `local-off` (r60) | 0.178 (0.266) | 0.154 | 0.186 (0.300) | 0.365 | 0.886 (0.730) | 0.265 | 0.022 | 0.806 / 0.761 (0.00 / +0.08) | +3.4 / +4.3 |
| **B3 merge of `local`** (r100) | 0.181 (0.256) | 0.152 | 0.198 (0.286) | 0.288 | 0.881 (0.701) | 0.168 | 0.011 | 0.723 / 0.721 (−0.08 / +0.04) | −2.2 / −1.4 |
| B3 merge of `local-off` (r60) | 0.227 (0.280) | 0.203 | 0.243 (0.342) | 0.236 | 0.811 (0.579) | 0.232 | 0.021 | 0.729 / 0.731 (−0.08 / +0.05) | −0.8 / +0.8 |
| **B4 federated CAA** (layer 8, widened grid) | 0.253 (0.301) | 0.213 | 0.289 (0.359) | 0.230 | 0.780 (0.700) | 0.042 | 0.000 | 0.602 / 0.512 (**−0.20 / −0.17**) | −0.2 / **+11.9** |
| **B1 prompt, k = 0** (client model) | 0.381 (0.395) | 0.250 | 0.498 (0.519) | 0.034 | **−0.029** | 0.716 | **0.701** | 0.627 / 0.629 (−0.18 / −0.05) | −3.0 / −7.0 |
| **B1 prompt, k = 3** (client model) | 0.383 (0.406) | 0.255 | 0.498 (0.523) | 0.054 | 0.032 | 0.178 | 0.164 | 0.578 / 0.582 (−0.23 / −0.10) | −3.1 / −7.3 |
| B1 prompt, k = 3, base model (archive 11065061) | 0.341 (0.355) | 0.243 | 0.467 (0.519) | 0.117 | 0.077 | 0.003 | 0.000 | 0.609 / 0.631 (−0.20 / −0.05) | **+82 / +72** |

### Paired per-client contrasts: `fed-aligned-off` − baseline

| Baseline | Better on | Worse on | Notes |
|---|---|---|---|
| B1 k = 0 / k = 3 / k = 3 base | **8/8** each | 0/8 | |
| B4 CAA | **7/8** | 0/8 | reuters.com tie overall (+0.005), better out of support (−0.057*); nypost.com −0.025* |
| B3 merge of `local-off` | **7/8** | 1/8 | reuters.com +0.029*, out of support +0.003 n.s. |
| B3 merge of `local` | **7/8** | 1/8 | reuters.com, 0.199 vs. 0.171 |
| `local` (for reference) | 4/8 | 0/8 | NR-65 |

### What the results say

1. **B1 prompting does not steer.**
   - Spearman ≈ 0 for the client model with or without shots; at k = 0, 70% of articles give near-identical text at α = 0 and 1 (endpoint near-tie 0.70).
   - The base model with 3 shots writes 72–82 words longer than the reference (about 110 tokens).
   - This is unchanged from archive entry 22 (0.378 / 0.378 / 0.341 there). The new client models (aligned calibration + offset) behave like the old ones once the direction is removed.
2. **B4 CAA steers partially and damages quality; the wider gain grid did not rescue it.**
   - Chosen gains: 0.8 for six clients (theguardian.com, people.com, wsj.com, aol.com, cbc.ca, nypost.com) and 0.5 for forbes.com and reuters.com.
   - **No client is at 3.2 (or above 0.8):** every choice is an interior optimum on the dev fit. Archive entry 22's under-tuning concern (6/8 at the old top gain 0.8) is resolved: larger gains are worse.
   - Result: 0.253 vs. 0.263 in the archive; Spearman 0.78; AlignScore gap −0.20 / −0.17; +11.9 words out of support.
   - Client CAA vectors agree with the average at cosine 0.65–0.86.
3. **B3 one-shot merging is worse than both the federated method and its own local source.**
   - B3 from `local` − `local`: worse on **7/8**, better on 1/8. B3 from `local-off` − `local-off`: worse 7/8, better 1/8.
   - **The stronger merge is from `local`** (0.181 vs. 0.227; better on 7/8 vs. the `local-off` merge). As the coordinator suspected, `local-off`'s drifting offsets (+0.24 to +1.10) make its directions merge badly. **B3 from `local` is the B3 to report.**
   - The one client where both merges beat the method is **reuters.com** (0.170–0.171 vs. 0.199), the same exception as archive entry 22 (+0.017 there). Its local directions are presumably the largest contributors to the merged direction at the high end.
   - B3's summaries are **shorter** than the references (−2.2 / −1.4 words) and less consistent in support (AlignScore gap −0.08): the merged direction is weaker than a client's own.
4. **Ordering:** `fed-aligned-off` (0.146) < `local` (0.157) < B3 (0.181) < B4 (0.253) < B1 (0.34–0.38). It is the same ordering and similar margins as archive entry 22, which used the old method (0.151). The calibration redesign kept every baseline gap.

**Gates:** G1 (B1 does not steer) and G3 (one-shot merge < iterative federated) pass again with the new method.

## NR-74. Two arms in the new parameterization, launched: `fed-linear-off` and `fed-shared-soff-sadapter` (2026-10-09, 12:10)

**Assigned by the coordinator (user request).** Common settings matched to NR-62 / NR-60: `model_name=meta-llama/Llama-3.2-1B-Instruct fed.rounds=100 max_train_per_client=4000 fed.fix_gain=true fed.warp_reg=0`, rotation 0, the full train → dev selection → test → quality → judge pipeline.

| Arm (= job name, `runs/<name>_…`) | Model | Overrides | cc job |
|---|---|---|---|
| `fed-linear-off` | g_i(α) = o_i + α: **no learned shape**, private offset. There is nothing to pool, so it stands for both the shared and aligned versions. | `lora.warp=none lora.warp_scope=module fed.calibration=shared lora.offset=true fed.private_offset=true` | **11241906** |
| `fed-shared-soff-sadapter` | **pure FedAvg**: shared adapter, per-layer h_l averaged, one shared o; g(α) = o + h_l(α) for every client. The new-setup A2. | `fed.adapter=shared fed.calibration=shared lora.offset=true lora.warp=kumaraswamy_mix lora.warp_scope=module` | **11241907** |

- Launch [NR-74.sh](launch/NR-74.sh). cc only, `--gres=gpu:A100:1` (A100s, like NR-62), `--exclude=ccc0387`, 24 h. No code change.
- **`warp_scope=module` with `warp=none` does not error.** `inject_steer_lora` uses scope `model` whenever `warp == "none"`, so the setting is recorded in `config.yaml` but has no effect (there are no shape parameters).

**CPU smoke** (2 rounds, `train_fed.py`, tiny config), both passed:
- `fed-linear-off`: warp none, offset true, `private_offset` true. The server's calibration state is `{u}` only (**no o**); each client holds only `steer_control.o`.
- `fed-shared-soff-sadapter`: adapter shared, so 112 B_p tensors are in the server state; `private_offset` false, so **o is in the server state**; per-layer shapes plus u in the server state (338 calibration tensors).
- To check in each real run's `config.yaml` after round 1.

**Comparisons:**
- `fed-linear-off`:
  - vs. `fed-aligned-off` (NR-62, 0.146) and `fed-shared-off` (NR-60, 0.150), i.e. whether a learned shape is needed once o_i is there;
  - vs. the archive's no-offset linear arm (entry 40, 11145307: 0.153).
- `fed-shared-soff-sadapter`:
  - vs. `fed-aligned-off`, `fed-shared-soff` (NR-60c, 0.153) and A2 (11094904: 0.159, worst 0.176; in support 0.177, out of support 0.136);
  - **in vs. out of support**, since A2's trade-off was worse in support and better out of it;
  - **house style** (`style_eval.py`): A2 was the floor (attribution 0.409).

**Report:** the test table with the usual columns; per-α cells with α = 0 and 1 separately; learned offsets; paired contrasts against `fed-aligned-off`.

**NR-74 round-1 config check (11:24):** both runs started on ccc0284 (A100-40GB, like NR-62).
- `nr74-fed-linear-off_20261009-111900_j11241906`: warp none, offset true, `private_offset` true, adapter private, calibration shared, `fix_gain` true, `warp_reg` 0, 100 rounds, 4000 per client.
- `nr74-fed-shared-soff-sadapter_20261009-111900_j11241907`: warp kumaraswamy_mix, scope module, offset true, `private_offset` false (o shared), **adapter shared**, calibration shared, `fix_gain` true, `warp_reg` 0.
- As intended. The server-state contents were verified in the CPU smoke; the first real snapshot is at round 10.

## NR-75. NR-74 results: a learned shape adds little once o_i is there; pure FedAvg repeats A2's trade-off and house-style floor (2026-10-09, 18:30)

Analysis only.

**State.** Both NR-74 runs are done on ccc0284 (A100-40GB), full pipeline, no errors:
- `fed-linear-off` (11241906): dev-selected **r100**;
- `fed-shared-soff-sadapter` (11241907): dev-selected **r70**, the same round A2 was selected at.

Reports:
- `reports/nr75_test.txt` (`compare_runs.py`, post-CAL-26 version; paired bootstrap);
- `reports/nr75_cells_test.txt` (per-α cells, output percentiles, o);
- `reports/nr75_style_test.txt` (`style_eval.py`).

Archive runs cited: `linear-noOff` = entry 40's no-offset linear run (11145307, gain fixed, no warp, no offset; same setting); A2 = 11094904 at r70. Single seed.

| Arm | Pct err (worst) | In-support | Out-of-support (worst) | Reach | Spearman (worst) | α = 0 cell | α = 1 cell | AlignScore gap in / out | Length gap in / out | Judge faithful in / out |
|---|---|---|---|---|---|---|---|---|---|---|
| **`fed-aligned-off`** (reference, NR-62) | **0.146** (0.199) | 0.135 | 0.151 (0.184) | 0.425 | 0.929 (0.892) | **0.195** | 0.104 | −0.02 / +0.05 | +2.4 / +3.4 | 0.83 / 0.79 |
| `fed-shared-off` (NR-60) | 0.150 (0.205) | 0.134 | 0.158 (0.184) | 0.393 | 0.928 (0.890) | 0.196 | 0.121 | −0.02 / +0.05 | +1.8 / +2.2 | 0.84 / 0.80 |
| **`fed-linear-off`** | 0.148 (**0.195**) | **0.132** | 0.157 (0.180) | 0.408 | 0.933 (0.908) | 0.201 | 0.115 | −0.02 / +0.05 | +2.6 / +2.8 | 0.84 / 0.81 |
| *`linear-noOff` (archive 40)* | *0.153 (0.217)* | *0.134* | *0.165 (0.207)* | *0.404* | *0.927* | | | *−0.02 / +0.06* | *+2.2 / +3.1* | *0.83 / 0.80* |
| `fed-shared-soff` (NR-60c) | 0.153 (0.222) | 0.137 | 0.160 (0.194) | 0.397 | 0.926 (0.859) | | | −0.02 / +0.05 | +2.2 / +2.8 | 0.84 / 0.79 |
| **`fed-shared-soff-sadapter`** (r70) | 0.171 (0.191) | **0.183** | 0.154 (**0.171**) | **0.498** | **0.948** (0.908) | 0.257 | **0.051** | **+0.09 / +0.13** | **+6.3 / +11.8** | **0.89 / 0.87** |
| *A2 (archive, r70)* | *0.159 (0.176)* | *0.177* | *0.136 (0.168)* | *0.500* | *0.951 (0.914)* | *0.205* | *0.061* | *+0.09 / +0.10* | *+7.2 / +10.1* | *0.89 / 0.86* |

### 1. `fed-linear-off`: once o_i is there, the learned shape adds little

**Paired per-client contrasts:**

| Contrast | Better on | Worse on |
|---|---|---|
| `fed-linear-off` − `fed-aligned-off` | 0/8 | 2/8: wsj.com +0.006, aol.com +0.007; out of support also people.com +0.011, wsj.com +0.014, aol.com +0.017 |
| `fed-linear-off` − `fed-shared-off` | 1/8 (reuters.com −0.010) | 0/8 |
| `fed-linear-off` − `linear-noOff` (archive) | **3/8** (reuters.com −0.023, nypost.com −0.012, theguardian.com −0.007) | 0/8 |

- **The mean differences are 0.002 against `fed-aligned-off` and −0.002 against `fed-shared-off`: unresolved.**
  - With a private level and a plain linear coefficient g_i(α) = o_i + α, steering is as good as with learned per-layer shapes.
  - It is better than or equal to FedAvg-learned shapes, with the best worst-client (0.195) and in-support (0.132) of the three.
- The aligned pooled shape still buys a little on the broad clients **out of support** (aol.com, wsj.com, people.com +0.011 to +0.017 for linear) and at the top (α = 1 cell 0.104 vs. 0.115; reach 0.425 vs. 0.408).
- **The offset is what matters for the copy-heavy clients.** Against the no-offset linear archive run, o_i fixes reuters.com, nypost.com and theguardian.com.
- **The offsets are the same as with a learned shape:** nypost.com −0.689, reuters.com −0.405, the others −0.03 to −0.19 (NR-62: −0.618 / −0.381). A fourth design with the same per-client levels.
- House style is unaffected: attribution 0.550, feature gap 0.104 (`fed-aligned-off` 0.562 / 0.110).

### 2. `fed-shared-soff-sadapter` (pure FedAvg) repeats A2's trade-off, a little worse, and the shared o does not settle

- **Against `fed-aligned-off`:**
  - Overall: worse on **6/8** (+0.018 to +0.049), better on 1/8 (nypost.com −0.014); reuters.com tie overall, better out of support (−0.024).
  - **In support:** much worse (0.183 vs. 0.135).
  - **Out of support:** about equal (0.154 vs. 0.151), with a better worst client (0.171 vs. 0.184).
  - Steering range: reach 0.498 vs. 0.425, Spearman 0.948, α = 1 cell 0.051 vs. 0.104.
  - The cost is the bottom: the α = 0 cell is 0.257 vs. 0.195.
- **This is A2's trade-off** (in-support precision for reach and extrapolation). The same mechanism as archive entry 30: a shared adapter cannot absorb a client's level, so everything goes through the shared D and calibration.
- **Against A2: worse on 7/8** (+0.009 to +0.016), better on none. The new setup's per-layer shape (averaged) and shared offset do not help the shared-adapter design; A2 (one shape, no offset, learned shared gain) was better.
- **The shared o drifts positive.**
  - The server's o (the authoritative value; per-client copies read 0.0, as the coordinator noted) is **+0.284 at the selected r70 and +0.606 at r100**.
  - Every client's α = 0 output percentile rises to 0.23–0.29: the α = 0 cell is the worst of any arm. The α = 1 cell is the best of any arm.
  - Compare `fed-shared-soff` (private adapter, shared o), where o settles at −0.026.
  - **Pattern across the runs:** o drifts when the adapter and the direction are in the same sharing class as o, i.e. all private (`local-off`, NR-65) or all shared (here). It settles when o is the client's level next to a shared D and a private adapter, or a shared level next to private adapters. Consistent with a redundancy between o · D and the other terms trained on the same data; single seed, a reading, not established.
- **Quality:** like A2, its summaries are longer (+6.3 / +11.8 words) and more consistent with the article (AlignScore gap +0.09 / +0.13), and the judge rates them more faithful (0.89 / 0.87). Those signs come with more copying; the judge prefers extractive summaries.

### 3. House style: the A2 floor repeats exactly

| Run | Attribution | P(own) | Feature gap |
|---|---|---|---|
| real summaries (ceiling) | 0.587 | 0.388 | — |
| `fed-aligned-off` | 0.562 | 0.381 | 0.110 |
| `fed-linear-off` | 0.550 | 0.375 | 0.104 |
| **`fed-shared-soff-sadapter`** | **0.406** | 0.293 | **0.183** |
| A2 (archive) | 0.409 | 0.295 | 0.190 |

- Per client: nypost.com **0.235** (A2 0.250; `fed-aligned-off` 0.635), theguardian.com 0.297 (A2 0.294), forbes.com 0.345.
- **Without a private adapter the house style is lost**, regardless of calibration design. With one (every other arm) it is kept.

### Conclusions

1. **The private offset carries most of the calibration's value.** g_i(α) = o_i + α already reaches 0.148, within noise of the learned-shape designs. The pooled shape adds a small, not significant, out-of-support and top-end gain on the broad clients.
2. **The private adapter is needed for house style and in-support precision.** Pure FedAvg (shared adapter) trades them for reach and extrapolation, exactly as A2 did. With a shared adapter, a shared offset does not settle.

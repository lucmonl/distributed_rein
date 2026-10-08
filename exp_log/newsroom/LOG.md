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

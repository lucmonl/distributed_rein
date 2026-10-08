# Experiment log: molecule

Updated after every code change and every job; newest entries at the bottom. Entry IDs: `MOL-<n>`, starting at MOL-23.
Conventions (paths, terminology, job names): `../../CONVENTIONS.md`. History before 2026-10-07: `../EXPERIMENT_LOG.md` (archive, read-only).


## MOL-23. MOL-18 NORESET: without the per-round Adam reset, federated is no better than local on ChEMBL (2026-10-07)

Analysis only; no new code, no new jobs (cc submissions are on hold per CAL-2). Job **11180423**
(`exp46_cov_private_noreset_20261007-054203_j11180423`) finished 10-07 11:03 and was never analysed
in the archive; the coordinator computed it in CAL-1 and the report is
`../reports/mol18_noreset_offset_test.txt` (archive path, test split, strict scorer, gain fixed at 1).
Numbers below re-checked here against the run directories.

**Arm labels (CONVENTIONS.md §4) and the archive names they replace.** The archive's MOL-18/MOL-22
entries and the report columns use the old names; this is the mapping used from here on:

| New label | Archive name | Overrides | cc job |
|---|---|---|---|
| `fed-private` | PRIV | `fed.calibration=private` | 11180421 |
| `fed-private-noreset` | NORESET | `fed.calibration=private fed.reset_shared_opt_state=false` | 11180423 |
| `local` | LOCAL | `fed.mode=local fed.calibration=private` | 11180422 |
| `fed-borrow` | COV | `fed.calibration=coverage fed.cov_lambda_max=1` | 11180420 |
| `fed-private-off` | PRIV_OFF | + private offset | MOL-21 pair |
| `local-off` | LOCAL_OFF | + private offset, local | MOL-21 pair |

All on `data/chembl_deco_skew_pruned`, 100 rounds, rotation 0, seed 0, gain fixed at 1, no offset
except the `-off` arms, `kumaraswamy_mix` shape, `warp_reg 0`, full participation, strict scorer.
**Single seed everywhere.**

### Result

| arm | penalized | pct err (worst) | in-support | out-of-support | reach | end near-tie | round |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fed-private` | 0.207 | 0.199 (0.279) | 0.164 | 0.226 | 0.327 | 0.037 | 80 |
| `fed-private-noreset` | 0.209 | 0.206 (0.294) | 0.169 | 0.239 | 0.274 | 0.071 | 80 |
| `local` | 0.207 | 0.205 (0.303) | 0.167 | 0.238 | 0.279 | 0.058 | 100 |
| `fed-borrow` | 0.207 | 0.197 (0.270) | 0.174 | 0.215 | 0.367 | 0.027 | 100 |
| `fed-private-off` | **0.193** | **0.188** (0.279) | 0.161 | 0.211 | 0.336 | 0.020 | 100 |
| `local-off` | 0.208 | 0.204 (0.303) | 0.169 | 0.238 | 0.277 | 0.039 | 80 |

Per-client paired bootstrap, 95% CI (report, §"PRIV − NORESET" and §"NORESET − LOCAL"):

- `fed-private` − `fed-private-noreset`: **better on 5/8 clients, worse on 1/8** (CHEMBL228 +0.017).
- `fed-private-noreset` − `local`: **better on 1/8 (CHEMBL228), worse on 2/8** (CHEMBL204 +0.014,
  CHEMBL4078 +0.019).

**So the per-round reset of the direction's Adam state, not the sharing, is what made federated look
better than local on ChEMBL.** Drop the reset and the fed arm falls back onto the local arm — and the
means have it marginally behind (0.209 vs 0.207 penalized, 0.206 vs 0.205). The MOL-22 offset win
(`fed-private-off` − `local-off`, better 5/8, worse 0/8) is measured *with* the reset in place, so
that result inherits the same confound: it is "fed + reset + offset beats local + offset", not
"fed beats local".

**The mechanism is reach, not in-support accuracy.** Removing the reset costs almost nothing
in support (0.169 vs 0.164) and most of the damage is outside it (0.239 vs 0.226), with reach
dropping 0.327 → 0.274 — i.e. to `local`'s 0.279 — and the endpoint near-tie rate nearly doubling
(0.037 → 0.071). Resetting the optimizer every round keeps the direction moving far enough to reach
levels outside a client's own support; keeping the state lets it settle, and the alpha levels
collapse back onto the boundary. This is the same α=0 immovability MOL-20/MOL-22 found from the other
direction.

**The dev metric agrees, and more bluntly.** Dev `pct_calib_err_penalized` (mean over clients) at the
selected round: `local` 0.1999, `fed-private` 0.2089, `fed-private-noreset` 0.2092. On the selection
metric itself `local` is the best of the three. Round selection is per-arm argmin on dev, which is
why the round column differs (80 / 100 / 80) — each arm is at its own best round, not truncated:
`fed-private` 20:0.2315 40:0.2258 60:0.2101 **80:0.2089** 100:0.2120; `fed-private-noreset`
20:0.2436 40:0.2168 60:0.2182 **80:0.2092** 100:0.2204; `local` 20:0.2210 40:0.2120 60:0.2108
80:0.2082 **100:0.1999**.

**What this makes necessary (not yet scheduled; cc is on hold).** A `local` arm with a periodic Adam
reset on its own private direction is now the missing control for every fed-vs-local claim on this
task, including MOL-22's. `fed.reset_shared_opt_state` only acts in `fedavg` mode (`fedsteer/fed.py`),
so this needs an opt-in shared-code flag — and shared code is granted to newsroom-iter1 right now.
Flagged to the coordinator rather than acted on.

### MOL-23a. Housekeeping: molecule log paths (CONVENTIONS.md §2)

SLURM stdout/stderr and gpumem logs for the molecule workstream now go to
`sbatch/logs/molecule/`, named by job name (`%x.o%j`) rather than by the hardcoded per-script name,
so the arm is readable from the filename. Changed, on cc and mirrored to Anvil:

| file | was | now |
|---|---|---|
| `sbatch/train_eval_chembl.sbatch` | `logs/fedsteer_chembl.o%j` | `logs/molecule/%x.o%j` |
| `sbatch/train_eval_chembl_anvil.sbatch` | `logs/fedsteer_chembl.o%j` (Anvil root) | `logs/molecule/%x.o%j` (Anvil root) |
| `sbatch/run_cmd_chembl.sbatch` | `logs/fedsteer_cmd.o%j` | `logs/molecule/%x.o%j` |
| `sbatch/eval_baselines_chembl.sbatch` | `logs/fedsteer_baselines_chembl.o%j` | `logs/molecule/%x.o%j` |
| `sbatch/train_eval_chembl.sbatch` (gpumem) | `sbatch/logs/gpumem.o$JOB` | `sbatch/logs/molecule/gpumem.o$JOB` |

`sbatch/logs/molecule/` created on Anvil (`/home/x-zchen17/lucmon/rein/sbatch/logs/molecule`); it
already existed on cc. Only these four files were synced to Anvil — not the tree — because
newsroom-iter1 holds the shared-code grant and a full rsync would have pushed half-edited
`fedsteer/` there. Race-watcher logs go to `sbatch/logs/molecule/race-<ID>.log` (a `race_watch.py`
argument, no code change). Old flat logs stay where they are; the archive refers to them by those
paths.

**Running jobs kept, untouched:** exp45 cc duplicates 11169098 (`exp45_pruned_fed_cc`) and 11169100
(`exp45_pruned_local_cc`). Editing an sbatch file does not affect them — SLURM copies the script at
submit time — and their logs stay at the old flat paths.

## MOL-24. Shared shape at gain 1 with per-layer shapes, ± private offset: pre-flight done, LAUNCH HELD (2026-10-07)

**Status: nothing submitted.** Pre-flight is complete and the launch scripts are final, but the user
instructed (2026-10-07, late) that no new experiments launch until they say so. Two pre-flight smoke
jobs submitted before that instruction are still queued (11206293 `mol24-smoke-fed-shared`, 11206294
`mol24-smoke-fed-shared-off`); they are verification, not experiments, and were left in.
Launch [MOL-24.sh](launch/MOL-24.sh), pre-flight [MOL-24_smoke.sh](launch/MOL-24_smoke.sh).

### Design decision by the user: the Adam reset is part of federated training, not a confound

This overrides CAL-3/CAL-4, which had made a local-with-reset arm the headline control on the
strength of MOL-23.

> FedAvg overwrites the client's shared parameters with the server average every round, so carrying
> stale Adam moments across that overwrite is the anomaly — resetting them is what FedAvg *is*.
> Therefore `fed` (with its per-round reset) against `local` (as local training is naturally
> implemented, state carried) is already the fair comparison.

**Consequence for MOL-23.** Its result is unchanged and is not withdrawn: without the reset, the fed
arm falls back onto local, and the reset's effect is reach (out-of-support 0.226 vs 0.239, reach
0.327 vs 0.274). Under this decision that is a statement about **why** FedAvg helps on this task,
not a confound to control away. The `local-off-reset` control is dropped, and
`fed.local_reset_opt_state` (NR-60) is not used by the molecule workstream.

**Consequence for MOL-24.** Dropping the reset arm would have left two fed arms and no local
baseline, so the third slot becomes plain **`local-off`** — local, private per-layer shape, private
offset, state carried. ChEMBL has never had a per-layer local arm with an offset (MOL-22's
`local-off` used one shape per client), so this is both the comparator the race needs and a gap in
CAL-1's missing list.

### Arms (final)

| arm | calibration | direction | overrides beyond the reference | what it isolates |
|---|---|---|---|---|
| `fed-shared` | g_{i,l}(α) = h_l(α) | D | — (reference calibration) | the reference: s = 1, per-layer shape, shared shape, no offset |
| `fed-shared-off` | g_{i,l}(α) = o_i + h_l(α) | D | `lora.offset=true fed.private_offset=true` | o_i, the private offset, under a shape shared across clients |
| `local-off` | g_{i,l}(α) = o_i + h_{i,l}(α) | D_i | `fed.mode=local fed.calibration=private lora.offset=true` | the local baseline at per-layer shapes with an offset |

Notation per CONVENTIONS.md §3. s = 1 in all three arms, so the gain is dropped from the
expressions. `h_l` carries no client index in `fed-shared`/`fed-shared-off` because the shape there
is literally shared by all clients (FedAvg of its parameters, one per layer l); `h_{i,l}` in
`local-off` is private to client i. The offset o_i is one scalar per client, never per layer, in
every arm that has one.

Reference overrides passed literally on every arm (CONVENTIONS.md §3 — "default" there means the
reference *design*, not the code or config defaults):
`lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0 lora.offset=false fed.calibration=shared`.
The code defaults to `warp_scope=model` and `configs/chembl_deco_skew_pruned.yaml` ships
`fed.warp_reg=0.01`, so omitting either would silently run something else.

Matched to MOL-18/MOL-22 otherwise: same config file, `data/chembl_deco_skew_pruned`, 100 rounds,
rotation 0, seed 0, strict scorer at every stage (`clogp_residual_deco_strict`, rescored with
`clogp_residual_deco`), dev selection on `pct_calib_err_penalized`, 150 test prompts, fedavg keeps
its per-round reset. **Single seed.** MOL-22's runs used one shape per client, so they are context,
not matched references.

**One race group** (MOL-18's rule): cc has rdkit 2025.09.6 and Anvil 2026.03.6, so splitting the
arms across hosts would confound the arms with the scorer version. Whichever host starts any member
first gets all three. Anvil cost if Anvil wins: ~6 SU per arm, ~18 of the 25 SU allocated in CAL-3.
Anvil balance 473.1 / 500 at the time of writing.

### Pre-flight (all complete)

1. **Shared tests on cc: 74/74 pass** in the `steer` env. (A first attempt failed on
   `ModuleNotFoundError: rdkit`: `source ~/.bashrc` alone does not activate the env, it needs
   `source activate steer`. Not a code problem.)
2. **Override resolution off-GPU:** all three arms loaded through `train_fed.load_config` and
   constructed `SteerLoraConfig` / `FedConfig` / `MonitorConfig` on both the real and smoke config.
   Catches unknown-key typos instantly — MOL-22b's cause 1.
3. **End-to-end smoke through the real production sbatch**, 2 clients × 3 rounds × 64 examples
   (`configs/chembl_deco_skew_pruned_smoke2.yaml`): full pipeline, train → `summarize_sweep` →
   `eval_direction` → `score_molecules` → `rescore_eval`. Clients CHEMBL243 (no high side) and
   CHEMBL228 (no low side) chosen for *opposite* support gaps. Warp and gain warmup shortened in
   that config only, so the shape and offset actually train inside 3 rounds. **The 2 clients are
   listed in the config, not passed through `--export`:** a 2-client list contains a comma and
   `--export` is itself comma-separated, which is MOL-22b's cause 2.

   **Arms 1 and 2 PASSED** (11206293 12m22s, 11206294 7m15s, both exit 0, ccc0465, 17:09–17:29).
   Both reached `DONE` with the full artifact set — test eval, `mols_`, `rescored_`,
   `code_snapshot.txt`. Effective `config.yaml` in both: `warp_scope=module`, `warp_reg=0`,
   `fix_gain=true` (reported gain 1.0 per client), `calibration=shared`, `mode=fedavg`,
   `private_offset` true in `fed-shared-off` and false in `fed-shared`. The snapshots carry a
   **252-entry shape bank** = 36 layers × 7 target modules, which is `warp_scope=module` actually
   materialising per-layer shapes rather than being accepted and ignored. In `fed-shared-off` the
   server state holds **no** `steer_control.o` while the two clients report **different** offsets
   (0.017468 and 0.017488), so `fed.private_offset` keeps o_i out of the FedAvg aggregate as
   designed — the one behaviour MOL-24 depends on that no previous ChEMBL run exercised. (The two
   arms' 3-round errors are identical to three decimals, 0.401 / 0.363 per client; expected, since
   an offset of 0.017 cannot move anything after 3 rounds. The smoke tests that it runs, not that
   it helps.)

   Arm 3's smoke was cancelled with the reset arm and has not been resubmitted under the launch
   hold. Both passing runs started at 17:09, hours before newsroom-iter1 began editing
   `fedsteer/fed.py` at 22:02, so they snapshotted a clean tree and are not void.
4. **New flags verified:** `fed.private_offset` (fed.py:109) and `fed.local_reset_opt_state`
   (fed.py:122), both default `False`; `private_offset` engages only when `calibration == "shared"`
   and raises if `lora.offset` is absent.
5. **Anvil code identical to cc:** md5 matches on 27 shared files, including the config.

### Analysis plan (fixed before the results exist)

1. **`fed-shared-off` vs `fed-shared`** — does a private offset help under a shared shape? Mirrors
   Newsroom's primary NR-60 question, so the two datasets' answers are comparable.
2. **`fed-shared(-off)` vs `local-off`** — fed vs local at matched per-layer shapes. `fed-shared`
   has no offset while `local-off` does, so only `fed-shared-off` is the offset-matched contrast.
3. **`fed-shared` vs MOL-18 `fed-private`** — context only; the shape scope differs, so this is not
   a controlled comparison of sharing.

Per client: in/out of support; the per-α cell decomposition with **α=0 and α=1 separately**, from the
saved score grids and each run's own `alpha_reference.json` per prompt×alpha cell (not curve means,
which measure bias and overstate the endpoints' share — MOL-20); the **achieved output percentile at
α=0**, which no arm has moved yet; the **learned offsets**; the unscorable rate. Primary metric
`pct_calib_err_penalized`, with strict `pct_calib_err` alongside, and a per-client paired bootstrap
with 95% CIs for every pairwise claim, as in MOL-23. Clients to call out: **CHEMBL243** (no high
side, worst in every MOL-18 arm), **CHEMBL228** (no low side), **CHEMBL4078** (middle-only).

**What would falsify the offset story:** MOL-22 found the private offset was the first change to
survive the failure penalty, but under a *private* shape. If `fed-shared-off` − `fed-shared` is flat
here while MOL-22's `fed-private-off` − `fed-private` was not, the offset's benefit depends on the
shape being private and the shared-shape design cannot inherit it.

### MOL-24a. Exp45 cc duplicates cancelled (2026-10-07)

**Cancelled by the user's instruction:** 11169098 (`exp45_pruned_fed_cc`) and 11169100
(`exp45_pruned_local_cc`), the MOL-17 pair, plus 11206295 (`mol24-smoke-local-off-reset`) with the
reset arm. In CONVENTIONS.md §4 terms these two were **`fed-private-gain-1shape`** and
**`local-gain-1shape`**: private adapter P_i, `kumaraswamy_mix` shape, private calibration,
**learned** gain (s_i = 0.606–0.956 fed / 1.191–1.544 local), **one shape per client**
(`warp_scope=model`), no offset, identity penalty `fed.warp_reg=0.01` (a deviation from §3's
override list with no §4 suffix). In §3 notation both are g_i(α) = s_i · h_i(α) — no layer index,
because the shape is one per client rather than per adapted matrix — and only the direction
differed: FedAvg-shared `D` vs per-client `D_i`.

**They had already finished, twice each.** Every earlier attempt is logged `PREEMPTED` and restarts
from round 0 with a new stamp, but four attempts reached round 100 and wrote a full pipeline
(test eval + molecule scoring + rescore + `code_snapshot.txt`), the last file landing in the *same
minute* SLURM recorded the preemption:

| arm | attempts | completed pipelines | state when cancelled |
|---|---:|---|---|
| `fed-private-gain-1shape` | 10 | 10-06 17:28, 10-07 11:26 | round 100/100, in test eval |
| `local-gain-1shape` | 8 | 10-06 09:33, 10-07 14:06 | round 56/100 |

So cc holds two complete copies of each arm and the running attempts were building a third, while
holding an A100 on ccc0284 that the MOL-24 smoke was queued behind. Early attempts also landed on
the Turing nodes `ccc0232`–`ccc0235` that MOL-18 later excluded from
`sbatch/train_eval_chembl.sbatch`; the requeues did not inherit that exclusion.

**Result of the completed pair** (test, strict scorer, round 100, dev-selected on
`pct_calib_err_penalized`; runs `..._20261007-061447_j11169098` and `..._20261007-094142_j11169100`):

| arm | penalized | pct err (worst) | in-support | out-of-support | reach |
|---|---:|---:|---:|---:|---:|
| `fed-private-gain-1shape` | 0.215 | 0.209 (0.313) | 0.165 | 0.246 | 0.263 |
| `local-gain-1shape` | **0.205** | **0.199** (0.287) | 0.164 | **0.229** | **0.292** |

Per client the shared direction is worse on 6 of 8 and better on 2 (CHEMBL2835 0.190 vs 0.202,
CHEMBL240 0.165 vs 0.168). The whole gap is outside support — in-support ties at 0.165/0.164 — and
reach is lower. This reproduces the Anvil copies' negative result and is the comparison the user's
design decision treats as already fair, so the question these jobs were launched to answer is
answered: **with a learned gain and one shape per client, the shared direction loses to a private
one on pruned ChEMBL.** Single seed, and `compare_runs.py` gives no paired bootstrap, so 6-of-8 is a
direction and not a significance claim.

### MOL-24b. Anvil gated off; expected arm change pending (2026-10-08)

**Anvil is unusable** (CAL-11): `/anvil/scratch/x-zchen17/lucmon/envs/rein` has lost stdlib files,
apparently to a scratch purge, and every Anvil job since 03:09 dies in seconds with `cannot import
name '_parser' from partially initialized module 're'`. The Anvil half of
[MOL-24.sh](launch/MOL-24.sh) is now **gated behind `ANVIL=1`** and off by default, so MOL-24 is a
cc-only race at **0 SU** until the environment is rebuilt and a short job completes there. The
rebuild is the user's call; the environment has not been touched from here. Anvil's *code* is still
md5-identical to cc on 27 shared files — only the conda env is damaged.

Incidentally this removes the one-race-group *reason*: the group existed because cc has rdkit
2025.09.6 and Anvil 2026.03.6, so a split would confound the arms with the scorer version. Cc-only,
all arms share one scorer regardless. The group is kept in the spec for when Anvil returns.

**`scripts/race_watch.py` is safe against the failing copies, contrary to a warning in CAL-11**
(since corrected there). The `STARTED` set at line 31, which contains `FAILED`, is **dead code,
referenced nowhere**. The live decision at lines 103–111 requires `RUNNING`/`COMPLETING` past
`--min_running_s`, or `COMPLETED`, with the comment "a copy that FAILED early must not win". Traced
for this case — Anvil `FAILED`, cc `PENDING` — no copy wins, the race holds only one `LIVE` copy, so
`live_races` is 0 and the watcher exits without cancelling; cc proceeds. The residual risk is the
opposite and milder: the watcher stops watching, so Anvil copies starting later would not be
cancelled.

**Shared code re-verified after newsroom-iter1's `fed.shared_offset` edit:** 76/76 tests pass on cc,
and the three staged arms still resolve with `shared_offset` defaulting to `False`, so nothing
MOL-24 depends on moved. (Two earlier failures were cc-login1 exhausting its process budget with
threads unpinned — `OpenBLAS blas_thread_init: pthread_create failed`, `libgomp: Thread creation
failed` — not code. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` is the workaround on the login node.)

**Expected arm change (CAL-11), not yet applied.** Newsroom's best design is `fed-aligned-off`,
g_{i,l}(α) = o_i + h̄_l(α), which beat `fed-shared` and `local` there and was worse on no client. The
coordinator expects MOL-24 to become `fed-shared`, `fed-aligned-off`, `local`, and possibly
`local-off` (on ChEMBL `local-off` ≈ `local` in MOL-22, whereas on Newsroom `local-off` misbehaved
with o_i drifting to +0.78). All four resolve off-GPU on this config, so the change is runnable:

| arm | calibration | overrides beyond the reference |
|---|---|---|
| `fed-shared` | g_{i,l}(α) = h_l(α) | — |
| `fed-aligned-off` | g_{i,l}(α) = o_i + h̄_l(α) | `fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01 lora.offset=true` |
| `local` | g_{i,l}(α) = h_{i,l}(α) | `fed.mode=local fed.calibration=private` |
| `local-off` | g_{i,l}(α) = o_i + h_{i,l}(α) | `fed.mode=local fed.calibration=private lora.offset=true` |

⚠️ **`fed-aligned-off` must not pass `fed.private_offset=true`.** Under `calibration=aligned`,
`lora.offset=true` already gives a private o_i (`fedsteer/fed.py:246`), and `private_offset` is
gated on `calibration == "shared"` (`fed.py:177`), so adding it is a silent no-op — there is a test
named `test_private_offset_is_a_noop_where_the_offset_is_already_private`. CAL-1's constraint that
`coverage`/`consensus`/`aligned` reject `lora.offset=true` is **superseded for aligned** by NR-62.

**Status: still nothing submitted**, on the user's launch hold. The final spec arrives with the
release; the arms table in MOL-24 above is superseded by whatever it says.

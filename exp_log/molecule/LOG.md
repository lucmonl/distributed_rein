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

## MOL-24. Shared shape at gain 1 with per-layer shapes, ± private offset: SUPERSEDED BY MOL-25, NEVER LAUNCHED (2026-10-07)

**Status: never launched; superseded by MOL-25 (CAL-19, 2026-10-09).** The user made
`fed-aligned-off` the main algorithm across datasets, so this arm set — `fed-shared` /
`fed-shared-off` / `local-off` — was replaced before any job was submitted. Nothing below was run
except the two pre-flight smokes. Kept because its pre-flight carries over to MOL-25 intact:
`fed-shared-off`'s override set is unchanged and already smoke-tested (11206294), and the
`warp_scope` / `warp_reg` / `private_offset` findings apply to MOL-25 as written.

Original status note: pre-flight was complete and the launch scripts final, but the user
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

## MOL-25. E1 on pruned ChEMBL with the aligned calibration as the method (2026-10-09)

**PROSPECTIVE.** Spec from the coordinator (CAL-19), on the user's decision to make `fed-aligned-off`
the main algorithm across datasets. Supersedes MOL-24, which was never launched. Launch
[MOL-25.sh](launch/MOL-25.sh), pre-flight [MOL-25_smoke.sh](launch/MOL-25_smoke.sh).

**Question.** All three arms carry a private offset o_i, so the contrasts isolate *how the shape is
obtained* rather than whether an offset helps — which MOL-22 already answered affirmatively under a
private shape. Newsroom found `fed-aligned-off` beat both `fed-shared` and `local` (CAL-11), worse on
no client; ChEMBL has never run an aligned calibration at all.

### Arms

| arm | calibration | direction | overrides beyond COMMON |
|---|---|---|---|
| `fed-aligned-off` | g_{i,l}(α) = o_i + h̄_l(α) | D | `fed.calibration=aligned fed.cov_grid=21 fed.cov_pool=saturating fed.cov_lambda_max=0.01 lora.offset=true` |
| `fed-shared-off` | g_{i,l}(α) = o_i + h_l(α) | D | `fed.calibration=shared lora.offset=true fed.private_offset=true` |
| `local-off` | g_{i,l}(α) = o_i + h_{i,l}(α) | D_i | `fed.mode=local fed.calibration=private lora.offset=true` |

`h̄_l` is the server-pooled table used in **both** training and inference (aligned); `h_l` is FedAvg
of the shape's *parameters* (shared); `h_{i,l}` is private. s = 1 everywhere, so the gain is dropped
from the expressions. COMMON on every arm:
`lora.warp=kumaraswamy_mix lora.warp_scope=module fed.fix_gain=true fed.warp_reg=0`.

⚠️ **`fed.private_offset` is deliberately absent from the aligned arm.** Under
`calibration=aligned`, `lora.offset=true` already gives a private o_i (`fedsteer/fed.py:246`), and
`private_offset` is gated on `calibration == "shared"` (`fed.py:177`), so passing it would be a
silent no-op (`test_private_offset_is_a_noop_where_the_offset_is_already_private`).

τ_pool, b and τ_local stay at the code defaults (100, 0.2, 100), as in Newsroom's NR-62; NR-69 found
τ did not matter once o_i was present.

Matched to MOL-18/MOL-22 otherwise: `configs/chembl_deco_skew_pruned.yaml`,
`data/chembl_deco_skew_pruned`, 100 rounds, rotation 0, seed 0, strict scorer at every stage
(`clogp_residual_deco_strict`, rescored with `clogp_residual_deco`), dev selection on
`pct_calib_err_penalized`, 150 test prompts. **Single seed.** MOL-22's `PRIV_OFF`/`LOCAL_OFF` used
one shape per client, so they are **context, not matched references**.

**cc only, 0 SU, no race group** (CAL-11/CAL-16): the Anvil conda env lost stdlib files to a scratch
purge and its code is stale on three files, so the Anvil half of the launch script is gated behind
`ANVIL=1`. With a single host there is nothing to race and no scorer-version split to protect
against — that was the only reason MOL-18 and MOL-24 needed one group. The Turing nodes and 16 GB
V100s stay excluded in the sbatch (NR-50/MOL-18); ccc0284 is never excluded.

### Analysis plan (fixed before the results exist)

Primary contrasts:

1. **`fed-aligned-off` vs `local-off`** — C1, matched: with o_i held equal on both sides, is a shared
   calibration better than none at all? This is the headline fed-vs-local claim for ChEMBL.
2. **`fed-aligned-off` vs `fed-shared-off`** — does the pooled table h̄_l beat FedAvg-of-parameters
   h_l, again with o_i on both sides? This is the question Newsroom answered in aligned's favour.

Per client: in/out of support; the per-α cell decomposition with **α=0 and α=1 separately**, from the
saved score grids and each run's own `alpha_reference.json` per prompt×alpha cell, not curve means
(MOL-20); the **achieved output percentile at α=0**, which no ChEMBL arm has yet moved; the **learned
o_i per arm**; the unscorable rate. Primary metric `pct_calib_err_penalized`, strict `pct_calib_err`
alongside, per-client paired bootstrap with 95% CIs for every pairwise claim (as MOL-23).

Specific diagnostics CAL-19 asks for:
- **`local-off`'s o_i trajectory per round.** On Newsroom it drifted upward to +0.78 (NR-65), which is
  why `local` rather than `local-off` is their local reference; in MOL-22 on ChEMBL it stayed between
  −0.09 and −0.25. If it drifts here too, the C1 contrast is measuring a diverging baseline and the
  comparison needs `local` as well.
- **For aligned: per-client `support_gap` and `disagreement` over rounds** (both logged by
  `fed.py:586` / `fed.py:617`), plus `tie_loss` and `proj_adjust_max`.

Clients to call out: **CHEMBL243** (no high side, worst in every MOL-18 arm at 0.28–0.30),
**CHEMBL228** (no low side), **CHEMBL4078** (middle-only support, which coverage borrowing hurt).

**What would falsify the aligned story on ChEMBL — corrected (CAL-19 follow-up).** My first version
of this conflated *support skew* with *shape disagreement*; they are not the same thing. CHEMBL243
having ~0 weight at high α, so that h̄_l there is formed from the other clients, is the pooling
working as designed, not a failure. The real question is whether clients disagree **where they both
have data**.

⚠️ **And the logged `disagreement` scalar cannot answer that as it stands, because skew deflates it
rather than inflating it.** `fed.py:609` computes a *weight-weighted* sd per (layer, grid point),

    dis = sqrt( Σ_c W_c (V_c − m)² / Σ_c W_c ).mean(),    m = Σ_c W_c V_c / Σ_c W_c

then means over **all** layers and grid points. Where only one client has non-trivial weight, m
collapses onto that client's own curve, so (V_c − m) → 0 and the others contribute ~0 via W_c → the
sd is structurally ~0 at that point. Those points cannot exhibit disagreement, yet they are averaged
in. With 8 skewed clients on a 21-point grid a large share of points qualify, so the scalar is
diluted toward 0 and **a low value must not be read as "the clients agree"**.

So MOL-25 reports disagreement **restricted to grid points where ≥ 2 clients carry non-trivial
weight** (W_c ≥ 5% of that point's total weight — relative, not absolute, since ChEMBL's clients
range from 643 to 3535 training examples and an absolute floor would track client size), alongside
the unrestricted scalar and the count of qualifying points per layer, so the dilution is visible
rather than silently corrected. Computable post hoc from the snapshots' `coverage` entry
(`self.cov["values"]` and `["pool_weights"]` are per layer × grid point), so **no shared-code change
is needed** — relevant because no grant is open.

**The falsification condition, restated:** if restricted disagreement is high and `support_gap` is
large for the one-sided clients, h̄_l is averaging incompatible curves and `fed-aligned-off` should
lose to `local-off` precisely on those clients. That would mean aligned's Newsroom gain is
conditional on client agreement rather than general — a result, not a failed run.

**Measured before the run: the restriction does not bind on ChEMBL either, and my prediction that it
would was wrong.** MOL-18's borrow arm (11180420) carries real ChEMBL `counts` for the same 8 clients
and dataset, and `pool_weights = counts / (tau_pool + counts)` (`fedsteer/coverage.py:107`), so the
qualifying-point count is computable from it at MOL-25's `tau_pool=100`. At floor 5% / min_clients 2,
**6–8 of the 8 clients clear the floor at every α and all 11 grid points qualify** (dilution ~1.00x),
exactly as on Newsroom (21/21).

**Why the prediction failed** — the same conflation the coordinator corrected once already: the
support intervals quoted throughout this log (CHEMBL243 `[0.01,0.41]`, CHEMBL228 `[0.46,0.97]`) are
the **output percentile range a client can reach**, whereas `pool_weights` measure **training-data
density over the α grid**. CHEMBL243 does have data across the whole grid, lopsidedly: 721 examples
near α=0 down to 5 near α=1 — and 5 still yields weight 5/105 ≈ 0.048, just under the floor, while a
point needs only 2 of 8 clients above it. The skew is real and visible in the counts; it does not
drive any grid point below a 2-client threshold.

So **the restriction is a lenient safeguard that does not bind on either dataset**, and the raw
`disagreement` values are comparable across datasets after all. Sensitivity (qualifying points):

| floor | min_clients | ChEMBL (11 pts) | Newsroom (21 pts) |
|---|---|---|---|
| 5% | 2 | 11/11 | 21/21 |
| 5% | 6 | 11/11 | 20/21 |
| 10% | 6 | 6/11 | 5/21 |
| 12.5% | 6 | 2/11 | 0/21 |

It only bites past a 10% floor with 6+ clients, and there it bites **Newsroom harder** — the opposite
of the predicted direction. Not worth chasing: 6-of-8 above 12.5% demands near-uniform density that
neither dataset has by construction, and Newsroom's 0/21 leaves the metric undefined. Keeping 5% / 2
as the default and reporting both numbers; the equality is the finding.

⚠️ Caveat: that is MOL-18's **11**-point grid against MOL-25's 21. Bandwidth 0.2 means neighbouring
points overlap heavily so per-point counts should hold up, but this is confirmed on the real MOL-25
aligned run, not asserted. Tool: [scripts/disagreement_report.py](../../scripts/disagreement_report.py),
which reproduces `fed.py`'s logged scalar before restricting (MATCH/MISMATCH) so a formula drift
cannot pass as a result.

### Pre-flight

- **Shared tests on cc: 79/79 pass** (re-run after NR-68; threads pinned, see MOL-24b).
- **All three arms resolved off-GPU** against `SteerLoraConfig`/`FedConfig`/`MonitorConfig`:
  `warp_scope=module`, `warp_reg=0`, `fix_gain=True` on all three; grid 21 + saturating pooling +
  λ_max 0.01 only on aligned; `private_offset=True` only on `fed-shared-off`; `shared_offset=False`
  throughout.
- **Aligned diagnostics confirmed present** before relying on them: `tie_loss` (fed.py:491),
  `proj_adjust_max` (fed.py:494), `support_gap` (fed.py:586), `disagreement` (fed.py:617).
- **Smokes submitted:** 11236796 `mol25-smoke-fed-aligned-off`, 11236797 `mol25-smoke-local-off`
  (2 clients × **6** rounds — six rather than three so the aligned table exists at round 5 for the
  252 × 21 check). `fed-shared-off` is not re-smoked: its override set is unchanged from MOL-24 arm 2
  (11206294), which passed end to end with the 252-entry shape bank and private o_i verified.

### MOL-25a. Launched; smokes, the shared-code crash they found, and the round-1 checks (2026-10-09)

**Submitted on cc, all three RUNNING.** No Anvil copies, no race group, no watcher (single host), 0 SU.

| arm | cc job | calibration | direction |
|---|---|---|---|
| `fed-aligned-off` | **11237283** | g_{i,l}(α) = o_i + h̄_l(α) | D |
| `fed-shared-off` | **11237284** | g_{i,l}(α) = o_i + h_l(α) | D |
| `local-off` | **11237285** | g_{i,l}(α) = o_i + h_{i,l}(α) | D_i |

**Round-1 effective `config.yaml`, read from each run directory (CAL-19 asked for this):**

| arm | scope | warp_reg | fix_gain | calibration | offset | private_offset | shared_offset | mode |
|---|---|---|---|---|---|---|---|---|
| `fed-aligned-off` | module | 0 | True | aligned | True | **False** | False | fedavg |
| `fed-shared-off` | module | 0 | True | shared | True | **True** | False | fedavg |
| `local-off` | module | 0 | True | private | True | **False** | False | local |

All 100 rounds, seed 0, rotation 0, grid 21 / saturating / λ_max 0.01 / τ_pool 100 on the aligned arm,
`data/chembl_deco_skew_pruned`, 4000 train per client. `private_offset` is True only where the offset
would otherwise be FedAvg'd; aligned privatises o_i itself and local is private throughout.

**Shared-code snapshot verified per arm.** Jobs copy the tree at start, so each arm's
`runs/_code/*/fedsteer/monitor.py` was checked against the fixed file: all three md5
`bba93ee28534f51796a26d2ff01261e2` with the `stat()` guard present. Any arm snapshotting the pre-fix
file would have been cancelled and resubmitted.

### The crash the smokes found (fixed by the coordinator as CAL-20)

`fedsteer/monitor.py` `format_monitor` indexed `s['pct_calib_err']['mean']` unguarded.
`metrics.summarize` only emits a key when at least one client has a non-None value, so when **every**
client's outputs are unscorable at a full evaluation the key is absent and the `KeyError` propagates
out of `fit()` (`fed.py:637`) — **a logging line killing a training run.** The adjacent lookup for
`pct_err_out_support` was already guarded with `.get`; this one was not.

It was a coin flip, not a property of any arm:

| smoke | round-2 dev unscorable rate | outcome |
|---|---|---|
| `local-off` 11236797 | CHEMBL243 1.0, CHEMBL228 **1.0** | FAILED at 3m26s |
| `fed-aligned-off` 11236796 | CHEMBL243 1.0, CHEMBL228 **0.75** | ran to DONE |

One scorable molecule was the whole difference. Fix guards every lookup in `format_monitor` via a
`stat()` helper, prints nan for a missing metric, is byte-identical for present ones, and adds
`test_format_monitor_survives_missing_metrics`. Suite **80/80** on cc, confirmed independently here.
Exposure to the real runs was low — `monitor.full_every=20`, and MOL-18 measured 0–4% unscorable by
then — but the cost would have been a dead 100-round run plus its queue wait, on any of the three
datasets.

### Smoke results

| arm | smoke job | outcome |
|---|---|---|
| `fed-aligned-off` | 11236796 | COMPLETED 10m13s. Table `z` **(252, 21)** = 36 layers × 7 modules × 21 grid points; `n_warps` 252. `o_i` **absent from the server state**, present and differing per client (`clients[c]["gain"]["steer_control.o"]`: CHEMBL228 0.036128, CHEMBL243 0.046560). `tie_loss` and `proj_adjust_max` logged per client. Offsets train 0.0074 → 0.029 → 0.064 while unscorable falls 1.0 → 0.25 → 0.0. |
| `fed-shared-off` | 11206294 (MOL-24) | COMPLETED. Override set unchanged, so not re-smoked; 252-entry shape bank and private o_i already verified. |
| `local-off` | 11236797 → **11237265** | First attempt hit the crash above. Re-run with `monitor.full_every=6` COMPLETED 5m32s: `warp_scope=module`, o_i private per client (CHEMBL243 0.016881, CHEMBL228 0.027712), no coverage table (correct for local+private), no `steer_control.o` in the server state, full artifact set. |

⚠️ **Stated gap:** `local-off`'s end-to-end validation ran against the **pre-fix** `monitor.py`
(confirmed by md5 and the absent `stat()` helper; the snapshot parses cleanly, so it was not a torn
copy of the concurrent edit). The race arms run the fixed version. The only delta is guarded lookups
inside a logging function, so the validation carries over — but the exact bytes the arms run were not
exercised by a smoke.

**Disagreement at 21 points, on the aligned smoke** — `unrestricted 0.00088, logged 0.00088 (log
round 5) MATCH`; `restricted 0.00208`, **7 of 21 grid points** keeping ≥ 2 clients at ≥ 5% weight,
dilution **2.37×**. This is a property of the **2-client smoke**, not a ChEMBL finding: CHEMBL243
(no high side) and CHEMBL228 (no low side) overlap only at α 0.35–0.65, so 14 of 21 points have a
single client and are structurally zero-disagreement. It does show the safeguard binding on real data
when overlap is thin, so the mechanism is confirmed even though the 8-client estimate stays 11/11.
The real 21-point, 8-client number comes from 11237283.

### MOL-25b. Results: the aligned calibration wins on ChEMBL, and α=0 finally moves (2026-10-09)

All three arms COMPLETED on cc, exit 0, no tracebacks: 11237283 (5:13:21), 11237284 (6:21:44),
11237285 (3:57:52). Test split, strict scorer, round 100, dev-selected on `pct_calib_err_penalized`.
**Single seed.**

| arm | penalized | pct err (worst) | in-support | out-of-support | reach | spearman |
|---|---:|---:|---:|---:|---:|---:|
| `fed-aligned-off` | **0.184** | **0.172** (0.244) | 0.162 | **0.178** | **0.414** | **0.898** |
| `fed-shared-off` | 0.186 | 0.178 (**0.237**) | 0.161 | 0.193 | 0.369 | 0.883 |
| `local-off` | 0.200 | 0.192 (0.288) | 0.161 | 0.217 | 0.324 | 0.843 |

**Primary contrasts (per-client paired bootstrap, 95% CI):**

1. **`fed-aligned-off` − `local-off`: better on 6/8 clients, worse on 0/8.** The C1 contrast, matched,
   with o_i on both sides. Largest gains on the one-sided clients: CHEMBL243 −0.041, CHEMBL228 −0.038,
   CHEMBL2835 −0.027, CHEMBL4078 −0.021, CHEMBL240 −0.018, CHEMBL325 −0.015. **This is the first
   unconfounded federated win on ChEMBL.**
2. **`fed-aligned-off` − `fed-shared-off`: better on 2/8, worse on 1/8** — much weaker than Newsroom's
   result. Means are near-tied on the penalized metric (0.184 vs 0.186). Aligned's clear advantage is
   confined to **CHEMBL4078** (−0.032, the middle-only client) and CHEMBL325 (−0.013); it loses
   marginally on CHEMBL2039 (+0.011).

   ⚠️ **HARDWARE-CONFOUNDED; NOT DIRECTIONAL (CAL-29).** MOL-25's three arms each landed on a
   different GPU type, which was not checked at submission: `fed-aligned-off` (11237283) ccc0284
   **A100 80GB PCIe**, `fed-shared-off` (11237284) ccc0389 **A100-SXM4-80GB**, `local-off` (11237285)
   ccc0465 **H200**. The aligned-vs-shared gap is 0.002 penalized, inside the ≤0.005 band CAL-29
   reports GPU mixing made unreadable on Newsroom, so **no directional claim should be drawn from
   this contrast** — it is a near-tie confounded by hardware, not a measured near-tie. The
   aligned-vs-local contrast (6/8, 0/8, per-client −0.015 to −0.041) is far outside that band and
   stands, though it too spans A100-PCIe against H200. A hardware-matched `fed-shared-off` rerun on
   ccc0284 would settle it; the user has deferred that decision until MOL-28's τ sweep is in, and it
   is **not queued**. MOL-28 is pinned to ccc0284 precisely so the sweep cannot inherit this.
3. `fed-shared-off` − `local-off`: better on 4/8, worse on 1/8. So **both** ways of sharing the shape
   beat not sharing; the choice between them is second-order on this dataset.

### The mechanism: the endpoints, and only the endpoints

Per-prompt×alpha cell, equal weight over clients (MOL-20's definition, not curve means):

| arm | metric | α=0 | α=.25 | α=.5 | α=.75 | α=1 | endpoints | interior |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fed-aligned-off` | pct err | **0.197** | 0.161 | 0.175 | 0.171 | **0.150** | **0.174** | 0.169 |
| `fed-shared-off` | pct err | 0.207 | 0.161 | 0.175 | 0.172 | 0.178 | 0.192 | 0.170 |
| `local-off` | pct err | 0.248 | 0.165 | 0.173 | 0.174 | 0.200 | 0.224 | 0.171 |

**The interior is a dead tie across all three arms (0.169–0.171), and the entire difference is at the
endpoints** (0.174 / 0.192 / 0.224). In-support error is likewise identical (0.162/0.161/0.161).

**α=0 has moved for the first time.** MOL-18/MOL-20 measured α=0 at 0.285–0.304 in every arm and
concluded it was structural — "the α=0 cell is the dominant error and no arm touches it", with the
achieved percentile stuck at 0.306 and an oracle remap moving it by −0.000. Here:

| arm | α=0 cell error | achieved percentile at α=0 | mean o_i |
|---|---:|---:|---:|
| MOL-18 arms (no offset) | 0.285–0.304 | 0.306 | — |
| `local-off` | 0.248 | 0.248 | −0.192 |
| `fed-shared-off` | 0.207 | 0.205 | −0.405 |
| `fed-aligned-off` | **0.197** | **0.196** | −0.371 |

So the private offset moves α=0 from 0.306 to 0.196 — the fix MOL-21/MOL-22 proposed, now confirmed
under a shared per-layer shape — and **the fed arms learn roughly twice the offset local does**
(−0.37/−0.41 against −0.19), which is where their endpoint advantage comes from.

**Hypothesis for why (not established):** with the shape shared or pooled, a client's shape is
anchored by the others, so the offset is the only free parameter able to absorb a level shift. In
local mode the private shape and private offset are both free and partly redundant, so the offset
settles at half the magnitude and α=0 stays high. Testable by a local arm with the shape frozen.

**CHEMBL228 is the clearest case of borrowing working.** Its support is [0.46, 0.97] — no low side at
all. Achieved percentile at α=0: `local-off` **0.466**, i.e. it cannot get below its own data floor;
`fed-aligned-off` 0.300, `fed-shared-off` 0.270. The shared calibration lets a one-sided client reach
a region its own data never covered.

### Offset trajectories: no drift on ChEMBL (CAL-19 asked)

Mean o_i over clients plateaus in every arm — no runaway:

| arm | r10 | r30 | r50 | r70 | r99 | final range |
|---|---:|---:|---:|---:|---:|---|
| `fed-aligned-off` | −0.305 | −0.368 | −0.381 | −0.388 | −0.371 | [−0.602, −0.083] |
| `fed-shared-off` | −0.308 | −0.384 | −0.394 | −0.401 | −0.405 | [−0.658, −0.073] |
| `local-off` | −0.090 | −0.150 | −0.171 | −0.182 | −0.192 | [−0.247, −0.081] |

`local-off`'s o_i stays **negative and bounded near −0.19**, squarely inside MOL-22's −0.09…−0.25
range, and converges rather than diverging. It does **not** reproduce Newsroom's NR-65 upward drift to
+0.78. **So `local-off` is a valid reference on ChEMBL and the extra plain `local` arm the coordinator
floated is not needed here.**

**Correction (CAL-21).** An earlier version of this entry said the offset's *sign* differs by dataset,
with Newsroom's positive. That was wrong — I generalised from NR-65's `local-off` drift. Verified
directly against `nr66-fed-aligned-off-tp3000` (11215858, test, round 100): Newsroom's federated
offsets are **negative 8/8, mean −0.2114, range [−0.6230, −0.0335]**, nypost.com most negative at
−0.6230 — the same sign as ChEMBL. The positive drift is a **`local-off`-on-Newsroom instability**,
not a dataset property.

### Why the fed offsets are larger: o_i learns the client's position on the global scale

The correction above led to the measurement that turns MOL-25b's mechanism hypothesis into a result.
Spearman correlation between a client's **support floor** and its learned **o_i**:

| arm | spearman(support_lo, o_i) | pearson |
|---|---:|---:|
| `fed-aligned-off` | **−0.905** | −0.862 |
| `fed-shared-off` | **−0.905** | −0.860 |
| `local-off` | **+0.190** | +0.021 |
| Newsroom `fed-aligned-off` | −0.429 | −0.908 |

Under a shared or pooled shape the offset is almost perfectly monotone in the client's support floor:

| client | support | o_i |
|---|---|---:|
| CHEMBL243 | [0.01,0.41] | −0.0832 |
| CHEMBL204 | [0.03,0.92] | −0.2192 |
| CHEMBL2039 | [0.13,0.90] | −0.3992 |
| CHEMBL2835 | [0.14,0.88] | −0.3412 |
| CHEMBL240 | [0.16,0.96] | −0.4346 |
| CHEMBL325 | [0.17,0.99] | −0.4735 |
| CHEMBL4078 | [0.30,0.80] | −0.4178 |
| CHEMBL228 | [0.46,0.97] | −0.6018 |

**Under a private shape it is uncorrelated** (+0.190, pearson +0.021): `local-off`'s offsets are all
≈−0.19 with no structure. So the two arms' offsets are not just different in magnitude, they are
different in *kind* — the fed arms' o_i encodes a real per-client quantity (how high that client sits
on the global percentile scale, hence how far it must be shifted down to reach α=0), while local's is
an undifferentiated nudge.

This supersedes the "hypothesis, not established" paragraph above with a measurement: the shape and
the offset compete for the same degree of freedom, and only when the shape is tied across clients
does the offset get identified. It also explains the rest of the entry — why CHEMBL228 (highest
floor, 0.46) takes the largest offset and gains most, and why α=0 moves only in the fed arms.
**Prediction for the frozen-shape local probe:** if a local arm's shape is frozen at the identity, its
o_i should become support-tracking and its α=0 error should fall toward the fed arms'. If it does not,
the sharing is doing something beyond identifying the offset.

### Disagreement at 21 points, 8 clients (CAL-19)

    unrestricted 0.05377  logged 0.05377 (log round 99)  MATCH
    RESTRICTED   0.05377  (21 of 21 grid points keep >= 2 clients at >= 5% weight)
    dilution factor 1.00x

**The prediction from MOL-18's 11-point counts holds on the real 21-point grid: 21/21 qualify**, 6–8
of 8 clients clearing the floor at every α, so the caveat in MOL-25's plan is resolved and ChEMBL's
number is undiluted and directly comparable with Newsroom's.

⚠️ **My falsification condition is refuted, which is the most interesting result here.** I predicted
that high disagreement would make the pooled h̄_l an average of incompatible curves, so
`fed-aligned-off` would lose to `local-off` on the one-sided clients. The opposite happened.
**ChEMBL's clients disagree nearly twice as much as Newsroom's (0.0538 vs 0.0280) and aligned still
wins 6/8, 0/8** — and it wins *most* on exactly the clients whose own curves deviate furthest from
the pooled table:

| client | support | support_gap | `fed-aligned-off` − `local-off` |
|---|---|---:|---:|
| CHEMBL243 | [0.01,0.41] | **0.1156** | **−0.041*** |
| CHEMBL228 | [0.46,0.97] | **0.0751** | **−0.038*** |
| CHEMBL4078 | [0.30,0.80] | 0.0392 | −0.021* |
| CHEMBL2039 | [0.13,0.90] | 0.0380 | +0.006 |
| CHEMBL240 | [0.16,0.96] | 0.0357 | −0.018* |
| CHEMBL204 | [0.03,0.92] | 0.0317 | +0.006 |
| CHEMBL325 | [0.17,0.99] | 0.0250 | −0.015* |
| CHEMBL2835 | [0.14,0.88] | 0.0214 | −0.027* |

The rank correlation is clear at the extremes: the two clients with the largest support_gap get the
two largest gains, and the two clients with no significant gain (CHEMBL2039, CHEMBL204) are broad-
support donors. **Reading:** a one-sided client's own out-of-support curve is an extrapolation from no
data, so an imperfect pooled table beats it easily — disagreement measures how much a client's curve
differs from the pool, not how much the pool hurts it. So aligned's benefit is **not** conditional on
client agreement, at least up to disagreement ≈ 0.054. Tools:
[scripts/disagreement_report.py](../../scripts/disagreement_report.py),
[scripts/alpha_cell_report.py](../../scripts/alpha_cell_report.py) (new; reuses
`fedsteer.data.alpha_reference_from_json` and `metrics`' own cell definition rather than
reimplementing the percentile mapping).

### MOL-25c. Median α is the better descriptor; and a qualification to MOL-25b's mechanism claim (2026-10-09)

**Bug in my own `scripts/offset_support_report.py`, found by newsroom-iter1 (CAL-21) and fixed.** The
glob `eval_round_[0-9]*__*.json` also matches E2 and baseline evals — `*` spans
`_e2_plugin_nall` — and `sorted()[-1]` then reads the **held-out clients' plugin offsets** instead of
the arm's own. On NR-62 that produced a spurious n=4, ρ=−1.000. Demonstrated on
`nr60-fed-shared-off`: the loose glob matches **11** files there and would pick
`eval_round_0100_e2_plugin_nall__…`; the strict pattern
`eval_round_[0-9][0-9][0-9][0-9]__*.json` (four digits immediately followed by `__`) matches **1**.

- Fixed in `offset_support_report.py` **and in `scripts/alpha_cell_report.py`**, which had the same
  glob. Both now share a `test_eval()` helper that also **refuses to guess** when several test rounds
  are present, requiring `--round`.
- **MOL-25b's numbers are unaffected**, verified rather than assumed: the three mol25 runs have no E2
  or baseline evals, and the loose glob's other matches were `_dev__` files that the existing filter
  already removed. Strict and loose globs select the same single file for all three.

**Median α is a better descriptor than the support floor in every arm of both datasets.** Median α is
`ref.cdf(local_q.quantile(0.5))`, built exactly as `fedsteer.data.client_support` builds the bounds
but at q=0.5.

| arm | floor ρ | floor r | **median ρ** | median r | mean o_i | range |
|---|---:|---:|---:|---:|---:|---|
| `mol25-fed-aligned-off` | −0.905 | −0.862 | **−1.000** | −0.990 | −0.3713 | [−0.602, −0.083] |
| `mol25-fed-shared-off` | −0.905 | −0.860 | **−1.000** | −0.984 | −0.4050 | [−0.658, −0.073] |
| `mol25-local-off` | +0.190 | +0.021 | **+0.405** | +0.231 | −0.1921 | [−0.247, −0.081] |
| `nr62-fed-aligned-off` | −0.571 | −0.894 | **−0.786** | −0.854 | −0.2094 | [−0.618, −0.021] |
| `nr60-fed-shared-off` | −0.643 | −0.913 | **−0.833** | −0.851 | −0.2222 | [−0.663, −0.020] |
| `nr60-local-off` (round 60) | −0.500 | −0.481 | **−0.786** | −0.681 | **+0.5227** | [+0.242, +1.101] |

On ChEMBL the fed arms are **perfectly monotone in median α** (ρ = −1.000, both of them). The floor
mis-ranks exactly the client it should: **CHEMBL4078** is middle-only, so its floor (0.30) is high
while its median (0.514) is mid-range — under the floor ordering its offset looks out of place, under
the median ordering it sits correctly between CHEMBL2039 (0.491) and CHEMBL240 (0.630). The median is
the client's centre of mass; the floor is an edge. **One descriptor serves both datasets.**

⚠️ **This qualifies MOL-25b's mechanism claim and I am not leaving that implicit.** MOL-25b said the
offset "is only *identified* when the shape is tied across clients", on the strength of ChEMBL's
`local-off` showing no correlation (+0.190 floor, +0.405 median). But **Newsroom's `local-off` tracks
the median at ρ = −0.786** — the same ordering as its fed arms — while its offsets are **positive,
+0.24 to +1.10**. So that arm learned the right *relative* structure with a badly wrong *absolute
level* (the NR-65 drift), rather than learning nothing.

The defensible statement across both datasets is therefore narrower than MOL-25b's:

- sharing or pooling the shape makes o_i **correctly levelled** — negative, bounded, and monotone in
  median α, in all four fed arms on both datasets;
- a private shape leaves the level unidentified, and what goes wrong then is **dataset-dependent**:
  ChEMBL's `local-off` collapses to a structureless ≈−0.19 for everyone, Newsroom's keeps the ordering
  but drifts to the wrong sign.

That still supports MOL-25's result — the fed arms' endpoint advantage comes from a correctly levelled
offset, and α=0 moves only there — but "the offset is not identified without sharing" is too strong as
written.

**Closed by the user (CAL-22, 2026-10-09).** The offset-vs-data-position analysis stops at the
statement above; no further correlations or descriptors. The **frozen-shape local probe is DROPPED by
the user** and will not be run, so the sharper prediction it would have tested — that freezing the
shape fixes the offset's *level* rather than creating an ordering `nr60-local-off` already had —
stays untested and should not be cited as pending work.

## MOL-26. Baselines B1 / B3 / B4 for fed-aligned-off on ChEMBL (2026-10-09)

Spec from the coordinator (CAL-25), on the user's request; mirrors Newsroom's NR-71/NR-73 (CAL-23).
Method under test `mol25-fed-aligned-off` (11237283, round 100). Test split, strict
`clogp_residual_deco_strict`, **150 prompts per client** (MOL-25's size), penalized error primary.
Launch [MOL-26.sh](launch/MOL-26.sh), pre-flight [MOL-26_smoke.sh](launch/MOL-26_smoke.sh).

| arm | cc job | state | elapsed |
|---|---|---|---|
| `mol26-b1-k0` | 11242545 | COMPLETED | 0:35 |
| `mol26-b1-k3` | 11242546 | COMPLETED | 0:52 |
| `mol26-b1-k3-base` | 11242547 | COMPLETED | 1:34 |
| `mol26-b4-caa` | 11242548 | **FAILED** (code bug, see below) | 0:29 |
| `mol26-b3-local-off` | 11242549 | COMPLETED | 0:34 |

### Results

| arm | penalized | pct err (worst) | in-sup | out-of-sup | reach | spearman | **unscorable** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fed-aligned-off` | **0.184** | **0.172** (0.244) | 0.162 | **0.178** | **0.414** | **0.898** | 0.071 |
| `local-off` | 0.200 | 0.192 (0.288) | 0.161 | 0.217 | 0.324 | 0.843 | 0.034 |
| `B3` merged local | 0.253 | 0.249 (0.307) | 0.198 | 0.300 | 0.198 | 0.723 | **0.013** |
| `B1` k=0 client | 1.000 | **n/a** | n/a | n/a | n/a | n/a | **1.000** |
| `B1` k=3 client | 0.993 | **n/a** | n/a | n/a | n/a | n/a | **1.000** |
| `B1` k=3 base | 1.000 | **n/a** | n/a | n/a | n/a | n/a | **1.000** |

Paired bootstrap, 95% CI: **`fed-aligned-off` − `B3`: better on 8/8, worse on 0/8** (−0.055 to −0.097
overall, −0.068 to −0.153 out-of-support). `local-off` − `B3`: better 7/8, worse 0/8 (CHEMBL243 n.s.).
So merging separately-trained local directions is clearly worse than either training jointly or not
sharing at all. Cleaner than Newsroom, where B3 lost 7/8 with reuters an exception (CAL-23).

### ⚠️ B1 on ChEMBL is a FORMAT failure, not a calibration failure

**`unscorable_row_rate = 1.000` for all 8 clients in all three B1 cells** — exactly 1.000, not
approximately. Every prompt produced output the strict deco scorer could not score, so
`pct_calib_err` is `None` throughout and the penalized figures (1.000 / 0.993 / 1.000) are **pure
unscorability penalty carrying no calibration information**.

Diagnosed from the smoke before the real run: B1's generations are plain SMILES such as `O=C(N)C`
(acetamide — valid, but a trivial fragment with no requested core), whereas this dataset's targets are
**decoration sets in attachment-point notation**, e.g. `[1*]C.[2*]C.[3*]C(=O)NS(N)(=O)=O.[4*]C.[5*]C`.
The template is not at fault — `prompt_with_level_molecule` carries "Decorate this core scaffold:
{scaffold}", the records carry `scaffold` (not `problem`/`article`) so the auto-dispatch picks the
molecule template, and k=3's shots show the notation explicitly. **In-context examples do not teach
the format**: k=3 recovers only isolated cells (0.984–0.997 vs 1.000) and never a complete row.

**So "fed-aligned-off beats B1 8/8" is not a meaningful claim on this task** and is not reported as
one. B1 here is not producing badly-calibrated decorations; it is not producing decorations. This is
*not* the same kind of evidence as Newsroom's B1 (0.38), where the model does emit summaries and the
error is genuinely calibration. Raised with the coordinator: a fair B1 for this task would need a
format-constrained variant or a non-strict supplementary row. **Neither was done** — both redefine the
baseline, which is the user's call.

### The validity / calibration trade-off: aligned's win is not bought by caution

**`B3` has the lowest unscorable rate (0.013) and the worst calibration; `fed-aligned-off` has the
highest of the three real arms (0.071) and the best.** So the method pushes harder — reach 0.414
against B3's 0.198 — and still wins on the *failure-penalized* metric despite producing more
unscorable output. That is a stronger statement than the headline number: the advantage is not an
artefact of generating conservatively.

Related: ALIGN's unscorable rate on **CHEMBL2039 is 0.235**, far above its 0.013–0.140 elsewhere, and
CHEMBL2039 is one of the two clients where ALIGN lost to LOCAL in MOL-25 (+0.006, n.s.). **That loss
is unscorability, not miscalibration.**

### B4 failed: a third unguarded `None`, and the fix is a scientific choice

CAA vectors fit correctly (norms 3.8–10.1 over the 8 clients, hidden norms 52–56), then:

    eval_baselines.py:126  g_best = min(fit, key=fit.get)
    TypeError: '<' not supported between instances of 'NoneType' and 'float'

`fit[g] = result_from_grid(...)["pct_calib_err"]` is `None` whenever **no dev cell scored at that
gain**, so at least one gain in the widened grid destroys validity completely — the risk CAL-23/CAL-25
named for activation steering on structured output, arriving as a crash rather than a bad number.

Two fixes, and the choice matters: skipping the `None` gains hides how much of the grid is invalid and
biases selection toward whatever worked, whereas selecting on `pct_calib_err_penalized` (never `None`)
ranks a validity-destroying gain correctly as worst. **Recommended the latter**, as an opt-in flag so
NR-71's B4 stays reproducible. `eval_baselines.py` is not in CONVENTIONS §5's shared list but both
datasets use it, so it was treated as shared and **no change was made** pending the coordinator.

**Prediction recorded before B4 is rerun:** given the B1 format finding, the chosen gains should
cluster at the **low** end (0.05–0.2), because steering strong enough to move logP also breaks the
notation. If gains land high with good validity, the format story is wrong.

**A third instance of one bug class.** `fedsteer/monitor.py` (CAL-20, fixed), `eval_baselines.py:126`
(above), and `scripts/compare_runs.py:142` — the last dying on
`f"{...['pct_calib_err']:14.3f}"` whenever any run has a client with nothing scorable, which is how
the first attempt at this table crashed. The pattern: **`pct_calib_err` is legitimately `None`
whenever nothing scores, and the codebase assumes that is rare** — true for summaries, routine for
molecules under a strict scorer. Proposed convention: treat it as nullable at every use site and
prefer `pct_calib_err_penalized` wherever a single number must exist. Worked around here with no code
change (comparison run without the B1 arms, a paired bootstrap on an arm with zero scorable cells
being meaningless anyway), so MOL-26 needed no shared-code grant.

### Traps handled (three from CAL-25, a fourth found here)

| trap | handling |
|---|---|
| `SCORER` defaults to `clogp_residual` | passed as `clogp_residual_deco_strict` |
| `DATA` defaults to `data/chembl_fed` | passed as `data/chembl_deco_skew_pruned` |
| `SIZE` hard-coded `--max_prompts 200` | new opt-in `PROMPTS`, default 200, passed 150 |
| `LOCAL_SNAP=best` selects on `pct_calib_err` | `round_0100.pt` passed explicitly |
| **`--caa_gains` defaults to the NARROW grid** (0.05–0.8, no 1.2/1.6/2.4/3.2) | new opt-in `CAA_GAINS`; without it the "none at the grid top" check would have been vacuous |

Plus `B1_SETS` so each B1 cell takes its own job name. All defaults reproduce the previous behaviour.
`CAA_GAINS` and `B1_SETS` use `+` separators, since `--export` is itself comma-separated and truncates
a value at its first comma (MOL-22b). Pre-flight smoke 11242038 COMPLETED across all three paths;
B1's score grid verified **α-invariant** (max within-row spread exactly 0) where cells are scorable,
as it must be with `lora_B_d` zeroed.

## MOL-26a. Baselines rerun after the CAL-26 template fix: aligned beats all four, 8/8 (2026-10-09)

Reruns MOL-26's B1 ×3 and B4 with the fixed B1 decoration prompt and `--caa_select`. MOL-26's B1
numbers are void; B3 stands from MOL-26. Launch [MOL-26a.sh](launch/MOL-26a.sh), smokes 11248372.
All four COMPLETED: 11250706 (0:41), 11250707 (0:49), 11250708 (1:12), 11250709 (3:01), exit 0.

| arm | penalized | pct err (worst) | in-sup | out-of-sup | reach | spearman | unscorable |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fed-aligned-off` | **0.184** | **0.172** (0.244) | 0.162 | **0.178** | 0.414 | **0.898** | 0.071 |
| `local-off` | 0.200 | 0.192 (0.288) | 0.161 | 0.217 | 0.324 | 0.843 | 0.034 |
| `B3` merged local | 0.253 | 0.249 (0.307) | 0.198 | 0.300 | 0.198 | 0.723 | 0.013 |
| `B4` CAA (gain 0.05) | 0.366 | 0.364 (0.419) | 0.227 | 0.497 | 0.035 | **−0.099** | 0.004 |
| `B1` k=0 client | 0.355 | 0.352 (0.419) | 0.224 | 0.477 | 0.048 | 0.043 | 0.008 |
| `B1` k=3 client | 0.467 | 0.313 (0.406) | 0.188 | 0.423 | 0.124 | 0.268 | 0.570 |
| `B1` k=3 base | 0.766 | *0.186* (0.246) | *0.352* | *0.102* | *0.886* | 0.713 | **0.981** |

**`fed-aligned-off` beats every baseline on 8/8 clients, 0/8 worse** (paired bootstrap, 95% CI):
vs B4, vs B1 k=0, vs B1 k=3, vs B1 k=3-base, and vs B3 (from MOL-26). No exceptions on any client —
cleaner than Newsroom, where B3 lost 7/8 with reuters an exception (CAL-23).

(`mean o_i` is deliberately omitted for the B1/B4 rows: those arms zero `lora_B_d`, so the offset
they inherit from the loaded snapshot — all reporting ALIGN's −0.3713 — multiplies a null direction
and means nothing.)

### B4: every client picks the grid MINIMUM, and the dev curve is monotone

All 8 clients chose **gain 0.05**, the smallest value offered, with penalized dev error rising
monotonically across the whole widened grid:

| client | 0.05 | 0.1 | 0.3 | 0.5 | 0.8 | 1.2 | 1.6 | 2.4 | 3.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CHEMBL2039 | **0.317** | 0.320 | 0.355 | 0.376 | 0.483 | 0.590 | 0.718 | 0.827 | 0.827 |
| CHEMBL4078 | **0.328** | 0.340 | 0.357 | 0.376 | 0.441 | 0.566 | 0.664 | 0.820 | 0.820 |
| CHEMBL243 | **0.434** | 0.442 | 0.466 | 0.510 | 0.638 | 0.705 | 0.815 | 0.875 | 0.875 |

**No interior optima**, unlike Newsroom's 0.5–0.8 (CAL-23). On ChEMBL activation steering is harmful
at every strength tested, and the optimum sits *at the boundary* — so the true optimum may lie below
0.05 and **the grid wanted extending downward, not upward**. Widening to 3.2 was the wrong direction:
nothing lands near the top, where error is uniformly worst (0.82–0.88). A downward extension
(0.01/0.02) would show whether 0.05 is a real optimum or just the grid edge; **not approved by the
user, not launched.**

The metrics say why: at its best gain B4's **near-tie rate is 0.931** and reach 0.035, so it barely
changes the output at all — and its **Spearman is −0.099**, i.e. what little it does change moves
*against* α. CAA on this task is either inert or wrong-signed.

### B1 k=0 ignores the level instruction

Spearman **0.043**, near-tie **0.913**, reach 0.048: k=0 prompting produces essentially the same
output regardless of α. With the copying analysis (below) showing 65% of its outputs are verbatim
training targets and only 73–167 distinct of 750, the picture is a fine-tuned model emitting a small
memorised set and ignoring the requested level entirely.

### ⚠️ B1 k=3 base is a selection-effect trap — the strongest argument for the penalized metric

Read on strict `pct_calib_err` alone, **`B1` k=3 base looks competitive with the trained method**:
0.186 against ALIGN's 0.172, with *better* out-of-support error (0.102 vs 0.178) and far higher reach
(0.886 vs 0.414). It is an untrained base model with three examples in the prompt.

It is an artefact. Its **unscorable rate is 0.981** — it fails to produce a scorable decoration 98% of
the time, and `pct_calib_err` is computed only on the ~2% that survive, which are the easy cases. The
penalized metric, which counts an unscorable cell as error 1, puts it last at **0.766**. Anyone
quoting the strict metric here would conclude an untrained model rivals the method. This is exactly
why CAL-25 made penalized primary, and it is worth one line in the paper.

### Copying vs distribution shift (CAL-26 question), 5870 generations per arm

| | `=a shot` | `=any train target` | `in top10` | distinct | unscorable |
|---|---:|---:|---:|---:|---:|
| k=0 (no shots shown) | 0.000 | **0.653** | 0.246 | 73–167 / 750 | **0.008** |
| k=3 | **0.130** | 0.542 | 0.220 | 123–311 / 750 | **0.570** |

**The memorisation belongs to the fine-tuned adapter, not to the shots.** With no examples in the
prompt at all, 65% of k=0's outputs are verbatim training targets — so "= any train target" was never
evidence about in-context learning, and the 0.519 measured on the 2-client smoke should not have been
read that way. k=3 copies its shown examples only 13% of the time and *reduces* verbatim training
reproduction (0.542 vs 0.653) while *increasing* diversity. What the shots do is **break validity**:
unscorable rises 0.008 → 0.570. So k=3 losing to k=0 is **distribution shift, not copying** — the
shots pull the model off the output distribution it was fine-tuned on. Tool:
[scripts/b1_copy_report.py](../../scripts/b1_copy_report.py); it reads `clients[c]["outputs"]` (all
5870 generations) rather than the `samples` block, which holds only the first prompt per client, and
rebuilds the shot pool with `build_clients` on the run's own `max_train`/seed/`tie_break`/`alpha_mode`
because `pick_shots` ranks on an `alpha` that is not in the jsonl.

**Hardware:** the B1 arms ran on ccc0284 and B4 on ccc0390, so the B4 row spans a node boundary. All
the margins here are ≥ 0.17, far outside the ≤ 0.005 band where that matters (contrast MOL-25b's
aligned-vs-shared note).

### Pending at 2026-10-09 (monitoring stopped at the user's request; jobs left RUNNING)

Nothing was cancelled on cc. When picking this up, check these and then report to the coordinator:

| jobs | arms | what to do |
|---|---|---|
| 11254743, 11254744 | MOL-27 `fed-linear-off`, `fed-shared-soff-sadapter` | both on ccc0284, so internally hardware-matched. Table like MOL-25 plus per-α cells, offsets, unscorable rate, paired contrasts vs `mol25-fed-aligned-off` / `mol25-fed-shared-off` / `mol25-local-off`; report in- and out-of-support separately for sadapter (CAL-27). |
| 11256739-42 | MOL-28 τ sweep: tp300, tp1000, tp3000, cnt | pinned `--partition=dali --nodelist=ccc0284`; **verify from each log's `GPU 0:` line that all four really landed there** before reporting any ≤0.005 difference. Rows per τ with τ=100 = MOL-25's 11237283; paired contrasts vs τ=100; pool-weight share of CHEMBL243 at high α and CHEMBL228 at low α; state whether τ matters with o_i present (CAL-29). |

Deferred by the user, **not queued**: the hardware-matched `fed-shared-off` rerun on ccc0284 (decide
after MOL-28), and B4's downward gain-grid extension below 0.05.

Analysis tools ready and validated: `scripts/disagreement_report.py`,
`scripts/alpha_cell_report.py`, `scripts/offset_support_report.py`, `scripts/b1_copy_report.py`.

## MOL-27. NR-74 ablations on ChEMBL: the nonlinear shape carries the whole benefit; pure FedAvg ties the method (2026-10-09)

Spec CAL-27 (user request). Both COMPLETED on **ccc0284, A100 80GB PCIe** — the same node as
`mol25-fed-aligned-off` (11237283), so both headline contrasts are **hardware-matched**.
11254743 `fed-linear-off` (3:28:56), 11254744 `fed-shared-soff-sadapter` (5:24:35), exit 0.
Launch [MOL-27.sh](launch/MOL-27.sh), smokes 11253935/36.

| arm | penalized | pct err (worst) | in-sup (worst) | out-of-sup | reach | spearman | **unscorable** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fed-shared-soff-sadapter` | **0.178** | 0.176 (0.240) | 0.162 (**0.206**) | 0.184 | 0.404 | **0.902** | **0.013** |
| `fed-aligned-off` (MOL-25) | 0.184 | **0.172** (0.244) | 0.162 (0.234) | **0.178** | **0.414** | 0.898 | 0.071 |
| `fed-shared-off` (MOL-25) | 0.186 | 0.178 (0.237) | 0.161 (0.229) | 0.193 | 0.369 | 0.883 | — |
| `local-off` (MOL-25) | 0.200 | 0.192 (0.288) | 0.161 (0.239) | 0.217 | 0.324 | 0.843 | 0.034 |
| `fed-linear-off` | 0.202 | 0.190 (0.268) | 0.163 (0.248) | 0.216 | 0.322 | 0.864 | 0.076 |

Per-α cells (equal weight over clients, MOL-20 definition, penalized):

| arm | α=0 | α=.25 | α=.5 | α=.75 | α=1 | endpoints | interior |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fed-shared-soff-sadapter` | **0.207** | 0.180 | 0.176 | **0.167** | **0.160** | **0.184** | 0.174 |
| `fed-aligned-off` | 0.212 | 0.164 | 0.176 | 0.176 | 0.190 | 0.201 | **0.172** |
| `local-off` | 0.259 | 0.171 | 0.179 | 0.178 | 0.213 | 0.236 | 0.176 |
| `fed-linear-off` | **0.272** | 0.167 | 0.175 | 0.172 | 0.223 | 0.248 | **0.172** |

### 1. The nonlinear per-layer shape carries the entire benefit of sharing

`fed-linear-off` (g_i(α) = o_i + α) is **worse than `fed-aligned-off` on 7/8 clients, better on 1/8**
(CHEMBL2039 −0.018), with per-client effects +0.015 to +0.037 overall and +0.022 to +0.072
out-of-support. It also loses to `fed-shared-off` 5/8, 0/8.

**The damage is entirely at the endpoints: the interior cell error is identical (0.172 vs 0.172)**
while α=0 goes 0.212 → 0.272 and α=1 goes 0.190 → 0.223. A linear calibration is adequate in the
middle of the α range and fails exactly where the curve has to bend.

**The sharper statement: `fed-linear-off` (0.202) ≈ `local-off` (0.200).** Ablating the shape to the
identity throws away *all* of the benefit of federating — sharing a **linear** calibration is worth
nothing over not sharing at all. Everything MOL-25 attributed to federation lives in the nonlinear
per-layer shape, not in sharing per se.

### 2. Pure FedAvg TIES the method — a tie, not a win

The means favour `sadapter` (0.178 vs 0.184) but the per-client paired bootstrap is **2/8 better, 2/8
worse, 4 n.s.**: better on CHEMBL204 (−0.017) and CHEMBL2039 (−0.013), worse on CHEMBL240 (+0.021)
and **CHEMBL4078 (+0.018)**. Against `local-off` it wins 3/8, 0/8 — clearly weaker than
`fed-aligned-off`'s 6/8, 0/8. **No directional claim is warranted**; the honest statement is that a
single global model with one shared calibration *matches* the best personalised design.

That CHEMBL4078 is where `sadapter` loses is mechanistically consistent: it is the **middle-only**
client, and it is also where `fed-aligned-off` beat `fed-shared-off` most in MOL-25 (−0.032). The
per-layer **pooled table** is what helps the client whose support is narrowest, and `sadapter` does
not have one.

### 3. The win on the penalized metric is VALIDITY, not calibration — decomposed

This is the part a single number hides:

| | strict `pct_calib_err` | unscorable | penalized |
|---|---:|---:|---:|
| `fed-aligned-off` | **0.172** (better) | 0.071 | 0.184 |
| `fed-shared-soff-sadapter` | 0.176 | **0.013** (5× lower) | **0.178** |

**On calibration `fed-aligned-off` is better; on validity `sadapter` is 5× better; the penalized
metric nets out marginally in `sadapter`'s favour.** Its worst-client in-support error is also much
better (0.206 vs 0.234). The plausible reason is data volume: a shared adapter is trained on all
eight clients' data at once, so it learns the decoration format better, while eight private adapters
each see one client's. Same validity/calibration trade-off as MOL-26a's B3 row, in the opposite
direction.

### 4. Offsets: MOL-25c's mechanism refined, not contradicted

| arm | learned offsets | mean |
|---|---|---:|
| `fed-linear-off` | **8 distinct**, −0.0692 … −0.6891, monotone in support floor | −0.4137 |
| `fed-aligned-off` (MOL-25) | 8 distinct, −0.083 … −0.602 | −0.3713 |
| `fed-shared-soff-sadapter` | **one value for all 8 clients** | −0.0717 |

`fed-linear-off` learned the **largest-magnitude offsets yet** while keeping the monotone ordering
(CHEMBL243 floor 0.01 → −0.069; CHEMBL228 floor 0.46 → −0.689) — exactly as MOL-25c predicts when the
shape is removed and the offset is the only free calibration parameter left.

`sadapter`'s single shared offset collapsed to −0.0717, about a fifth of the private mean, **and it
performed fine.** That refines rather than refutes MOL-25c: the offset encodes *where a client sits
relative to the others*, so once a shared adapter makes every client's model identical there is no
per-client position to encode and one offset suffices. The collapse toward 0 (rather than toward the
private mean of −0.371) is what a single parameter does when clients would pull it different ways.

**Newsroom mismatch:** CAL-27 noted the archive A2 was *worse in support and better out of it*. ChEMBL
is the reverse — `sadapter` matches in-support (0.162) with a **better** worst client (0.206 vs
0.234) and is slightly **worse** out-of-support (0.184 vs 0.178).

**Caveats.** Single seed. `fed-linear-off`-vs-`fed-aligned-off` and `sadapter`-vs-`fed-aligned-off`
are hardware-matched (all ccc0284); `fed-linear-off`-vs-`fed-shared-off` crosses to ccc0389 (A100-SXM4)
and `sadapter`-vs-`local-off` to ccc0465 (H200), so those two carry the MOL-25b caveat. With
`warp=none` the code forces scope `model` (`warp_indices(targets, cfg.warp_scope if cfg.warp != "none"
else "model")`), so `fed-linear-off` has one shape per client by construction — it is an ablation of
*both* nonlinearity and per-layer resolution, not of nonlinearity alone.

## MOL-28. τ_pool sweep for fed-aligned-off: τ does not matter, and the reason is quantified (2026-10-10)

Spec CAL-29 (user request), mirroring Newsroom's NR-66/NR-69. All four COMPLETED, each ~5h08m, **all
on ccc0284** — so the sweep and its τ=100 reference (`mol25-fed-aligned-off`, 11237283) are
hardware-matched, which this experiment needs since it resolves differences of ~0.005.
11256739 tp300, 11256740 tp1000, 11256741 tp3000, 11256742 cnt. Launch [MOL-28.sh](launch/MOL-28.sh).
Configs verified per arm: `cov_pool`/`cov_tau_pool` as specified, grid 21, λ_max 0.01, aligned,
`lora.offset=true`, per-layer scope.

| τ | pct err (worst) | penalized | in-sup | out-of-sup | α=0 | α=1 | reach | spearman | sel. round | vs τ=100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **100** (MOL-25) | 0.172 (0.244) | 0.184 | 0.162 | 0.178 | 0.212 | 0.190 | 0.414 | 0.898 | 100 | — |
| 300 | 0.177 (0.250) | 0.187 | 0.166 | 0.182 | 0.220 | 0.186 | 0.399 | 0.885 | 100 | 1 better, **3 worse** |
| 1000 | **0.162** (0.229) | **0.182** | **0.157** | **0.162** | **0.242** | 0.168 | 0.461 | **0.914** | 100 | **2 better**, 1 worse |
| 3000 | 0.170 (0.244) | 0.189 | 0.167 | 0.169 | 0.226 | 0.189 | 0.461 | 0.901 | 100 | 1 better, 1 worse |
| count | 0.185 (0.264) | 0.190 | 0.179 | 0.188 | 0.235 | **0.161** | 0.401 | 0.891 | **60** | 1 better, **3 worse** |

(α=0 / α=1 are penalized per-prompt×alpha cells, MOL-20 definition.)

### τ does not matter — ChEMBL reproduces NR-69

Penalized error spans **0.182–0.190** across the whole sweep (0.008), and **no setting separates from
τ=100 by more than two clients** in either direction. τ=1000 is nominally best on almost every column
but is 2-better/1-worse on a single seed with a 0.002 penalized edge — inside the noise band even with
hardware matched. **The answer CAL-29 asked for: with o_i present, τ is not a hyperparameter that
needs tuning or defending.** Same conclusion as Newsroom, reached independently on a dataset whose
clients disagree about twice as much (MOL-25b: disagreement 0.054 vs 0.028).

### Why: τ controls how much a single dense client owns a grid point

Pool weight is `c/(τ+c)` (`fedsteer/coverage.py:107`), so τ sets how strongly the weights saturate.
Share of the total pool weight held by the two one-sided clients, derived from the counts:

| pooling | CHEMBL243 @ α=1 *(no data)* | CHEMBL243 @ α=0 *(dense)* | CHEMBL228 @ α=0 *(no data)* | CHEMBL228 @ α=1 *(dense)* |
|---|---:|---:|---:|---:|
| τ=100 | 0.0130 | **0.2804** | 0.0034 | 0.1878 |
| τ=300 | 0.0072 | 0.3558 | 0.0018 | 0.1986 |
| τ=1000 | 0.0047 | 0.4203 | 0.0011 | 0.1964 |
| τ=3000 | 0.0039 | 0.4518 | 0.0008 | 0.1919 |
| count | 0.0035 | **0.4728** | 0.0007 | 0.1880 |

Raw counts at the endpoints: CHEMBL243 has **721 at α=0 but 5 at α=1**; CHEMBL228 has **1 at α=0 and
287 at α=1**. Totals 1525 (α=0) and 1529 (α=1).

Two things follow, and they pull against each other:
- **Where a client has no data, higher τ correctly suppresses it further** (CHEMBL243 at α=1: 0.0130 →
  0.0035). That is not where the action is — it is already negligible at every τ.
- **Where a client is dense, higher τ lets it dominate.** CHEMBL243's share of the α=0 row rises from
  **28% to 47%** — and 47.3% is exactly its share of the raw counts (721/1525), i.e. count pooling is
  pure proportional weighting. So τ is really a dial between *equal-client* pooling (low τ, strong
  saturation) and *volume-proportional* pooling (high τ / count).

**That explains the α=0 pattern.** α=0 is the dominant error cell on this task (MOL-20), and raising τ
hands that row to CHEMBL243 — which suits CHEMBL243 and not the other seven. Hence τ=1000 has the
**best strict error (0.162) and the worst α=0 cell (0.242)**: it calibrates better on average by
fitting α=0 mostly to one client.

**Unscorability is NOT monotone in τ, and an earlier draft of this entry wrongly said it was.** The
measured rates are **τ=100 0.071, τ=300 0.060, τ=1000 0.122, τ=3000 0.115, count 0.034** — elevated
at τ=1000/3000 but *lowest of all* for count pooling, which is the most volume-proportional setting of
the five. So "higher τ spends validity" is contradicted by the count arm, and the honest statement is
narrower: **the two middle-high τ values carry roughly double the unscorability of the others, and the
sweep gives no monotone validity story.** Since penalized error moves only 0.008 across the sweep,
these validity swings largely cancel against the strict-error swings rather than driving a ranking.

**o_i is essentially invariant to τ:** mean −0.3713 / −0.3814 / −0.3898 / −0.3874 / −0.3834 across the
five arms, with near-identical ranges (≈ −0.60 to −0.09). A 0.019 spread in the mean. So whatever τ
changes, **it is not the offset** — which is the mechanical reason τ stops mattering once o_i is
present, and the sharpest form of the CAL-29 answer.

### count pooling is the only arm that did not reach round 100

Its dev selection landed at **round 60**, uniquely — the unsaturated weighting peaked early and then
degraded — and it is also the worst arm overall (penalized 0.190, in-support 0.179, worst-client
in-support 0.284). So the saturating weight earns its place: it exists to stop one dense client
dominating (`coverage.py:109`, "equal-client spirit"), and removing it costs both stability and
accuracy. **If any τ recommendation is made, it should be "keep saturation on; the value is not
critical", not a specific τ.**

**Cross-check.** The pool-weight shares above were computed two independent ways — derived from
`coverage.json` counts via `c/(τ+c)`, and read from each arm's round-100 snapshot `pool_weights` —
and agree to four decimals, so the table is not an artefact of either route.

**Caveats.** Single seed. All five arms share ccc0284, so none of the above is hardware-confounded —
unlike MOL-25b's aligned-vs-shared contrast.

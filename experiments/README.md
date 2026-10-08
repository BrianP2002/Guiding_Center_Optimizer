# Experiments: gradient estimators on the guiding-center flow

This directory holds the research code behind `docs/findings.md`: the scripts that were run (one script = one Slurm array task, or
one plain Python call), the Slurm array scripts, and the text outputs of the summary scripts for the stored results
(`reference_results/`). Nothing here is needed to use the library `nilss_jax`; it documents and reproduces the studies.

```
experiments/
  scripts/            one script per experiment (TASK_ID on the command line) and the summarize_*.py scripts that read their results
  slurm/              sbatch array scripts that call the scripts above; common.sh sets the environment of every job
  reference_results/  text output of the summarize_*.py scripts on the stored results (first line: the command that produced it)
```

## Running

Install the package once (`pip install -e .` from the repository root), then run everything **from the repository root**.
Every script also runs without Slurm: `python experiments/scripts/gc_official_fd.py 7 --par eps_t --h 0.01` computes task 7 and writes
`results/gc_official_fd/task_007.json`. Results go to `results/` and job logs to `logs/` (both are git-ignored).

For Slurm:

```bash
mkdir -p logs                                         # the --output directory has to exist when the job is submitted
export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>    # or pass --account / --partition to sbatch
export NILSS_PYTHON=/path/to/env/bin/python           # python of the environment with `pip install -e .` (default: python)
sbatch --array=0-119 experiments/slurm/gc_official_scan.sbatch
```

All jobs use one CPU (`JAX_PLATFORMS=cpu`, one thread). The array ranges below are the ones that were used; add `%N` to cap the
number of tasks that run at once. A task takes from seconds to about ten minutes (typical costs at the end).
`experiments/slurm/pytest.sbatch` runs the test suite on a compute node (`PYTEST_MARK="not slow"` and `PYTEST_FILES="..."` are optional).

## The studies, in order

The model is the vacuum Boozer guiding-center flow `nilss_jax.systems.guiding_center` (`rhs4`, J = s). A "draw" is a random parameter set of
`gc_official_scan.py`, identified by its seed (581, 835); the numbers of the studies for each draw are packaged in
`nilss_jax/systems/data/gc_draws.json` (`guiding_center.load_draw`).

**0. The teacher's model (`legacy/app_gc.py`).** Is it chaotic, and is `<x>` independent of the initial condition? (No: with `B` a function of one helical
angle the flow is integrable; a second helicity does not change that for the amplitudes tried.)

| script | slurm | results |
| --- | --- | --- |
| `gc_lyapunov.py` | `gc_lyapunov.sbatch` (`--array=0-65`) | `results/gc_lyapunov` |
| `gc_symmetry_breaking.py` | `gc_symmetry_breaking.sbatch` (`--array=0-23`) | `results/gc_symmetry_breaking` |
| `lorenz_ensemble.py` | `lorenz_ensemble.sbatch` (`--array=0-31`) | `results/lorenz_ensemble` (validation of the NILSS code on Lorenz 63) |
| `legacy/app_gc.py`, `legacy/app_gc_opt.py` | `cli_defaults.sbatch` (`--array=0-4`) | `results/cli/<parameter>` |

`gc_lyapunov.py` and `gc_symmetry_breaking.py` import `legacy/app_gc.py` (the model of the project); they add `legacy/` to `sys.path` themselves,
because `legacy/` is not a package.

**1. Where is the official flow chaotic and bounded?** A random scan of the field (helicity, amplitudes), of kappa, iota and the initial condition (position and pitch),
Lyapunov spectrum of the 4D flow over T = 4000 per draw.

```bash
sbatch --array=0-119 experiments/slurm/gc_official_scan.sbatch                 # results/gc_official_scan, 8 draws per task
python experiments/scripts/summarize_gc_official_scan.py results/gc_official_scan
```

**2. Are the chaotic draws ergodic?** The 9 chaotic draws are rerun for T = 2e5 from 16 initial conditions each (the long-time `<s>` must not depend on the initial
condition inside the sea).

```bash
sbatch --array=0-143 experiments/slurm/gc_official_candidates.sbatch           # results/gc_official_candidates
python experiments/scripts/summarize_gc_official_candidates.py results/gc_official_candidates
```

**3. Ground truth: finite differences of the long-time `<s>`.** Independent long orbits (copies) of the sea at parameter values `p0 + h (k - npoint // 2)`;
the same copies at every value; the summary keeps the copies that sample the sea (the standard deviation of `<s>` over 20 parts of the run is above a threshold).
Draw 581 (default `GC_SEED=581`):

```bash
sbatch --array=0-199 experiments/slurm/gc_official_fd.sbatch                                  # eps_t, 5 points, 40 copies -> results/gc_official_fd
FD_PAR=eps_t FD_H=0.0025 FD_NPOINT=13 FD_NCOPY=60 FD_OUT=results/gc_official_fd2_eps_t sbatch --array=0-779%60 experiments/slurm/gc_official_fd.sbatch
FD_PAR=kappa FD_H=0.01   FD_NPOINT=7  FD_NCOPY=60 FD_OUT=results/gc_official_fd2_kappa sbatch --array=0-419%60 experiments/slurm/gc_official_fd.sbatch
FD_PAR=iota  FD_H=0.02   FD_NPOINT=7  FD_NCOPY=60 FD_OUT=results/gc_official_fd2_iota  sbatch --array=0-419%60 experiments/slurm/gc_official_fd.sbatch
FD_PAR=eps_m FD_H=0.005  FD_NPOINT=7  FD_NCOPY=60 FD_OUT=results/gc_official_fd2_eps_m  sbatch --array=0-419%60 experiments/slurm/gc_official_fd.sbatch
python experiments/scripts/summarize_gc_official_fd.py results/gc_official_fd2_eps_t       # slopes of <s> in windows around the base value
```

**4. NILSS on the same orbits** (streaming, four parameters at once, `nus` = 1 and 2, segment lengths 50 / 200 / 800; 70 runs, T = 2e5 each):

```bash
sbatch --array=0-69%35 experiments/slurm/gc_official_nilss.sbatch                              # results/gc_official_nilss
python experiments/scripts/summarize_gc_official_nilss.py results/gc_official_nilss          # NILSS against the finite differences, tails of |v|, convergence
```

**5. A second regime (draw 835).** The sea initial conditions of the draw are found first (one task = one candidate, then collected into `ics.json`);
`GC_SEED=835` and `GC_ICS_ARGS` redirect the finite-difference and NILSS scripts to them.

```bash
R2_SEED=835 sbatch --array=0-95%48 experiments/slurm/gc_official_regime2_find_ics.sbatch
python experiments/scripts/gc_official_regime2_find_ics.py --collect --seed 835               # results/gc_official_regime2_seed835_ics/ics.json
ICS="--ics-file results/gc_official_regime2_seed835_ics/ics.json"
GC_SEED=835 GC_ICS_ARGS="$ICS" FD_PAR=eps_2 FD_H=0.001  FD_NPOINT=13 FD_NCOPY=60 FD_OUT=results/gc_official_fd_seed835_eps_2 sbatch --array=0-779%60 experiments/slurm/gc_official_fd.sbatch
GC_SEED=835 GC_ICS_ARGS="$ICS" FD_PAR=eps_h FD_H=0.0035 FD_NPOINT=9  FD_NCOPY=60 FD_OUT=results/gc_official_fd_seed835_eps_h sbatch --array=0-539%60 experiments/slurm/gc_official_fd.sbatch
GC_SEED=835 GC_ICS_ARGS="$ICS" FD_PAR=kappa FD_H=0.01   FD_NPOINT=9  FD_NCOPY=60 FD_OUT=results/gc_official_fd_seed835_kappa sbatch --array=0-539%60 experiments/slurm/gc_official_fd.sbatch
GC_SEED=835 GC_ICS_ARGS="$ICS" FD_PAR=iota  FD_H=0.01   FD_NPOINT=9  FD_NCOPY=60 FD_OUT=results/gc_official_fd_seed835_iota  sbatch --array=0-539%60 experiments/slurm/gc_official_fd.sbatch
GC_SEED=835 GC_ICS_ARGS="$ICS" NILSS_PARS=eps_2,eps_h,kappa,iota NILSS_LAYOUT=main_nus2 NILSS_NMAIN=48 NILSS_NNUS2=12 \
    NILSS_OUT=results/gc_official_nilss_seed835 sbatch --array=0-59%30 experiments/slurm/gc_official_nilss.sbatch
```

Check that the failure is not an artefact of the polar coordinates `(s, theta)` near the axis: the same runs in near-axis Cartesian coordinates
`X = sqrt(s) cos(theta)`, `Y = sqrt(s) sin(theta)` (`gc_official_regime2_nilss_cart.py`, same options and output format as `gc_official_nilss.py`):

```bash
NILSS_LAYOUT=main_nus2 NILSS_NMAIN=30 NILSS_NNUS2=6 NILSS_OUT=results/gc_official_nilss_cart_seed581 sbatch --array=0-35%36 experiments/slurm/gc_official_regime2_nilss_cart.sbatch
GC_SEED=835 GC_ICS_ARGS="$ICS" NILSS_PARS=eps_2,eps_h,kappa,iota NILSS_LAYOUT=main_nus2 NILSS_NMAIN=30 NILSS_NNUS2=6 \
    NILSS_OUT=results/gc_official_nilss_cart_seed835 sbatch --array=0-35%36 experiments/slurm/gc_official_regime2_nilss_cart.sbatch
python experiments/scripts/summarize_gc_official_nilss.py results/gc_official_nilss_seed835 --sea-lambda 0.001 --lambda-scan 0.0003,0.001,0.002,0.0027 \
    --fd-sea-rel 0.05 --fd eps_2=results/gc_official_fd_seed835_eps_2 --fd eps_h=results/gc_official_fd_seed835_eps_h \
    --fd kappa=results/gc_official_fd_seed835_kappa --fd iota=results/gc_official_fd_seed835_iota
```
(The finite-difference summaries of draw 835 use `summarize_gc_official_fd.py DIR --sea-rel 0.05`: the sea test is relative to `<s>`, which is 0.04 there.)

**6. Plain pathwise (forward-tangent) gradient of a finite-time ensemble objective** (`gc_pathwise.py`; ensemble E1: uniform `s`, both signs of `v_par`, E2: snapshots of the sea orbits).
The horizons 100 ... 30000 are read out of one integration; a task that hits the time limit leaves a `.part.npz` and continues from it when the same command is repeated.

```bash
PW_ARGS="--mode e2ics" sbatch experiments/slurm/gc_pathwise.sbatch                                            # E2_ics.npz
PW_ARGS="--mode pathwise --ens E1 --n 4000 --per-task 50" sbatch --array=0-79%60 experiments/slurm/gc_pathwise.sbatch
PW_ARGS="--mode pathwise --ens E2 --n 550  --per-task 50" sbatch --array=0-10    experiments/slurm/gc_pathwise.sbatch
PW_ARGS="--mode fd --ens E1 --n 4000 --per-task 25" sbatch --array=0-15%16 experiments/slurm/gc_pathwise.sbatch                 # finite differences of the sample mean
PW_ARGS="--mode pop --par eps_t --per-task 500 --h-rel 0.1 0.3 --times 100 300 1000 3000" sbatch --array=0-39%40 experiments/slurm/gc_pathwise.sbatch   # population finite differences (also --par kappa)
PW_ARGS="--mode pathwise --ens E1 --n 100 --per-task 100 --times 100 300 1000 --dt 0.005" PW_OUT=results/gc_pathwise_dt0005 sbatch --array=0 experiments/slurm/gc_pathwise.sbatch   # time-step check
python experiments/scripts/summarize_gc_pathwise.py results/gc_pathwise --ens E1 E2 --dt-check results/gc_pathwise_dt0005
```

**7. Pitch-angle collisions** (`gc_collisions.py` is the model; `gc_collision.py` the experiments): response curves of `E[g(s_T)]` against `eps_t` (independent paths) and the pathwise
derivative with common random numbers, for several collision frequencies `nu`.

```bash
sbatch --array=0-175%60 experiments/slurm/gc_collision_resp.sbatch                             # results/gc_collision_resp   (4 nu x 11 values of eps_t x 4 blocks of 1000 paths)
sbatch --array=0-31     experiments/slurm/gc_collision_crn.sbatch                              # results/gc_collision_crn    (4 nu x 8 blocks)
TS=50,100,150,200,250,300,400,500 NPER=500 NBLOCK=4 OUT=results/gc_collision_crn_fine sbatch --array=0-15 experiments/slurm/gc_collision_crn.sbatch
COUPLING=gradient TS=50,100,150,200,250,300,400,500 NPER=500 NBLOCK=4 OUT=results/gc_collision_crn_fine_gradient sbatch --array=0-15 experiments/slurm/gc_collision_crn.sbatch
PILOT=timing sbatch experiments/slurm/gc_collision_pilot.sbatch                                # sizing runs (PILOT=timing|dt|frac|lyap)
python experiments/scripts/summarize_gc_collision.py results/gc_collision_resp results/gc_collision_crn
```

**8. Control for the tails of the shadowing direction.** NILSS on Lorenz 63, where the norm of `v_perp` along the orbit is light-tailed:
`python experiments/scripts/lorenz_vnorm.py --T 4000 --rho 28` (prints median / 99 % / max of `|v_perp|` and the share of the largest per-segment contributions).

## Reference results

`reference_results/` contains the output of the summary scripts on the stored results of these studies (the raw `results/` are about 90 MB and are not in the
repository). Each file starts with the command that produced it.

| file | content |
| --- | --- |
| `scan_summary.txt`, `candidates_summary.txt` | where the official flow is chaotic and bounded; ergodicity of the candidate draws |
| `fd_draw581_*.txt`, `fd_draw835_*.txt` | finite-difference slopes of the long-time `<s>` (sea copies) in windows around the base value |
| `nilss_draw581.txt`, `nilss_draw835.txt` | NILSS against the finite differences; tail index of `\|v_perp\|`; convergence with the run length; `nus` = 1 against 2 |
| `nilss_cartesian_draw581.txt`, `nilss_cartesian_draw835.txt` | the same in near-axis Cartesian coordinates |
| `pathwise_summary.txt` | pathwise gradient against horizon, sample and population finite differences, time-step check |
| `collision_summary.txt`, `collision_crn_*.txt` | collisions: response curves, pathwise estimator, growth rates |
| `teacher_model_*.txt`, `lorenz_ensemble.txt` | the teacher's model (Lyapunov exponents, symmetry breaking) and the Lorenz 63 validation |

## Cost and reproducibility

Typical cost of one task (one CPU, on the cluster of the studies): finite-difference copy 25 s; scan task 1 min; candidate 6 min; NILSS run 7 min; Cartesian NILSS run 10 min;
collision response task 6 min; pathwise tasks 5 min to 1 h (the longest horizons).

The scripts are deterministic on a CPU. After the move into the package, one task of every script family was rerun and compared with the stored result: the
scan, candidate, finite-difference, sea-initial-condition, Lorenz, collision (response and common-random-number), pathwise (sample and population finite
differences, horizons 100 - 1000), Lyapunov, symmetry-breaking and legacy CLI outputs are **bit-identical**. NILSS runs with one homogeneous tangent
(`nus = 1`) have a bit-identical orbit (`<s>`) and agree with the stored sensitivities to 2e-11 .. 2e-10 (per-segment contributions to 1e-8, rarely 1e-6): the
tangent equations are now evaluated with Jacobian-vector products instead of the full Jacobian, which changes the rounding. Runs with `nus = 2` do **not**
reproduce beyond about 1e-3 (first segments already differ by 1e-5 .. 1e-3 between the two evaluations, and the second exponent by 0.5 %), because the second
tangent direction of this flow is numerically sensitive; the conclusions of the studies do not rest on digits of single `nus = 2` runs.

# nilss-jax

[![CI](https://github.com/BrianP2002/Guiding_Center_Optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/BrianP2002/Guiding_Center_Optimizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Non-intrusive least squares shadowing (NILSS) in JAX**: the sensitivity of long-time averages of chaotic flows to many
parameters at once, with memory that does not grow with the length of the trajectory, **plus a reliability report that tells you
when the answer should not be trusted**.

The repository name is historical. The tool grew out of applying NILSS to the guiding-center (GC) motion of charged particles in
stellarator fields; the GC equations are bundled as an application and as a stress test, and what that study found is part of the
package (see [Can I trust the result?](#can-i-trust-the-result)).

## Features

- **Streaming.** The tangent integrals of the NILSS least-squares problem are accumulated while integrating. Memory grows with the
  number of segments (a few numbers each), not with the number of time steps; a run of 2 x 10^5 time units needs a few MB.
- **Many parameters, one pass.** The homogeneous tangents are shared; every extra parameter costs one more inhomogeneous tangent.
- **Automatic differentiation.** Write the vector field and the objective with `jax.numpy`; the tangent equations use
  Jacobian-vector products, so there are no hand-derived Jacobians and the cost per step grows with the number of tangents, not
  with the dimension of the state.
- **Hamiltonian flows.** An optional conserved quantity (the energy) puts the tangents on the perturbed invariant surface.
- **Ensembles with robust statistics.** Mean and standard error, median with a bootstrap interval, trimmed mean; worker processes
  or one Slurm task per orbit.
- **A reliability report.** Tail index of the shadowing direction, error scaling, ergodicity, resolved exponent, consistency in
  the number of tangents. It flags the failure signatures that were found in real flows (below).
- **A finite-difference reference** (`nilss_jax.fd`) to check any sensitivity you rely on, and an optimiser wrapper
  (`nilss_jax.optimize`) with NILSS or finite-difference gradients.
- **Bundled systems.** Lorenz 63 (validation) and the vacuum Boozer guiding-center flow of SIMSOPT / FIRM3D (`gc_vac`), with two
  chaotic parameter sets found by a random scan.

## Install

```bash
git clone https://github.com/BrianP2002/Guiding_Center_Optimizer
cd Guiding_Center_Optimizer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
nilss-jax selftest        # Lorenz 63 against the published values, a few seconds
```

Python 3.10 or newer. JAX on the CPU is enough; double precision is switched on when the package is imported.

## Quickstart

```python
import numpy as np
from nilss_jax import NILSS, run_ensemble
from nilss_jax.systems import lorenz63

# one run, three parameters at once
nilss = NILSS(lorenz63.rhs, lorenz63.J, params=("rho", "sigma", "beta"), dt=0.005, T_seg=0.5, nus=1)
res = nilss.run(lorenz63.DEFAULTS, lorenz63.initial_condition(0), T=200.0, T_spinup=20.0)
print(res.summary())

# an ensemble of orbits: robust statistics and the reliability report
u0s = np.array([lorenz63.initial_condition(i) for i in range(8)])
ens = run_ensemble(nilss, lorenz63.DEFAULTS, u0s, T=200.0, T_spinup=20.0)
print(ens.summary())
print(ens.report())
```

`d<z>/d(rho)` comes out near 1.02 and the leading exponent near 0.90, the values of Ni & Wang for the same system (1.0 and 0.9056).
The report says `CAUTION`, and it is right to: the `sigma` sensitivity (+0.133) is about 11 % below the finite-difference value
(+0.150 +- 0.004), even on Lorenz 63 (see below and [docs/reliability.md](docs/reliability.md)).

## Your own system

```python
import jax.numpy as jnp

def rhs(u, p):                       # du/dt; p is a dict of scalars, u a state vector
    return jnp.array([...])

def J(u):                            # instantaneous objective, a scalar function of the state
    return u[0] ** 2

nilss = NILSS(rhs, J, params=("a", "b"), dt=0.01, T_seg=1.0, nus=1)
ens = run_ensemble(nilss, {"a": 1.0, "b": 2.0, "c": 0.5}, u0s, T=500.0, T_spinup=50.0)
```

`nus` must be at least the number of positive Lyapunov exponents (check with `nilss_jax.lyapunov.lyapunov_spectrum`), and
`T` should be many times `1/lambda_1`. [docs/usage.md](docs/usage.md) explains the choices (`dt`, `T_seg`, `nus`, `T`, spin-up),
how to define the objective, and how to cross-check against finite differences; [examples/](examples/) has runnable scripts
(Lorenz 96 with 8 variables, the guiding-center flow, optimisation, a Slurm ensemble).

## Can I trust the result?

NILSS is derived for uniformly hyperbolic systems. Lorenz 63 at rho = 28 is the benchmark where it works: `d<z>/d(rho)` = 1.02 and
`lambda_1` = 0.89 - 0.90 from 8 orbits, against 1.0 and 0.9056 in the literature; the `rho` and `beta` sensitivities are within 1.5 % of finite
differences and the exact identities of the invariant measure (for example `d<dz/dt>/dp = 0`) hold to 1e-4. Even there the `sigma`
sensitivity is about 11 % off (+0.133 against +0.150 +- 0.004 from finite differences, stable against `T`, `nus`, `dt` and `T_seg`), and the report flags it. Lorenz 96 with
8 variables and the Roessler attractor are reported `UNRELIABLE` (tail index about 1.2). Most systems of interest are not uniformly hyperbolic, and then one run can look
converged and be wrong.

The guiding-center flow of a stellarator-like field is a Hamiltonian flow. With a single helical symmetry it is integrable;
breaking the symmetry gives a mixed phase space with a thin bounded chaotic sea (about 1 % of random parameter draws).
NILSS and finite differences over many orbits (T = 2 x 10^5 per run, the same orbits) disagree there:

| regime | parameter | NILSS (mean +- SEM of the runs) | finite differences |
| --- | --- | --- | --- |
| draw 581 (35 runs) | `eps_t` | -0.41 +- 0.23 | +1.75 +- 0.1 |
| | `eps_m` | -2.13 +- 0.60 | +1.8 ... +3.5 |
| | `iota` | +0.268 +- 0.020 | -0.05 +- 0.03 |
| | `kappa` | +1.00 +- 0.05 | +0.4 ... +0.7 |
| draw 835 (48 runs) | `eps_2` | +0.94 +- 0.32 | +0.80 +- 0.05 |
| | `iota` | +0.074 +- 0.041 | +0.11 +- 0.005 |
| | `eps_h` | -0.57 +- 0.02 | -0.86 +- 0.03 |
| | `kappa` | +0.22 +- 0.03 | +0.01 +- 0.006 |

In both regimes the norm of the shadowing direction has a power-law tail, P(|v| > x) ~ 1/x (Hill index 1.0), so the
per-segment contributions have infinite variance: the error bar from the run-to-run scatter means nothing and the estimate converges
far more slowly than 1/sqrt(T). Some parameters agree with finite differences (draw 835, `eps_2` and `iota`), others do not, and a
tight spread says nothing about bias. `ens.report()` detects these signatures and returns `UNRELIABLE` for both regimes; it is meant
as an alarm, not a certificate: an `OK` verdict means that no known failure signature is present, and `UNRELIABLE` means that the error bar cannot be trusted,
not that the value is wrong (in draw 835 two of four agree). Details and the calibration of the
thresholds are in [docs/reliability.md](docs/reliability.md), the whole study in [docs/findings.md](docs/findings.md).

Practical advice: use an ensemble of orbits, read the report, and check the sensitivities you depend on with `nilss_jax.fd`.

## Command line

```bash
nilss-jax selftest                                  # installation check
nilss-jax lorenz --par rho sigma --runs 8 --T 200   # ensemble + report on Lorenz 63
nilss-jax gc --draw 835 --runs 8 --T 20000          # the chaotic guiding-center sea (expect UNRELIABLE)
```

## Documentation

| | |
| --- | --- |
| [docs/usage.md](docs/usage.md) | API tour, choosing `dt`, `T_seg`, `nus`, `T`; common errors |
| [docs/theory.md](docs/theory.md) | NILSS, the streaming form, multi-parameter sharing, the energy-surface constraint |
| [docs/reliability.md](docs/reliability.md) | what each check measures, calibration, limits |
| [docs/findings.md](docs/findings.md) | the guiding-center studies: NILSS against finite differences, pathwise gradients, collisions |
| [docs/hpc.md](docs/hpc.md) | worker processes, Slurm arrays, costs |
| [experiments/README.md](experiments/README.md) | the scripts and job files that produced the studies, and their stored summaries |

## Repository layout

```
src/nilss_jax/      core.py (NILSS, NILSSResult), ensemble.py, diagnostics.py, fd.py, optimize.py, lyapunov.py, cli.py,
                    reference.py (dense one-parameter implementation used to validate core.py), systems/ (Lorenz 63, guiding center)
examples/           runnable scripts
docs/               documentation
experiments/        scripts, Slurm job files and stored summaries of the guiding-center studies
legacy/             the original one-parameter scripts (a guiding-center model and an L-BFGS-B driver)
tests/              pytest; `pytest -m "not slow"` is the quick suite
```

## Tests

```bash
pytest -m "not slow"        # about 3 minutes on one core
pytest                      # everything
```

The tests check, among other things, that the streaming code reproduces the dense reference implementation to 1e-10 on Lorenz 63,
the guiding-center flow and an 8-dimensional Lorenz 96 system; that the shadowing direction it reports is the displacement of the
orbit of the perturbed system to first order; and that the tangent equations match finite differences for every parameter of the
guiding-center flow.

## References

- A. Ni and Q. Wang, *Sensitivity analysis on chaotic dynamical systems by Non-Intrusive Least Squares Shadowing (NILSS)*,
  J. Comput. Phys. 347 (2017), [arXiv:1611.00880](https://arxiv.org/abs/1611.00880).
- A. Ni, Q. Wang, P. Fernandez and C. Talnikar, *Sensitivity analysis on chaotic dynamical systems by Finite Difference
  Non-Intrusive Least Squares Shadowing (FD-NILSS)*, J. Comput. Phys. 394 (2019), [arXiv:1711.06633](https://arxiv.org/abs/1711.06633)
  (the Schur-complement solve used here).
- A. Ni and C. Talnikar, *Adjoint sensitivity analysis on chaotic dynamical systems by Non-Intrusive Least Squares Adjoint
  Shadowing (NILSAS)*, J. Comput. Phys. 395 (2019), [arXiv:1801.08674](https://arxiv.org/abs/1801.08674).
- The guiding-center equations: the `gc_vac` mode of [SIMSOPT](https://simsopt.readthedocs.io) and FIRM3D (Littlejohn Lagrangian in
  Boozer coordinates).

## Acknowledgements

NILSS is due to Angxiu Ni and Qiqi Wang. The first version of this repository adapted the public Python implementation that accompanies
the NILSS paper (https://github.com/niangxiu/nilss, which carries no license file). The code in this release was written afresh:
`nilss_jax.core` is a new streaming JAX implementation, and `nilss_jax.reference` is a dense implementation rewritten from the equations of
the paper, structured differently (all steps in memory, the full KKT system solved with a dense solver) and used only to cross-check
`core`. The earlier adaptation remains in the git history of this repository.

## Citing

See [CITATION.cff](CITATION.cff). If you use the diagnostics or the guiding-center results, cite the repository and the NILSS paper.

## License

MIT, see [LICENSE](LICENSE).

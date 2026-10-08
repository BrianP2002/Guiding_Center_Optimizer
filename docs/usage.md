# Using nilss-jax

A tour of the API. Every `python` block below is executed, in order, by `tests/test_examples.py`, so the snippets are
true of the code. The runnable programs are in [`examples/`](../examples); the method itself is in [`theory.md`](theory.md),
the question of when a result can be believed in [`reliability.md`](reliability.md), clusters in [`hpc.md`](hpc.md).

## 1. Install and check

```bash
pip install -e .            # from the repository root; needs Python >= 3.10, JAX, NumPy, SciPy
nilss-jax selftest          # Lorenz 63 against the published values, a few seconds
```

The package switches JAX to double precision when it is imported (NILSS needs it: the tangent solutions grow by many orders of
magnitude between re-orthonormalisations).

## 2. Define a problem and run it

A problem is a flow `rhs(u, p)`, an objective `J(u)`, a dict `p` of parameter values and the names of the parameters to
differentiate. `rhs` and `J` are written with `jax.numpy`: they are traced and differentiated automatically (there is no
Jacobian to code). `J` is a function of the state only; the parameters differentiated must be scalars.

```python
import jax.numpy as jnp
import numpy as np
from nilss_jax import NILSS

def rhs(u, p):                                   # du/dt; p is a dict of floats
    x, y, z = u
    return jnp.array([p['sigma'] * (y - x), x * (p['rho'] - z) - y, x * y - p['beta'] * z])

def J(u):                                        # instantaneous objective: the long-time mean of z is differentiated
    return u[2]

p = {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0 / 3.0}
u0 = np.array([1.0, 1.0, 20.0])

nilss = NILSS(rhs, J, params=('rho', 'beta'), dt=0.005, T_seg=0.5, nus=1)    # compiled on the first run
res = nilss.run(p, u0, T=40.0, T_spinup=20.0)
print(res.summary())
print(res.J, res.dJdp, res.lyapunov)
```

`NILSS(rhs, J, params, dt, T_seg, nus=1, invariant=None)`:

| argument | meaning |
| --- | --- |
| `params` | names (keys of `p`) of the parameters differentiated. They share the homogeneous tangents, so more parameters cost little. |
| `dt` | the Runge-Kutta 4 step, the same for the orbit and the tangents. |
| `T_seg` | length of one segment, rounded to a multiple of `dt` (a warning says so if it is not). See section 4. |
| `nus` | number of homogeneous tangents; at least the number of positive Lyapunov exponents. |
| `invariant` | `E(u, p)` conserved by the flow for every `p`; see section 8. |

`nilss.run(p, u0, T, *, T_spinup=0.0, w0=None, seed=0, verbose=False)` integrates `T_spinup` first (to reach the attractor,
nothing is accumulated) and then follows the orbit for `T`. The initial homogeneous tangents are random unit vectors from `seed`
unless `w0` (shape `[nus, n]`) is given. The first call compiles (some seconds); calling `run` again on the same object, with other
parameter values or other orbits, does not recompile. `run_segments(p, u0, nseg, nseg_ps, ...)` is the lower-level entry that works
in numbers of segments and returns `(J, dJdp, info)`.

## 3. The result

```python
print(res.nseg, res.T, res.dJdp_array, res.lyapunov_time_product)
print(res.segments.shape, res.vnorm.shape, res.J_segments.shape, res.coeffs.shape)
res.save('one_run.npz')
from nilss_jax import NILSSResult
print(NILSSResult.load('one_run.npz'))
```

| field | content |
| --- | --- |
| `J` | long-time average of the objective along the orbit |
| `dJdp` | sensitivities `d<J>/dp`, a dict by parameter name (`dJdp_array`: the same in the order of `params`) |
| `lyapunov` | the `nus` exponents of the homogeneous tangents (projected perpendicular to the flow); `lyapunov_time_product` is `max(lyapunov) * T` |
| `T`, `nus`, `T_seg`, `dt`, `params`, `seed` | what was run |
| `segments` | `[nseg, npar]`, contribution of each segment to `dJdp`; `segments.sum(axis=0)` is the sensitivity |
| `vnorm` | `[nseg, npar]`, norm of the shadowing direction (perpendicular to the flow) at the end of each segment |
| `J_segments` | `[nseg]`, mean of `J` over each segment |
| `coeffs` | `[nseg, nus, npar]`, the coefficients of the homogeneous tangents |
| `finite` | `False` if the integration produced non-finite numbers (the numeric fields are then NaN) |

`segments` and `vnorm` are what the reliability diagnostics are computed from.

## 4. Choosing dt, T_seg, nus, T and T_spinup

* **`dt`**: small enough that RK4 resolves the fastest time scale of the flow; the tangent equations are integrated with the same
  scheme. Halve it once: the sensitivities of an ensemble should move by much less than their standard error. (The guiding-center
  studies used 0.01 and the energy stayed conserved to better than 1e-6 over T = 2e5; the Lorenz examples use 0.005.)
* **`T_seg`**: segments only set how often the tangents are re-orthonormalised, so the sensitivity must not depend on them. Ni & Wang
  (arXiv:1611.00880, sec. 4.2) advise a segment no longer than about `1 / (lambda_1 - lambda_nus)`: longer segments make the
  covariance matrices ill-conditioned, shorter ones make the least-squares problem larger. In the guiding-center studies
  (`1/lambda_1` about 170) `T_seg` = 50, 200 and 800 gave the same sensitivity on identical orbits to 1.5e-4 (50 against 200) and 1.4e-2 (800
  against 200) in absolute terms, for sensitivities of order 1. The examples use 0.5 for Lorenz 63 and 200 for the guiding-center flow.
* **`nus`**: at least the number of positive exponents, which you can measure first with `nilss_jax.lyapunov` (section 7); the paper
  (sec. 4.2) raises it until the next exponent is negative. `nus = 2` against `nus = 1` on the same orbits is a consistency check (`report(nus_check=...)`): in a
  uniformly hyperbolic system the answer must not change; where it does (4 of 10 and 4 of 12 orbits in the guiding-center studies) the
  hyperbolicity assumption is in doubt.
* **`T`**: the orbit must resolve a positive exponent, `lambda_1 * T` of at least 20 (the default `min_lyap_time` of the ensemble
  functions; `summary()` warns below 3). A longer `T` helps only as long as the error of one run shrinks like `1/sqrt(T)`; where it
  does not (heavy tails), more orbits are better than longer ones, and the report says which case you are in.
* **`T_spinup`**: long enough to reach the attractor, a few times `1/lambda_1` or more. The guiding-center studies used 6000
  (about 36 / lambda_1).

## 5. Many orbits: ensembles

One run follows one orbit. Run several from different initial conditions on the same attractor and look at the statistics.

```python
from nilss_jax import run_ensemble

u0s = np.array([[1.0, 1.0, 20.0] + 0.5 * np.random.RandomState(i).randn(3) for i in range(8)])
ens = run_ensemble(nilss, p, u0s, T=40.0, T_spinup=20.0)       # workers=4 runs the orbits in 4 processes
print(ens.summary())
print(ens.mean(), ens.sem())
print(ens.median(), ens.median_interval())                    # bootstrap interval of the median over the runs
print(ens.trimmed_mean(0.1))
ens.save('ensemble_dir')
from nilss_jax import load_ensemble
print(load_ensemble('ensemble_dir'))
```

`run_ensemble(nilss, p, u0s, T, *, T_spinup=0.0, seeds=None, min_lyap_time=20.0, workers=1, executor=None, verbose=False, save_dir=None)`
returns an `EnsembleResult`. Runs whose `lambda_1 * T` is below `min_lyap_time` (no resolved positive exponent), or that are not
finite, are kept in `ens.results` but left out of the statistics (`ens.kept`, `ens.used`). `ens.values()` is the array of
sensitivities `[n_used, npar]`; `mean()`, `sem()`, `median()`, `median_interval()` and `trimmed_mean()` are per parameter. The mean is
what NILSS defines; the median and the trimmed mean are robust to the outliers of heavy tails but are not the same quantity when the
tails carry part of the answer. `workers > 1` needs `rhs`, `J` and `invariant` to be top-level functions of an importable
module (see [`hpc.md`](hpc.md)).

## 6. The reliability report

```python
rep = ens.report()
print(rep)
print(rep.verdict, rep.ok)
print(sorted(rep.to_dict()))
```

`ens.report(nus_check=other_ensemble)` (or `nilss_jax.reliability_report(runs)` on a list of `NILSSResult`) runs the checks of
`nilss_jax.diagnostics` and gives a verdict `ok`, `caution` or `unreliable`:

| check | flags |
| --- | --- |
| number of runs | fewer than 8 |
| resolved exponent | median `lambda_1 T` below 20 (caution) or 3 (unreliable) |
| exponent spread | runs with `lambda_1` below 0.3 x the median (probably another component, e.g. a regular torus) |
| ergodicity | spread of `<J>` over the runs above 3 x the sampling error of one run |
| tail of the norm of the shadowing direction, of the segment contributions | Hill tail index below 4 (caution) or 2 (unreliable: infinite variance, the standard errors mean nothing) |
| error scaling | the spread of the estimate falls more slowly than `L^-0.35` (caution) or `L^-0.2` (unreliable) with the averaging length `L` (0.5 expected) |
| nus consistency | orbits whose estimate moves by more than 3 robust sigma with one more tangent |

`ok` means that no known failure signature is present, not that the answer is right. How to read the checks, and the
numbers they were calibrated on, are in [`reliability.md`](reliability.md).

## 7. Before NILSS: the Lyapunov spectrum, and after: finite differences

`nilss_jax.lyapunov.lyapunov_spectrum(f, u0, dt, T_spinup, T, args=(...))` computes the spectrum of the flow `f(u, *args)` by the
QR method, independently of the NILSS code. NILSS needs a positive exponent; count the ones clearly above zero (the dict also has
the finite-time estimates `lyap_finite_time` at four checkpoints to judge the scatter); the neutral flow direction shows up as an exponent
near 0.

```python
from nilss_jax import lyapunov

def flow(u, rho):                                              # f(u, *args): the parameter comes as an argument
    return rhs(u, {**p, 'rho': rho})

spec = lyapunov.lyapunov_spectrum(flow, u0, dt=0.005, T_spinup=20.0, T=60.0, args=(28.0,))
print(spec['lyap'])
```

The check that does not depend on any shadowing assumption is the finite difference of the long-time average over many orbits.
`nilss_jax.fd.finite_difference` needs no tangent equations; its cost is `npoint * ncopy * (T + T_spinup) / dt` RK4 steps.

```python
from nilss_jax import fd

res_fd = fd.finite_difference(rhs, J, p, 'rho', h=1.0, u0s=u0s, T=100.0, dt=0.01, npoint=3, T_spinup=20.0)
print(res_fd.summary())
print(fd.compare(ens, {'rho': res_fd}))
```

The same initial conditions (displaced once by `jitter`) are used at every parameter value. `project(u, p)` puts them back on a
constraint surface at each value (the energy shell of a Hamiltonian flow depends on the parameters); `select(parts)` keeps only some
copies at each value (for instance those on the chaotic sea: `lambda parts: parts.std(axis=1, ddof=1) > 0.05 * np.abs(parts.mean(axis=1))`).
`compare` prints the NILSS mean against the slope and the difference in combined standard errors, which is only as meaningful as the NILSS
standard error is (see the verdict).

**What agreement means.** Both numbers carry systematic errors that their statistical error bars do not include: the finite-difference slope
is a secant slope over a window of `+-2h` (curvature of `<J>(p)` biases it; shrink `h` until it stops moving, as long as the noise allows),
and one NILSS run is a finite-time estimate of an infinite-time quantity. On Lorenz 63 at rho = 28, 16 NILSS orbits of T = 1000 give 1.017,
0.134 and -1.658 for rho, sigma and beta (the same to the last digit for `dt` = 0.01, 0.005, 0.0025, for `T_seg` = 0.25 to 1 and for `nus` = 1, 2);
finite differences over 256 copies of T = 1000 give 1.004 +- 0.001 (`h` = 0.5) for rho, -1.647 +- 0.004 (`h` = 0.125) for beta, and for sigma
0.167, 0.154, 0.149 and 0.144 (+- 0.004) for `h` = 1, 0.5, 0.25, 0.125, still moving toward the NILSS value. So expect agreement at the level of
one to two percent of the slope for rho and beta and about ten percent for sigma, whose sensitivity is small, and not "within the error bars":
the statistical errors of both methods are far smaller than that. The report flags the sigma sensitivity of Lorenz 63 as CAUTION (the
per-segment contributions have a Hill index of about 2.5), which is the parameter with the largest disagreement.

## 8. Hamiltonian flows: the invariant

For a flow with a conserved quantity `E(u, p)` for every `p` (the energy), pass `invariant=E`. The initial tangents are then put on the
perturbed energy surface and stay there; without it the conserved quantity is a second neutral direction and the shadowing least-squares
problem has the wrong structure. The bundled guiding-center system does this for you:

```python
from nilss_jax.systems import guiding_center as gc

draw = gc.load_draw(835)                       # parameters, sea initial conditions and the reference results of the studies
nilss_gc = gc.make_nilss(('eps_2',), dt=0.01, T_seg=200.0, nus=1)
# the same as NILSS(gc.rhs4, gc.mean_radius, ('eps_2',), 0.01, 200.0, 1, invariant=gc.energy)
u0_gc = gc.project_to_energy_shell(draw['sea_ics'][0], draw['params'])
res_gc = nilss_gc.run(draw['params'], u0_gc, T=400.0, T_spinup=200.0)
print(res_gc.summary())
```

On the chaotic seas of this flow the verdict is `unreliable`: see `examples/04_guiding_center_sea.py` and
[`findings.md`](findings.md).

## 9. Optimisation

`nilss_jax.optimize.minimize` wraps `scipy.optimize.minimize` (L-BFGS-B). At every evaluation it runs the same ensemble (same initial
conditions and seeds) and uses `<J>` and its sensitivities; `target=` minimises `(<J> - target)^2`, `maximize=True` maximises `<J>`.

```python
from nilss_jax import optimize

opt = optimize.minimize(nilss, p, free=['rho'], u0s=u0s[:2], T=30.0, T_spinup=10.0,
                        bounds={'rho': (26.0, 40.0)}, target=30.5, maxiter=3, on_unreliable='ignore')
print(opt.x, opt.J, len(opt.history), opt.message)
```

`gradient='fd'` (with `fd_h`) uses central finite differences of the ensemble average instead of NILSS, at about
`2 * len(free) + 1` times the cost of one `<J>`; use it where the verdict is `unreliable`. `on_unreliable` is `'warn'` (once),
`'raise'` or `'ignore'`. The finite-time average of a chaotic orbit is a rough function of the parameter (about +-0.05 in `<z>` for Lorenz 63
at T = 100), so L-BFGS-B often ends with a line-search failure (`ABNORMAL`) at that noise floor: the result is then as good as the
ensemble allows.

## 10. Command line

```bash
nilss-jax --version
nilss-jax selftest                                  # Lorenz 63 against the published values; exit code 1 on failure
nilss-jax lorenz --par rho sigma --T 200 --runs 8   # ensemble on Lorenz 63: statistics and reliability report
nilss-jax gc --draw 835 --par eps_2 iota --runs 4   # ensemble on a chaotic sea of the guiding-center flow (expect UNRELIABLE)
```

`nilss-jax gc` options: `--draw {581,835}`, `--par`, `--T` (default 20000), `--T-spinup`, `--T-seg`, `--dt`, `--runs`, `--workers`.
`python -m nilss_jax ...` is the same.

## 11. Common errors and warnings

| message | meaning and remedy |
| --- | --- |
| `NILSS needs double precision: set jax.config.update("jax_enable_x64", True) ...` | something switched JAX to float32 after the import of `nilss_jax`; enable x64 before any JAX computation. |
| `params must be a non-empty list of distinct parameter names` | `params` is empty, has repeats or has non-strings. |
| `need 0 < dt <= T_seg` / `nus must be at least 1` | bad constructor arguments. |
| warning `T_seg = ... is not a multiple of dt = ...; using n steps per segment` | `T_seg` was rounded to `n * dt`. |
| `parameters [...] are not in p` / `parameter '...' must be a scalar` | `p` lacks a name given in `params`, or its value is an array. |
| `u0 must be a finite 1-D state vector` | NaN or wrong shape in the initial condition. |
| `rhs(u, p) has shape ..., expected the shape of the state ...` / `J(u) must return a scalar` | `rhs` must return an array shaped like `u`, `J` a scalar. |
| `nus = k must be smaller than the dimension of the state` | the flow direction is projected out, so at most `n - 1` tangents exist. |
| `T = ... is shorter than one segment` | `T` below `T_seg`. |
| `WARNING: non-finite numbers in the integration` (`res.finite` is False) | the orbit left the domain or `dt` is too large for this parameter value. |
| `WARNING: lambda_1 * T < 3` | no positive exponent resolved: `T` too short or the system is not chaotic. |
| `... none usable (lambda_1 T < 20 or non-finite)` | every run was left out of the ensemble statistics; lengthen `T` (or lower `min_lyap_time` knowingly). |
| `need one seed per initial condition` | `seeds` and `u0s` have different lengths. |
| `workers > 1 needs rhs, J and invariant to be top-level functions of an importable module` | a lambda or closure cannot be sent to a worker process; define the functions at the top level of a module. |
| `... is too short for 20 blocks` / `fewer than 2 copies kept` (`fd`) | lengthen `T`, or loosen `select`. |
| `free parameters [...] are not differentiated by this NILSS object` / `gradient='fd' needs fd_h` / `no usable NILSS run at ...` (`optimize`) | see section 9. |
| warning `the NILSS reliability verdict is UNRELIABLE ...` (`optimize`) | use `gradient='fd'`, or accept the warning knowingly. |

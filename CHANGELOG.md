# Changelog

## 0.1.0 - 2026-10-07

First public release of `nilss_jax`.

### Added
- `nilss_jax.NILSS`: streaming, multi-parameter NILSS (Ni & Wang, arXiv:1611.00880; Schur-complement solve of arXiv:1711.06633) in JAX.
  The flow and the objective are ordinary `jax.numpy` functions; the tangent equations are Jacobian-vector products
  (`jax.linearize`), so no Jacobian is coded or formed. Memory does not depend on the number of time steps; any number of parameters share
  the homogeneous tangents; an optional conserved quantity (`invariant=`, the energy of a Hamiltonian flow) puts the tangents on the
  perturbed invariant surface. `NILSSResult` holds the sensitivities, the Lyapunov exponents, the per-segment contributions and the norm of the
  shadowing direction, and can be saved and loaded (`.npz`).
- `nilss_jax.run_ensemble` / `EnsembleResult` / `load_ensemble`: many orbits, serial, in worker processes, or from cluster tasks; mean and
  standard error, median with bootstrap interval, trimmed mean.
- `nilss_jax.reliability_report` (`ensemble.report()`): checks for the failure signatures of shadowing (resolved exponent, ergodicity,
  exponent spread, Hill tail index of the shadowing direction and of the segment contributions, scaling of the error with the averaging length,
  consistency in `nus`) and a verdict `ok` / `caution` / `unreliable`; `hill_index`, `tail_index`, `convergence_exponent`.
- `nilss_jax.fd`: finite-difference reference for a long-time average over many orbits (`time_average`, `finite_difference`, `compare`).
- `nilss_jax.optimize.minimize`: L-BFGS-B on an ensemble average with NILSS or finite-difference gradients.
- `nilss_jax.lyapunov`: Lyapunov spectrum by the QR method, independent of the NILSS code.
- `nilss_jax.reference`: a dense one-parameter NumPy NILSS rewritten from the equations of the paper (all steps in memory, the full KKT system solved
  densely), kept as an independent implementation that the tests compare the streaming code against to 1e-10. It replaces the earlier
  `nilss.py`, which adapted the unlicensed public code of the NILSS paper; see the acknowledgements in the README.
- `nilss_jax.systems`: Lorenz 63 (the validation problem) and the vacuum Boozer-coordinate guiding-center flow of SIMSOPT / FIRM3D
  (`guiding_center`: 4D and 3D forms, the energy invariant, two bounded chaotic regimes found by a random scan with their reference results).
- Command line `nilss-jax` (`selftest`, `lorenz`, `gc`) and `python -m nilss_jax`.
- `examples/` (six runnable examples including a cluster template), `docs/` (usage, theory, HPC, reliability, findings).
- `experiments/`: the scripts, Slurm templates and reference outputs of the guiding-center studies (random scan, chaotic candidates,
  finite-difference ground truth, NILSS runs, pathwise-gradient and collision prototypes). `legacy/`: the original `app_gc.py` and
  `app_gc_opt.py` (the teacher's model and its one-parameter optimiser), now using `nilss_jax.reference`.

### Known limitations
- **NILSS is not reliable on the guiding-center flow with a mixed phase space.** On the chaotic seas of two scan draws (T = 2e5, 35 and 48 orbits)
  the ensemble sensitivities agreed with finite differences for 0 of 4 parameters (draw 581) and 2 of 4 (draw 835), the shadowing direction has
  a power-law tail with Hill index about 1 and the estimate converges much more slowly than `T^-1/2`. The reliability report flags both as
  `unreliable`; check any such sensitivity against finite differences (`docs/findings.md`).
- The thresholds of the reliability report are heuristics calibrated on these cases and Lorenz 63. Lorenz 63 passes every check for the rho
  sensitivity; the sigma sensitivity is flagged `caution`, and NILSS is about 10% from finite differences for it.
- The finite-time average of a chaotic orbit is a rough function of the parameters, which limits gradient-based optimisation to the noise
  floor of the ensemble.
- Per-segment records are kept for the whole run (about 30 KB per segment measured for tiny segments), so very long runs with very short
  segments use more memory than their number of steps suggests.
- CPU only; Python >= 3.10; double precision (enabled on import).

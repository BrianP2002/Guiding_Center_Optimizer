"""nilss_jax: non-intrusive least squares shadowing (NILSS) in JAX, with diagnostics for when it cannot be trusted.

>>> from nilss_jax import NILSS, run_ensemble
>>> from nilss_jax.systems import lorenz63
>>> nilss = NILSS(lorenz63.rhs, lorenz63.J, ('rho',), dt=0.005, T_seg=0.5)
>>> res = nilss.run(lorenz63.DEFAULTS, lorenz63.initial_condition(0), T=100.0, T_spinup=20.0)   # doctest: +SKIP
"""
from .core import NILSS, NILSSResult, NILSSStreamer, solve_shadowing_coeffs_multi
from .diagnostics import ReliabilityReport, hill_index, reliability_report, tail_index
from .ensemble import EnsembleResult, load_ensemble, run_ensemble
from . import fd, lyapunov

__version__ = '0.1.0'

__all__ = ['NILSS', 'NILSSResult', 'NILSSStreamer', 'solve_shadowing_coeffs_multi', 'EnsembleResult', 'run_ensemble', 'load_ensemble',
           'ReliabilityReport', 'reliability_report', 'hill_index', 'tail_index', 'fd', 'lyapunov', '__version__']

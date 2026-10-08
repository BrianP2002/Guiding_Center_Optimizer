import os
import numpy as np
from scipy.optimize import minimize
from nilss_jax.reference import nilss
from app_gc import make_problem, canon_par, add_nilss_args, PARAM_BOUNDS
import argparse

def optimize_guiding_center_nilss(par_name, par_bounds, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, maxiter = 100, tol = 1e-6):
    history = [] # (parameter value, J, dJ/dpar) of every evaluation

    def objective(par_value):
        J, dJdpar = nilss(
            dt=dt,
            nseg=nseg,
            T_seg=T_seg,
            nseg_ps=nseg_ps,
            u0=u0,
            nus=nus,
            par=par_name,
            s=par_value,
            integrator=integrator,
            fJJu=fJJu
        )
        history.append((float(par_value), float(J), float(dJdpar)))
        return J, dJdpar

    def scipy_objective(par_value):
        J, dJdpar = objective(par_value[0])
        return J, np.array([dJdpar])

    result = minimize(
        fun=scipy_objective,
        x0=[(par_bounds[0] + par_bounds[1]) / 2],
        jac=True,
        method='L-BFGS-B',
        bounds=[par_bounds],
        options={
            'maxiter': maxiter,
            'ftol': tol,
            'disp': True
        }
    )
    result.history = history

    return result


def main():
    parser = add_nilss_args(argparse.ArgumentParser(description='Optimize a guiding center parameter with NILSS gradients.'))
    parser.add_argument('--seed', type=int, default=20250402, help='seed of the initial condition and the random tangents')
    parser.add_argument('--maxiter', type=int, default=100)
    args = parser.parse_args()
    par = canon_par(args.par)

    np.random.seed(args.seed)
    x = 0.1 * np.random.rand()
    y = 2.0 * np.pi * np.random.rand()
    z = 2.0 * np.pi * np.random.rand()
    u0 = np.array([x, y, z])
    integrator, fJJu = make_problem(args.dt)

    result = optimize_guiding_center_nilss(par, PARAM_BOUNDS[par], u0, args.nus, args.dt, args.nseg, args.T_seg,
                                           args.nseg_ps, integrator, fJJu, maxiter=args.maxiter)
    os.makedirs(args.outdir, exist_ok=True)
    text_result_path = os.path.join(args.outdir, f"optimization_guiding_center_results_{par}.txt")
    with open(text_result_path, "w") as f:
        f.write("Optimization Result:\n")
        f.write(f"  Optimal {par}: {result.x[0]:.4f}\n")
        f.write(f"  Minimum cost J: {result.fun:.4e}\n")
        f.write(f"  Evaluations: {len(result.history)}\n")
        f.write(f"  {par}  J  dJ/d{par}  (every evaluation)\n")
        for p, J, g in result.history:
            f.write(f"  {p:.6f}  {J:.6e}  {g:.6e}\n")


if __name__ == '__main__':
    main()

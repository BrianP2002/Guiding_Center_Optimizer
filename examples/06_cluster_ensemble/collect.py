"""Collect the run_XXXX.npz files of a cluster ensemble: statistics and reliability report.

    python examples/06_cluster_ensemble/collect.py runs
"""
import argparse

from nilss_jax import load_ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', nargs='?', default='runs')
    ap.add_argument('--min-lyap-time', type=float, default=20.0, help='runs with lambda_1 * T below this are left out')
    args = ap.parse_args()
    ens = load_ensemble(args.outdir, min_lyap_time=args.min_lyap_time)
    print(ens.summary())
    print()
    print(ens.report())


if __name__ == '__main__':
    main()

import numpy as np

from args import get_args
from simulator_loop import SimulatorLoop

if __name__ == "__main__":
    """Init"""
    args = get_args()
    np.random.seed(args.seed)

    """Simulator loop"""
    if args.uncertain or args.detumbling:
        if args.uncertain and args.detumbling:
            raise ValueError("Only one of --uncertain and --detumbling can be specified")

        # set log_dir
        log_dir = args.log_dir if args.log_dir is not "default" else ("detumbling" if args.detumbling else "uncertain")

        SimulatorLoop.monte_carlo(args.num_runs, args.num_step, args.tsim, args.tsample, args.f_st, args.omega_norm,
                                  log_dir, args.uncertain, args.detumbling, args.nominal_only, args.eci, args.ecef,
                                  args.fixed_init, args.tumble_off)
    elif args.coarse:
        SimulatorLoop.coarse(args.num_step, args.tsim, args.tsample, args.f_st)
    elif args.fine:
        SimulatorLoop.fine(args.num_step, args.tsim, args.tsample, args.f_st)
    elif args.core:

        # set log_dir
        log_dir = args.log_dir if args.log_dir is not "default" else "core"

        SimulatorLoop.core(args.num_step, args.tsim, args.tsample, f_st=args.f_st, omega_norm_deg=args.omega_norm,
                           log_dir=log_dir, nominal_only=args.nominal_only, use_eci=args.eci, use_ecef=args.ecef,
                           fixed_init=args.fixed_init, tumble_off=args.tumble_off)

import argparse


def get_args():
    """
    Function for handling command line arguments
    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='C3S - CubeSat Simulation Suite\n Attitude Determination and Control')

    # mode selection
    parser.add_argument('--core', action='store_true', default=False,
                        help='run core simulation')
    parser.add_argument('--uncertain', action='store_true', default=False,
                        help='use uncertain inertia matrix')
    parser.add_argument('--detumbling', action='store_true', default=False,
                        help='use uncertain inertia matrix')
    parser.add_argument('--coarse', action='store_true', default=False,
                        help='run coarse sweep')
    parser.add_argument('--fine', action='store_true', default=False,
                        help='run fine sweep')
    parser.add_argument('--nominal-only', action='store_true', default=False,
                        help='only SIA control')
    parser.add_argument('--eci', action='store_true', default=False,
                        help='attitude target is given in the ECI frame')
    parser.add_argument('--ecef', action='store_true', default=False,
                        help='attitude target is given in the ECEF frame')
    parser.add_argument('--fixed-init', action='store_true', default=True,
                        help='use hardwired initial values')
    parser.add_argument('--tumble-off', action='store_true', default=False,
                        help='turns off initial tumbling')


    # simualtion parameters
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')
    parser.add_argument('--num-runs', type=int, default=10, metavar='NUM_RUNS',
                        help='number of runs (for random variants)')
    parser.add_argument('--tsim', type=int, default=250, metavar='TIMESTEP',
                        help='timestep of the simulation in ms')
    parser.add_argument('--tsample', type=int, default=250, metavar='T_SAMPLE',
                        help='sample time of the ADCS in ms')
    parser.add_argument('--f-st', type=int, default=800, metavar='F_ST',
                        help='frequency of updating the attitude with star tracker values [steps]')
    parser.add_argument('--omega-norm', type=float, default=80, metavar='OMEGA_NORM',
                        help='the norm of the initial angular velocity vector in DEGREES (valid only if --detumbling is specified')
    parser.add_argument('--num-step', type=int, default=50000, metavar='NUM_STEPS',
                        help='number of simulation steps')
    parser.add_argument('--log-dir', type=str, default='default',
                        help='logging directory')

    # model parameters
    parser.add_argument('--h-ref', type=float, default=.007, metavar='H_REF',
                        help='reference angular momentum of th RWs')


    # star metric parameters
    parser.add_argument('--num-experiments', type=int, default=2000, metavar='NUM_EXPERIMENTS',
                        help='number of experiments')
    parser.add_argument('--num-stars', type=int, default=10, metavar='NUM_STARS',
                        help='number of stars')
    parser.add_argument('--mag-noise-std', type=float, default=0, metavar='MAG_NOISE_STD',
                        help='standard deviation of the magnitude noise')

    # Argument parsing
    return parser.parse_args()

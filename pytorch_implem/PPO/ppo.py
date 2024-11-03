import argparse

import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default = os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default = 2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='total timesteps of the experiments')
    parser.add_argument()

#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import gym
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env_name', type=str)
    parser.add_argument('--render', dest='render', type=bool, default=1)
    parser.add_argument('--train', dest='train', type=int, default=1)
    # parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--type', dest='type', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9)
    parser.add_argument('--e_start', dest='e_start', type=float, default=0.5)
    parser.add_argument('--e_end', dest='e_end', type=float, default=0.05)
    parser.add_argument('--e_step', dest='e_step', type=float, default=1e-5)
    parser.add_argument('--replace_iter', dest='replace_iter', type=int, default=200)
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=50000)
    parser.add_argument('--burn_in', dest='burn_in', type=int, default=10000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--n_episode', dest='n_episode', type=int, default=100)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    from IPython import embed
    embed()

if __name__ == '__main__':
    main(sys.argv)


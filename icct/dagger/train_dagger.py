# Created by Yaru Niu

import gym
import os
import numpy as np
import argparse
from icct.dagger.dt_policy import DTPolicy
from icct.dagger.dagger import DAgger

from icct.rl_helpers.sac import SAC
import highway_env
from flow.utils.registry import make_create_env
from icct.sumo_envs.accel_ring import ring_accel_params
from icct.sumo_envs.accel_ring_multilane import ring_accel_lc_params
from icct.sumo_envs.accel_figure8 import fig8_params
from stable_baselines3.common.utils import set_random_seed

def make_env(env_name, seed):
    set_random_seed(seed)
    if env_name == 'lunar':
        env = gym.make('LunarLanderContinuous-v2')
        name = 'LunarLanderContinuous-v2'
    elif env_name == 'cart':
        env = gym.make('InvertedPendulum-v2')
        name = 'InvertedPendulum-v2'
    elif env_name == 'lane_keeping':
        env = gym.make('lane-keeping-v0')
        name = 'lane-keeping-v0'
    elif env_name == 'ring_accel':
        create_env, gym_name = make_create_env(params=ring_accel_params, version=0)
        env = create_env()  
        name = gym_name
    elif env_name == 'ring_lane_changing':
        create_env, gym_name = make_create_env(params=ring_accel_lc_params, version=0)
        env = create_env()  
        name = gym_name
    elif env_name == 'figure8':
        create_env, gym_name = make_create_env(params=fig8_params, version=0)
        env = create_env()  
        name = gym_name 
    else:
        raise Exception('No valid environment selected')
    env.seed(seed)
    return env, name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Decision Trees with DAgger')
    parser.add_argument('--env_name', help='environment to run on', type=str, default='lunar')
    parser.add_argument('--max_depth', help='the maximum depth of the decision tree', type=int, default=5)
    parser.add_argument('--n_rollouts', help='number of rollouts in a training batch', type=int, default=10)
    parser.add_argument('--iterations', help='maximum number of training iterations', type=int, default=80)
    parser.add_argument('--eval_episodes', help='number of episodes to evaluate the trained decision tree', type=int, default=20)
    parser.add_argument('--oracle_load_path', help='the path of loading the oracle model', type=str, default='saved_mlp_models')
    parser.add_argument('--oracle_load_file', help='which oracle model file to load', type=str, default='best_model')
    parser.add_argument('--seed', help='random seed', type=int, default=42)
    parser.add_argument('--render', help='if render the tested environment', action='store_true')
    parser.add_argument('--gpu', help='if run on a GPU (depending on the loaded file)', action='store_true')
    parser.add_argument('--save', help='the path to save the decision tree model', type=str, default='saved_dt_models')
    parser.add_argument('--load', help='the path to load the decision tree model', type=str, default=None)

    args = parser.parse_args()
    env, env_n = make_env(args.env_name, args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_dir = args.save + "/" + "best_dt.pkl"
    
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    
    oracle_model = SAC.load(args.oracle_load_path + "/" + args.oracle_load_file, device=args.device)
    oracle_model.set_random_seed(args.seed)
    dt_model = DTPolicy(env.action_space, args.max_depth)
    dagger = DAgger(env, oracle_model, dt_model, args.n_rollouts, args.iterations)
    if args.load:
        dagger.load_best_dt(args.load)
    else:
        dagger.train(save_dir)
    dagger.evaluate(args.eval_episodes)


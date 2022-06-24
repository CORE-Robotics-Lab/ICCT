# Created by Yaru Niu

import gym
import numpy as np
import copy
import argparse
import random
import os
import torch
from icct.rl_helpers import ddt_policy
from icct.core.icct_helpers import convert_to_crisp
from icct.rl_helpers.save_after_ep_callback import EpCheckPointCallback
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)

from stable_baselines3 import SAC
import highway_env
from flow.utils.registry import make_create_env
from icct.sumo_envs.accel_ring import ring_accel_params
from icct.sumo_envs.accel_figure8 import fig8_params
from icct.sumo_envs.accel_ring_multilane import ring_accel_lc_params
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
    parser = argparse.ArgumentParser(description='ICCT Testing')
    parser.add_argument('--env_name', help='environment to run on', type=str, default='lunar')
    parser.add_argument('--seed', help='random seed', type=int, default=42)
    parser.add_argument('--load_path', help='the path of saving the model', type=str, default='test')
    parser.add_argument('--num_episodes', help='number of episodes to test', type=int, default=20)
    parser.add_argument('--render', help='if render the tested environment', action='store_true')
    parser.add_argument('--gpu', help='if run on a GPU', action='store_true')
    parser.add_argument('--load_file', help='which model file to load and test', type=str, default='best_model')
    

    args = parser.parse_args()
    env, env_n = make_env(args.env_name, args.seed)
    
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    
    model = SAC.load("../../" + args.load_path + "/" + args.load_file, device=args.device)
    obs = env.reset()
    episode_reward_for_reg = []
    for _ in range(args.num_episodes):
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward+= reward
            if args.render:
                env.render()
            if done:
                obs = env.reset()
                episode_reward_for_reg.append(episode_reward)
                break
    print('fuzzy results:')
    print(episode_reward_for_reg)
    print(np.mean(episode_reward_for_reg))
    print(np.std(episode_reward_for_reg))

    env, env_n = make_env(args.env_name, args.seed)
    if hasattr(model.actor, 'ddt'):
        model.actor.ddt = convert_to_crisp(model.actor.ddt, training_data=None)
        obs = env.reset()
        discrete_episode_reward_for_reg = []
        for _ in range(args.num_episodes):
            done = False
            episode_reward = 0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                if args.render:
                    env.render()
                if done:
                    obs = env.reset()
                    discrete_episode_reward_for_reg.append(episode_reward)
                    break
        print('crisp results:')
        print(discrete_episode_reward_for_reg)
        print(np.mean(discrete_episode_reward_for_reg))
        print(np.std(discrete_episode_reward_for_reg))
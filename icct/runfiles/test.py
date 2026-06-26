import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import numpy as np
import gym
import gym.wrappers
import torch
import highway_env

from stable_baselines3.common.utils import set_random_seed
from icct.rl_helpers.sac import SAC
from icct.core.icct_helpers import convert_to_crisp


def make_env(env_name, seed):
    set_random_seed(seed)
    if env_name == 'cart':
        env = gym.make('InvertedPendulum-v2')
    elif env_name == 'lunar':
        env = gym.make('LunarLanderContinuous-v2')
    elif env_name == 'lane_keeping':
        env = gym.wrappers.TimeLimit(gym.make('lane-keeping-v0').unwrapped, max_episode_steps=500)
    elif env_name == 'ring_accel':
        from icct.sumo_envs.accel_ring import ring_accel_params
        from flow.utils.registry import make_create_env
        create_env, _ = make_create_env(params=ring_accel_params, version=0)
        env = create_env()
    elif env_name == 'ring_lane_changing':
        from icct.sumo_envs.accel_ring_multilane import ring_accel_lc_params
        from flow.utils.registry import make_create_env
        create_env, _ = make_create_env(params=ring_accel_lc_params, version=0)
        env = create_env()
    elif env_name == 'figure8':
        from icct.sumo_envs.accel_figure8 import fig8_params
        from flow.utils.registry import make_create_env
        create_env, _ = make_create_env(params=fig8_params, version=0)
        env = create_env()
    else:
        raise ValueError(f'Unknown environment: {env_name}')
    env.seed(seed)
    return env


def evaluate(model, env, n_episodes, deterministic=True):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    return np.array(rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--load_file', type=str, default='best_model')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--nn', action='store_true', help='skip crisp conversion (evaluate fuzzy/NN directly)')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(args.load_path, args.load_file)
    model = SAC.load(model_path, device=device)

    env = make_env(args.env_name, args.seed)

    print(f"fuzzy results:")
    print()
    rewards = evaluate(model, env, args.num_episodes)
    print(f"  {rewards.mean():.4f}")
    print(f"  {rewards.std():.4f}")

    if not args.nn and hasattr(model.actor, 'ddt'):
        model.actor.ddt = convert_to_crisp(model.actor.ddt, training_data=None)
        print(f"crisp results:")
        print()
        c_rewards = evaluate(model, env, args.num_episodes)
        print(f"  {c_rewards.mean():.4f}")
        print(f"  {c_rewards.std():.4f}")

    env.close()

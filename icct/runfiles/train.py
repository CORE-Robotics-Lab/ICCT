# Created by Yaru Niu

import gym
import numpy as np
import copy
import argparse
import random
import os
import torch
from icct.rl_helpers import ddt_sac_policy
from icct.rl_helpers import ddt_td3_policy
from icct.core.icct_helpers import convert_to_crisp
from icct.rl_helpers.save_after_ep_callback import EpCheckPointCallback
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)

from icct.rl_helpers.sac import SAC
from icct.rl_helpers.td3 import TD3
import highway_env
from flow.utils.registry import make_create_env
from icct.sumo_envs.accel_ring import ring_accel_params
from icct.sumo_envs.accel_ring_multilane import ring_accel_lc_params
from icct.sumo_envs.accel_figure8 import fig8_params
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor


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
    parser = argparse.ArgumentParser(description='ICCT Training')
    parser.add_argument('--env_name', help='environment to run on', type=str, default='lunar')
    parser.add_argument('--alg_type', help='sac or td3', type=str, default='sac')
    parser.add_argument('--policy_type', help='mlp or ddt', type=str, default='ddt')
    parser.add_argument('--mlp_size', help='the size of mlp (small|medium|large)', type=str, default='medium')
    parser.add_argument('--seed', help='the seed number to use', type=int, default=42)
    parser.add_argument('--num_leaves', help='number of leaves used in ddt (2^n)', type=int, default=16)
    parser.add_argument('--submodels', help='if use sub-models in ddt', action='store_true', default=False)
    parser.add_argument('--sparse_submodel_type', help='the type of the sparse submodel, 1 for L1 regularization, 2 for feature selection, other values for not sparse', type=int, default=0)
    parser.add_argument('--hard_node', help='if use differentiable crispification', action='store_true', default=False)
    parser.add_argument('--gpu', help='if run on a GPU', action='store_true', default=False)
    parser.add_argument('--lr', help='learning rate', type=float, default=3e-4)
    parser.add_argument('--buffer_size', help='buffer size', type=int, default=1000000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--gamma', help='the discount factor', type=float, default=0.9999)
    parser.add_argument('--tau', help='the soft update coefficient (between 0 and 1)', type=float, default=0.01)
    parser.add_argument('--learning_starts', help='how many steps of the model to collect transitions for before learning starts', type=int, default=10000)
    parser.add_argument('--training_steps', help='total steps for training the model', type=int, default=500000)
    parser.add_argument('--argmax_tau', help='the temperature of the diff_argmax function', type=float, default=1.0)
    parser.add_argument('--ddt_lr', help='the learning rate of the ddt', type=float, default=3e-4)
    parser.add_argument('--use_individual_alpha', help='if use different alphas for different nodes', action='store_true', default=False)
    parser.add_argument('--l1_reg_coeff', help='the coefficient of the l1 regularization when using l1-reg submodels', type=float, default=5e-3)
    parser.add_argument('--l1_reg_bias', help='if consider biases in the l1 loss when using l1-reg submodels', action='store_true', default=False)
    parser.add_argument('--l1_hard_attn', help='if only sample one linear controller to perform L1 regularization for each update when using l1-reg submodels', action='store_true', default=False)
    parser.add_argument('--num_sub_features', help='the number of chosen features for submodels', type=int, default=1)
    parser.add_argument('--use_gumbel_softmax', help='if use gumble softmax instead of the differentiable argmax proposed in the paper', action='store_true', default=False)
    # evaluation and model saving
    parser.add_argument('--min_reward', help='minimum reward to save the model', type=int)
    parser.add_argument('--save_path', help='the path of saving the model', type=str, default='test')
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluation frequence of the model', type=int, default=1500)
    parser.add_argument('--log_interval', help='the number of episodes before logging', type=int, default=4)

    
    args = parser.parse_args()
    env, env_n = make_env(args.env_name, args.seed)
    eval_env = gym.make(env_n)
    eval_env.seed(args.seed)
    save_folder = args.save_path
    log_dir = '../../' + save_folder + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if args.policy_type == 'ddt':
        if not args.submodels and not args.hard_node:
            method = 'm1'
        elif args.submodels and not args.hard_node:
            method = 'm2'
            if args.sparse_submodel_type == 1 or args.sparse_submodel_type == 2:
                raise Exception('Not a method we want to test')
        elif not args.submodels and args.hard_node:
            method = 'm3'    
        else:
            if args.sparse_submodel_type != 1 and args.sparse_submodel_type != 2:
                method = 'm4'
            elif args.sparse_submodel_type == 1:
                method = 'm5a'
            else:
                method = f'm5b_{args.num_sub_features}'
    elif args.policy_type == 'mlp':
        if args.mlp_size == 'small':
            method = 'mlp_s'
        elif args.mlp_size == 'medium':
            method = 'mlp_m'
        elif args.mlp_size == 'large':
            method = 'mlp_l'
        else:
            raise Exception('Not a valid MLP size')
    else:
        raise Exception('Not a valid policy type')
    
    monitor_file_path = log_dir + method + f'_seed{args.seed}'
    env = Monitor(env, monitor_file_path)
    eval_monitor_file_path = log_dir + 'eval_' + method + f'_seed{args.seed}'
    eval_env = Monitor(eval_env, eval_monitor_file_path)
    callback = EpCheckPointCallback(eval_env=eval_env, best_model_save_path=log_dir, n_eval_episodes=args.n_eval_episodes,
                                    eval_freq=args.eval_freq, minimum_reward=args.min_reward)
    
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        
    if args.env_name == 'lane_keeping':
        features_extractor = CombinedExtractor
    else:
        features_extractor = FlattenExtractor

    if args.env_name == 'cart':
        args.fs_submodel_version = 1
    else:
        args.fs_submodel_version = 0
    
    if args.alg_type != 'sac' and args.alg_type != 'td3':
        raise Exception('Not a valid RL algorithm type')

    if args.policy_type == 'ddt':
        ddt_kwargs = {
            'num_leaves': args.num_leaves,
            'submodels': args.submodels,
            'hard_node': args.hard_node,
            'device': args.device,
            'argmax_tau': args.argmax_tau,
            'ddt_lr': args.ddt_lr,
            'use_individual_alpha': args.use_individual_alpha,
            'sparse_submodel_type': args.sparse_submodel_type,
            'fs_submodel_version': args.fs_submodel_version,
            'l1_reg_coeff': args.l1_reg_coeff,
            'l1_reg_bias': args.l1_reg_bias,
            'l1_hard_attn': args.l1_hard_attn,
            'num_sub_features': args.num_sub_features,
            'use_gumbel_softmax': args.use_gumbel_softmax,
            'alg_type': args.alg_type
        }
        policy_kwargs = {
            'features_extractor_class': features_extractor,
            'ddt_kwargs': ddt_kwargs
        }
        if args.alg_type == 'sac':
            policy_name = 'DDT_SACPolicy'
            policy_kwargs['net_arch'] = {'pi': [16, 16], 'qf': [256, 256]} # [256, 256] if a default setting in sb3 for SAC
        else:
            policy_name = 'DDT_TD3Policy'
            policy_kwargs['net_arch'] = {'pi': [16, 16], 'qf': [400, 300]} # [400, 300] if a default setting in sb3 for TD3

    elif args.policy_type == 'mlp':
        if args.env_name == 'lane_keeping':
            policy_name = 'MultiInputPolicy'
        else:
            policy_name = 'MlpPolicy'
            
        if args.mlp_size == 'small':
            if args.env_name == 'cart':
                pi_size = [6, 6]
            elif args.env_name == 'lunar':
                pi_size = [6, 6]
            elif args.env_name == 'lane_keeping':
                pi_size = [6, 6]
            elif args.env_name == 'ring_accel':
                pi_size = [3, 3]
            elif args.env_name == 'ring_lane_changing':
                pi_size = [3, 3] 
            else:
                pi_size = [3, 3] 
        elif args.mlp_size == 'medium':
            if args.env_name == 'cart':
                pi_size = [8, 8]
            elif args.env_name == 'lunar': 
                pi_size = [10, 10]
            elif args.env_name == 'lane_keeping':
                pi_size = [14, 14]
            elif args.env_name == 'ring_accel':
                pi_size = [12, 12]
            elif args.env_name == 'ring_lane_changing':
                pi_size = [32, 32] 
            else:
                pi_size = [20, 20] 
        elif args.mlp_size == 'large':
            if args.alg_type == 'sac':
                pi_size = [256, 256]
            else:
                pi_size = [400, 300]
        else:
            raise Exception('Not a valid MLP size')
        if args.alg_type == 'sac':
            policy_kwargs = {
                'net_arch': {'pi': pi_size, 'qf': [256, 256]},
                'features_extractor_class': features_extractor,
            }
        else:
            policy_kwargs = {
                'net_arch': {'pi': pi_size, 'qf': [400, 300]},
                'features_extractor_class': features_extractor,
            }
    else:
        raise Exception('Not a valid policy type')
        
    if args.alg_type == 'sac':
        model = SAC(policy_name, env,
                    learning_rate=args.lr,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    ent_coef='auto',
                    train_freq=1,
                    gradient_steps=1,
                    gamma=args.gamma,
                    tau=args.tau,
                    learning_starts=args.learning_starts,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_dir,
                    verbose=1,
                    device=args.device,
                    seed=args.seed)
    else:
        model = TD3(policy_name, env,
                    learning_rate=args.lr,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    tau=args.tau,
                    learning_starts=args.learning_starts,
                    target_policy_noise=0.1,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_dir,
                    verbose=1,
                    device=args.device,
                    seed=args.seed)
    
    model.learn(total_timesteps=args.training_steps, log_interval=args.log_interval, callback=callback)


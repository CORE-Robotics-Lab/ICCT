import os   
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gym
import numpy as np
import argparse
import csv
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from icct.rl_helpers.sac import SAC
from icct.core.oblique_tree import ObliqueTree
import highway_env

from flow.utils.registry import make_create_env


def make_env(env_name, seed):
    set_random_seed(seed)
    if env_name == 'lunar':
        env = gym.make('LunarLanderContinuous-v2')
        name = 'LunarLanderContinuous-v2'
    elif env_name == 'cart':
        env = gym.make('InvertedPendulum-v2')
        name = 'InvertedPendulum-v2'
    elif env_name == 'lane_keeping':
        env = gym.wrappers.TimeLimit(gym.make('lane-keeping-v0').unwrapped, max_episode_steps=500)
        name = 'lane-keeping-v0'
    elif env_name == 'ring_accel':
        from icct.sumo_envs.accel_ring import ring_accel_params
        create_env, gym_name = make_create_env(params=ring_accel_params, version=0)
        env = create_env()
        name = gym_name
    elif env_name == 'ring_lane_changing':
        from icct.sumo_envs.accel_ring_multilane import ring_accel_lc_params
        create_env, gym_name = make_create_env(params=ring_accel_lc_params, version=0)
        env = create_env()
        name = gym_name
    elif env_name == 'figure8':
        from icct.sumo_envs.accel_figure8 import fig8_params
        create_env, gym_name = make_create_env(params=fig8_params, version=0)
        env = create_env()
        name = gym_name
    else:
        raise Exception(f'No valid environment selected: {env_name}')
    env.seed(seed)
    return env, name


def collect_observations(env, oracle, n_rollouts):
    """
    Roll out the oracle deterministically and collect the processed observations
    the actor sees (via get_sa_pair), matching the DAgger observation format.
    """
    all_obs = []
    for _ in range(n_rollouts):
        obs = env.reset()
        done = False
        while not done:
            action, _ = oracle.predict(obs, deterministic=True)
            processed_obs, _ = oracle.actor.get_sa_pair()
            all_obs.append(processed_obs.cpu())
            obs, _, done, _ = env.step(action)
    return torch.cat(all_obs, dim=0)


def evaluate_tree(tree, env, oracle, n_episodes, device):
    """
    Evaluate the distilled tree by running it in the environment.
    Uses oracle.actor.get_sa_pair() to get processed obs, same as DAgger evaluate.

    Returns list of per-episode rewards.
    """
    tree.eval()
    episode_rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Run oracle forward pass to get processed observation
            oracle.predict(obs, deterministic=True)
            processed_obs, _ = oracle.actor.get_sa_pair()

            obs_t = processed_obs.to(device)
            with torch.no_grad():
                if tree.alg_type == 'sac':
                    mus, _ = tree(obs_t)
                else:
                    mus = tree(obs_t)

            # Apply tanh to squash pre-tanh mus to [-1,1], then scale to action space
            # (tree learns to match oracle's pre-tanh mu, so tanh is needed at inference)
            action = torch.tanh(mus).cpu().numpy().squeeze()
            low, high = env.action_space.low, env.action_space.high
            action = low + 0.5 * (action + 1.0) * (high - low)
            action = np.clip(action, low, high)

            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)

    return episode_rewards


def write_eval_csv(csv_path, epoch, mean_reward, std_reward):
    """Write a row to the eval CSV. Format matches learning_curve_plot.py expectation"""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'mean_reward', 'std_reward'])
        writer.writerow([epoch, mean_reward, std_reward])


def write_loss_csv(csv_path, loss_log):
    """Write distillation loss log to CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        writer.writerows(loss_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distil oracle SAC into ObliqueTree')
    parser.add_argument('--env_name', type=str, default='lunar')
    parser.add_argument('--oracle_load_path', type=str, default='saved_mlp_models')
    parser.add_argument('--oracle_load_file', type=str, default='best_model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='test',
                        help='folder to save model and CSV logs (matches --save_path in train.py)')
    # rollout collection
    parser.add_argument('--n_rollouts', type=int, default=500,
                        help='number of oracle rollouts to collect observations from')
    # tree
    parser.add_argument('--num_leaves', type=int, default=16)
    parser.add_argument('--use_individual_alpha', action='store_true', default=False)
    # distillation
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lam', type=float, default=0.1,
                        help='leaf usage penalty weight (Hinton eq. 4)')
    parser.add_argument('--log_every', type=int, default=20)
    # evaluation (matches train.py convention)
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                        help='episodes per evaluation checkpoint (matches train.py)')
    parser.add_argument('--eval_freq', type=int, default=20,
                        help='evaluate every N distillation epochs')
    parser.add_argument('--load', type=str, default=None,
                        help='path to a saved best_model.pt to load and evaluate, skipping training')
    parser.add_argument('--eval_episodes', type=int, default=20,
                        help='episodes to evaluate when using --load')

    args = parser.parse_args()
    device = 'cuda' if args.gpu else 'cpu'

    eval_env, _ = make_env(args.env_name, args.seed)

    # Load oracle model
    oracle = SAC.load(
        os.path.join(args.oracle_load_path, args.oracle_load_file),
        device=device
    )
    oracle.set_random_seed(args.seed)
    print(f'Loaded oracle from {args.oracle_load_path}/{args.oracle_load_file}')

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        tree = ObliqueTree(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            num_leaves=checkpoint['num_leaves'],
            use_individual_alpha=args.use_individual_alpha,
            device=device,
            alg_type='sac',
        )
        tree.load_state_dict(checkpoint['state_dict'])
        tree.to(device)
        print(f'Loaded distilled tree from {args.load}')
        rewards = evaluate_tree(tree, eval_env, oracle, args.eval_episodes, device)
        print(f'Evaluation over {args.eval_episodes} episodes: '
              f'{np.mean(rewards):.2f} +/- {np.std(rewards):.2f}')
        sys.exit(0)

    env, env_n = make_env(args.env_name, args.seed)

    # Set up log dir matching train.py layout: save_path/method_seedN
    method = 'distill_oblique'
    log_dir = args.save_path + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Wrap envs with Monitor so reward CSVs are written automatically,
    # matching the format that learning_curve_plot.py reads
    monitor_path = log_dir + method + f'_seed{args.seed}'
    eval_monitor_path = log_dir + 'eval_' + method + f'_seed{args.seed}'
    env = Monitor(env, monitor_path)
    eval_env = Monitor(eval_env, eval_monitor_path)

    print(f'Collecting observations from {args.n_rollouts} oracle rollouts...')
    observations = collect_observations(env, oracle, args.n_rollouts)
    print(f'Collected {observations.shape[0]} observations of dim {observations.shape[1]}')

    # Build student tree
    input_dim  = observations.shape[1]
    output_dim = eval_env.action_space.shape[0]

    tree = ObliqueTree(
        input_dim=input_dim,
        output_dim=output_dim,
        num_leaves=args.num_leaves,
        use_individual_alpha=args.use_individual_alpha,
        device=device,
        alg_type='sac',
    )

    # CSV paths — written alongside Monitor CSVs so learning_curve_plot.py can find them
    eval_csv_path = log_dir + 'eval_' + method + f'_seed{args.seed}_distil_eval.csv'
    loss_csv_path = log_dir + method + f'_seed{args.seed}_distil_loss.csv'

    # Distil epoch by epoch so we can evaluate periodically
    print(f'\nDistilling into ObliqueTree ({args.num_leaves} leaves, depth {tree.depth})...\n')

    best_mean_reward = -np.inf
    best_model_path  = log_dir + 'best_model.pt'
    full_loss_log    = []

    # Single optimizer shared across all chunks — preserves Adam momentum state
    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr)

    # Run distillation in eval_freq-sized chunks
    epochs_done = 0
    while epochs_done < args.epochs:
        chunk = min(args.eval_freq, args.epochs - epochs_done)

        loss_log = tree.train_distil(
            teacher=oracle.policy.actor,
            observations=observations,
            epochs=chunk,
            batch_size=args.batch_size,
            lr=args.lr,
            lam=args.lam,
            log_every=args.log_every,
            optimizer=optimizer,
        )
        # Offset epoch numbers to be global
        loss_log = [(e + epochs_done, l) for e, l in loss_log]
        full_loss_log.extend(loss_log)
        epochs_done += chunk

        # Evaluate
        rewards = evaluate_tree(tree, eval_env, oracle, args.n_eval_episodes, device)
        mean_r  = float(np.mean(rewards))
        std_r   = float(np.std(rewards))
        print(f'[eval] epoch {epochs_done:4d} | mean reward {mean_r:.2f} ± {std_r:.2f}')
        write_eval_csv(eval_csv_path, epochs_done, mean_r, std_r)

        if mean_r >= best_mean_reward:
            best_mean_reward = mean_r
            torch.save({
                'state_dict': tree.state_dict(),
                'input_dim':  input_dim,
                'output_dim': output_dim,
                'num_leaves': args.num_leaves,
            }, best_model_path)
            print(f'  -> saved best model (reward {mean_r:.2f})')

    # Write full loss log
    write_loss_csv(loss_csv_path, full_loss_log)
    print(f'\nDone. Best reward: {best_mean_reward:.2f}')
    print(f'Eval CSV : {eval_csv_path}')
    print(f'Loss CSV : {loss_csv_path}')
    print(f'Best model: {best_model_path}')

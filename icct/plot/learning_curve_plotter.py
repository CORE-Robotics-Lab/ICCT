# Created by Yaru Niu

from typing import Callable, List, Optional, Tuple

import csv
import json
import os
import time
from glob import glob
from typing import Dict, List, Optional, Tuple, Union
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import pandas as pd


class Learning_Curve_Plotter(object):
    def __init__(self,
                 log_dir,
                 eval_freq=1500,
                 n_eval_episodes=5,
                 eval_smooth_window_size=10,
                 non_eval_sample_freq=2000,
                 non_eval_smooth_window_size=1,
                 method_names=None,
                 env_name='random',
                 show_legend=False) -> None:
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_smooth_window_size = eval_smooth_window_size
        self.non_eval_sample_freq = non_eval_sample_freq
        self.non_eval_smooth_window_size = non_eval_smooth_window_size
        self.env_name = env_name
        self.show_legend = show_legend
        
        if method_names == None:
            self.method_names = {'CDDT': 'm1',
                                 'CDDT-controllers': 'm2',
                                 'ICCT-static': 'm3', 
                                 'ICCT-complete': 'm4',
                                 'ICCT-L1-sparse': 'm5a',
                                 'ICCT-1-feature': 'm5b_1',
                                 'ICCT-2-feature': 'm5b_2',
                                 'ICCT-3-feature': 'm5b_3',
                                 'MLP': 'mlp_l',
                                 'MLP-U': 'mlp_m',
                                 'MLP-L': 'mlp_s',
                                 'MLP-L1': 'mlp_l1',
                                 'MLP-L2': 'mlp_l2',
                                 'Oblique-DT': 'oblique_tree'}
        else:
            self.method_names = method_names

        self.non_eval_monitor_files = {}
        self.eval_monitor_files = {}
        for method, method_name in self.method_names.items():
            self.non_eval_monitor_files[method] = self.get_monitor_files(self.log_dir, method_name)
            self.eval_monitor_files[method] = self.get_monitor_files(self.log_dir, method_name, eval=True)
            
        self.non_eval_data = None
        self.eval_data = None

        
    def process_data(self):
        self.non_eval_data = self._process_non_eval_data(self.non_eval_monitor_files, self.non_eval_sample_freq, self.non_eval_smooth_window_size)
        self.eval_data = self._process_eval_data(self.eval_monitor_files, self.eval_freq, self.n_eval_episodes, self.eval_smooth_window_size)
        
        return

    
    def plot(self):
        self._plot_non_eval()
        self._plot_eval()
        
        return

    
    def _plot_non_eval(self):
        sns.set_style("whitegrid")
        matplotlib.rcParams.update({'font.size': 25})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2
        plt.figure(figsize=(12, 6), dpi=100)
        if self.show_legend:
            legend_flag = 'auto'
        else:
            legend_flag = False
        hue_order = ['ICCT-complete',
                     'ICCT-1-feature',
                     'ICCT-2-feature',
                     'ICCT-3-feature',
                     'ICCT-static',
                     'ICCT-L1-sparse',
                     'CDDT',
                     'CDDT-controllers',
                     'Oblique-DT',
                     'MLP',
                     'MLP-U',
                     'MLP-L',
                     'MLP-L1',
                     'MLP-L2']
        hue_order.reverse()
        color_map = {'CDDT': 'purple',
                     'CDDT-controllers': 'brown',
                     'ICCT-static': 'gold', 
                     'ICCT-complete': 'red',
                     'ICCT-L1-sparse': 'grey',
                     'ICCT-1-feature': 'darkorange',
                     'ICCT-2-feature': 'green',
                     'ICCT-3-feature': 'blue',
                     'MLP': 'darkturquoise',
                     'MLP-U': 'skyblue',
                     'MLP-L': 'pink',
                     'MLP-L1': 'teal',
                     'MLP-L2': 'coral',
                     'Oblique-DT': 'darkviolet'}
        sns.lineplot(data=self.non_eval_data, x="timesteps", y="rollout_rewards_mean", hue="method", ci=68, hue_order=hue_order, legend=legend_flag, palette=color_map)
        plt.xlabel('Time Step (k)')
        plt.ylabel('Reward')
        if self.show_legend:
            plt.legend(title=None, ncol=1, fontsize=6)
        plt.savefig(f'{self.env_name}_rollout_reward_curves.png', bbox_inches='tight')
        plt.close()        
        return
    

    def _plot_eval(self):
        sns.set_style("whitegrid")
        matplotlib.rcParams.update({'font.size': 25})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2
        plt.figure(figsize=(12, 6), dpi=100)
        if self.show_legend:
            legend_flag = 'auto'
        else:
            legend_flag = False
        hue_order = ['ICCT-complete',
                     'ICCT-1-feature',
                     'ICCT-2-feature',
                     'ICCT-3-feature',
                     'ICCT-static',
                     'ICCT-L1-sparse',
                     'CDDT',
                     'CDDT-controllers',
                     'Oblique-DT',
                     'MLP',
                     'MLP-U',
                     'MLP-L',
                     'MLP-L1',
                     'MLP-L2']
        hue_order.reverse()
        color_map = {'CDDT': 'purple',
                     'CDDT-controllers': 'brown',
                     'ICCT-static': 'gold', 
                     'ICCT-complete': 'red',
                     'ICCT-L1-sparse': 'grey',
                     'ICCT-1-feature': 'darkorange',
                     'ICCT-2-feature': 'green',
                     'ICCT-3-feature': 'blue',
                     'MLP': 'darkturquoise',
                     'MLP-U': 'skyblue',
                     'MLP-L': 'pink',
                     'MLP-L1': 'teal',
                     'MLP-L2': 'coral',
                     'Oblique-DT': 'darkviolet'}
        sns.lineplot(data=self.eval_data, x="eval_timesteps", y="eval_rewards_mean", hue="method", ci=68, hue_order=hue_order, legend=legend_flag, palette=color_map)
        plt.xlabel('Time Step (k)')
        plt.ylabel('Reward')
        if self.show_legend:
            plt.legend(title=None, ncol=1, fontsize=6)
        plt.savefig(f'{self.env_name}_eval_reward_curves.png', bbox_inches='tight')
        plt.close()
        
        return
    
    def _process_non_eval_data(self, dict_monitor_files, sample_freq, smooth_window_size):
        data_frames = []
        for method, file_names in dict_monitor_files.items():
            if len(file_names) == 0:
                pass
            for file_name in file_names:
                with open(file_name, "rt") as file_handler:
                    first_line = file_handler.readline()
                    assert first_line[0] == "#"
                    data_frame = pd.read_csv(file_handler, index_col=None)
                    rewards = data_frame['r'].to_numpy()
                    timesteps = data_frame['l'].to_numpy().cumsum()
                    new_timesteps = np.arange(0, timesteps[-1] + 1, sample_freq)
                    dist = np.tile(new_timesteps.reshape(new_timesteps.shape[0], -1),
                                   timesteps.shape[0]) - timesteps
                    sample_idx = np.argmin(np.abs(dist), axis=-1)
                    sampled_rewards = rewards[sample_idx]
                    sampled_rewards = self.moving_average(sampled_rewards, window=smooth_window_size)
                    new_timesteps = new_timesteps[new_timesteps.shape[0] - sampled_rewards.shape[0]:]/1000
                    processed_data = pd.DataFrame(
                        np.stack([sampled_rewards, new_timesteps], axis=-1), 
                        columns = ['rollout_rewards_mean', 'timesteps'])
                    method_names = [method] * new_timesteps.shape[0]
                    processed_data['method'] = method_names
                data_frames.append(processed_data)
        data_frame = pd.concat(data_frames)
        data_frame.reset_index(inplace=True)
        return data_frame

    
    def _process_eval_data(self, dict_monitor_files, eval_freq, n_eval_episodes, smooth_window_size):
        data_frames = []
        for method, file_names in dict_monitor_files.items():
            if len(file_names) == 0:
                pass
            for file_name in file_names:
                with open(file_name, "rt") as file_handler:
                    first_line = file_handler.readline()
                    assert first_line[0] == "#"
                    data_frame = pd.read_csv(file_handler, index_col=None)
                    eval_rewards = data_frame['r'].to_numpy().reshape(-1, n_eval_episodes)
                    eval_rewards_mean = eval_rewards.mean(axis=-1)
                    eval_rewards_mean = self.moving_average(eval_rewards_mean, window=smooth_window_size)
                    eval_rewards_std = eval_rewards.std(axis=-1)
                    eval_rewards_std = self.moving_average(eval_rewards_std, window=smooth_window_size)
                    
                    eval_lengths = data_frame['l'].to_numpy().reshape(-1, n_eval_episodes)
                    eval_lengths_mean = eval_lengths.mean(axis=-1)
                    eval_lengths_mean = self.moving_average(eval_lengths_mean, window=smooth_window_size)
                    eval_lengths_std = eval_lengths.std(axis=-1)
                    eval_lengths_std = self.moving_average(eval_lengths_std, window=smooth_window_size)
                    
                    eval_timesteps = np.arange(eval_freq, eval_freq * eval_rewards.shape[0] + 1, eval_freq)
                    eval_timesteps = eval_timesteps[eval_timesteps.shape[0] - eval_rewards_mean.shape[0]:]/1000
                    
                    processed_data = pd.DataFrame(
                        np.stack([eval_rewards_mean, eval_rewards_std, eval_lengths_mean, eval_lengths_std, eval_timesteps], axis=-1), 
                        columns = ['eval_rewards_mean', 'eval_rewards_std', 'eval_lengths_mean', 'eval_lengths_std', 'eval_timesteps'])
                    method_names = [method] * eval_timesteps.shape[0]
                    processed_data['method'] = method_names
                data_frames.append(processed_data)
        data_frame = pd.concat(data_frames)
        data_frame.reset_index(inplace=True)
        return data_frame
        
        

    def get_monitor_files(self, path, method_name, eval=False) -> List[str]:
        eval_files = glob(os.path.join(path, '*' + 'eval' + '*' + method_name + '*' + 'monitor.csv'))
        all_files = glob(os.path.join(path, '*' + method_name + '*' + 'monitor.csv'))
        non_eval_files = list(set(all_files) - set(eval_files))
        
        if eval:
            ret = eval_files
        else:
            ret = non_eval_files
        
        return ret


    def moving_average(self, values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')


    # ---- Results table and Pareto plot generation ----
    # Merged from results_plotter.py. Provides:
    #   - generate_results_table(): Table 1-style (reward + param counts)
    #   - plot_pareto(): Figure 5-style (reward vs. active params)

    # Maps display name -> (file prefix, policy_type for param_counter)
    METHOD_REGISTRY = {
        'CDDT':             ('m1',      'ddt'),
        'CDDT-controllers': ('m2',      'ddt'),
        'ICCT-static':      ('m3',      'ddt'),
        'ICCT-complete':    ('m4',      'ddt'),
        'ICCT-L1-sparse':   ('m5a',     'ddt'),
        'ICCT-1-feature':   ('m5b_1',   'ddt'),
        'ICCT-2-feature':   ('m5b_2',   'ddt'),
        'ICCT-3-feature':   ('m5b_3',   'ddt'),
        'MLP':              ('mlp_l',   'mlp'),
        'MLP-U':            ('mlp_m',   'mlp'),
        'MLP-L':            ('mlp_s',   'mlp'),
        'MLP-L1':           ('mlp_l1',  'mlp'),
        'MLP-L2':           ('mlp_l2',  'mlp'),
        'Oblique-DT':       ('oblique_tree', 'oblique_tree'),
    }

    def load_final_eval_rewards(self, method_prefix, n_eval_episodes=None):
        """
        Load final evaluation rewards for a method across all seeds.
        Returns list of per-seed mean rewards.
        """
        if n_eval_episodes is None:
            n_eval_episodes = self.n_eval_episodes
        eval_files = glob(os.path.join(self.log_dir, f'*eval*{method_prefix}*monitor.csv'))
        seed_rewards = []
        for f in eval_files:
            try:
                with open(f, 'rt') as fh:
                    first_line = fh.readline()
                    if first_line[0] != '#':
                        continue
                    df = pd.read_csv(fh, index_col=None)
                    if len(df) == 0:
                        continue
                    rewards = df['r'].to_numpy()
                    if len(rewards) >= n_eval_episodes:
                        final_rewards = rewards[-n_eval_episodes:]
                    else:
                        final_rewards = rewards
                    seed_rewards.append(np.mean(final_rewards))
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")
        return seed_rewards

    def generate_results_table(self, model_dir=None, methods=None,
                               alg_type='sac', threshold=0.005):
        """
        Generate a paper-style results table with:
          Method | Mean Reward +/- Std | Active Params | Total Params

        :param model_dir: directory with saved models (for param counting).
                          If None, skips param counting.
        :param methods: dict of {display_name: (prefix, policy_type)}.
                        Defaults to METHOD_REGISTRY.
        :param alg_type: 'sac' or 'td3' (for loading RL models)
        :param threshold: active param threshold
        :return: pandas DataFrame
        """
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        from icct.core.param_counter import count_active_params

        if methods is None:
            methods = self.METHOD_REGISTRY

        rows = []
        for display_name, (prefix, policy_type) in methods.items():
            seed_rewards = self.load_final_eval_rewards(prefix)
            if len(seed_rewards) == 0:
                continue

            mean_reward = np.mean(seed_rewards)
            std_reward = np.std(seed_rewards)
            row = {
                'Method': display_name,
                'Mean Reward': mean_reward,
                'Std Reward': std_reward,
                'N Seeds': len(seed_rewards),
                'Reward': f'{mean_reward:.1f} +/- {std_reward:.1f}',
            }

            if model_dir:
                param_info = self._count_params_for_method(
                    model_dir, prefix, policy_type, alg_type, threshold
                )
                if param_info:
                    active, total, analytical = param_info
                    row['Active Params'] = active
                    row['Total Params'] = total
                    row['Analytical Params'] = analytical if analytical else ''
                else:
                    row['Active Params'] = ''
                    row['Total Params'] = ''
                    row['Analytical Params'] = ''

            rows.append(row)

        return pd.DataFrame(rows)

    def _count_params_for_method(self, model_dir, prefix, policy_type,
                                 alg_type='sac', threshold=0.005):
        """Count active params for one method's best model."""
        from icct.core.param_counter import count_active_params

        # Find model files
        patterns = [
            os.path.join(model_dir, f'*{prefix}*seed*', 'best_model.zip'),
            os.path.join(model_dir, f'best_model*{prefix}*.zip'),
        ]
        files = []
        for pattern in patterns:
            files.extend(glob(pattern))
        if not files:
            return None

        model_path = sorted(set(files))[0]
        try:
            if policy_type == 'oblique_tree':
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                if alg_type == 'sac':
                    from icct.rl_helpers.sac import SAC as Alg
                else:
                    from icct.rl_helpers.td3 import TD3 as Alg
                model = Alg.load(model_path, device='cpu')
            results = count_active_params(model, policy_type=policy_type,
                                          threshold=threshold, include_bias=True)
            return (results['active_params'], results['total_params'],
                    results.get('analytical_active_params'))
        except Exception as e:
            print(f"Warning: Could not load model {model_path}: {e}")
            return None

    def plot_pareto(self, df, save_path=None):
        """
        Pareto scatter: Active Params (x, log scale) vs. Mean Reward (y).
        Matches Figure 5 style in the paper.

        :param df: DataFrame from generate_results_table()
        :param save_path: output file path (default: {env_name}_pareto_plot.png)
        """
        if 'Active Params' not in df.columns:
            print("No parameter count data for Pareto plot.")
            return

        plot_df = df.dropna(subset=['Active Params']).copy()
        plot_df['Active Params'] = pd.to_numeric(plot_df['Active Params'], errors='coerce')
        plot_df = plot_df.dropna(subset=['Active Params'])
        if len(plot_df) == 0:
            print("No valid data for Pareto plot.")
            return

        color_map = {'CDDT': 'purple', 'CDDT-controllers': 'brown',
                     'ICCT-static': 'gold', 'ICCT-complete': 'red',
                     'ICCT-L1-sparse': 'grey', 'ICCT-1-feature': 'darkorange',
                     'ICCT-2-feature': 'green', 'ICCT-3-feature': 'blue',
                     'MLP': 'darkturquoise', 'MLP-U': 'skyblue', 'MLP-L': 'pink',
                     'MLP-L1': 'teal', 'MLP-L2': 'coral',
                     'Oblique-DT': 'darkviolet'}

        sns.set_style("whitegrid")
        matplotlib.rcParams.update({'font.size': 14})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2
        fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

        for _, row in plot_df.iterrows():
            name = row['Method']
            color = color_map.get(name, 'black')
            ax.errorbar(
                row['Active Params'], row['Mean Reward'],
                yerr=row['Std Reward'],
                fmt='o', markersize=10, capsize=5,
                color=color, label=name,
                markeredgecolor='black', markeredgewidth=0.5,
            )

        ax.set_xlabel('Active Parameters', fontsize=14)
        ax.set_ylabel('Mean Reward', fontsize=14)
        ax.set_title(f'{self.env_name} — Reward vs. Active Parameters', fontsize=16)
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.set_xscale('log')
        plt.tight_layout()

        out = save_path or f'{self.env_name}_pareto_plot.png'
        plt.savefig(out, bbox_inches='tight')
        print(f"Pareto plot saved to {out}")
        plt.close()

    def print_latex_table(self, df):
        """Print results in LaTeX tabular format."""
        print(f"\n% Table for {self.env_name}")
        print("\\begin{tabular}{l r r r}")
        print("\\toprule")
        print("Method & Reward & Active Params & Total Params \\\\")
        print("\\midrule")
        for _, row in df.iterrows():
            reward_str = row.get('Reward', '')
            active = row.get('Active Params', '')
            total = row.get('Total Params', '')
            print(f"{row['Method']} & {reward_str} & {active} & {total} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


    
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
                                 'MLP-L': 'mlp_s'}
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
                     'MLP',
                     'MLP-U',
                     'MLP-L']
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
                     'MLP-L': 'pink'}
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
                     'MLP',
                     'MLP-U',
                     'MLP-L']
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
                     'MLP-L': 'pink'}
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

    
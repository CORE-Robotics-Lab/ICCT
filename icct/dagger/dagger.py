# Created by Yaru Niu
# Reference: https://github.com/obastani/viper/blob/master/python/viper/core/rl.py

import numpy as np
import pickle
import torch

class DAgger:
    def __init__(self,
                 env,
                 oracle_model, 
                 dt_model,
                 n_rollouts,
                 iterations,
                 max_samples,
                 is_reweight,
                 n_q_samples):
        self.env = env
        self.oracle = oracle_model
        self.dt = dt_model
        self.best_dt = None
        self.n_rollouts = n_rollouts
        self.iterations = iterations
        self.max_samples = max_samples
        self.is_reweight = is_reweight
        self.n_q_samples = n_q_samples
        self.dataset_obs = []
        self.dataset_act = []
        self.dataset_prob = []

    def generate_uniform_combinations(self, n_samples=10, act_dim=2):
        # Step 1: Generate n values in [-1, 1]
        values = torch.linspace(-1, 1, n_samples)
        # Step 2: Create a meshgrid of all combinations (n^act_dim total combinations)
        grids = torch.meshgrid([values] * act_dim)
        # Step 3: Stack and reshape to shape (n^act_dim, act_dim)
        combinations = torch.stack(grids, dim=-1).reshape(-1, act_dim)
        return combinations

    def get_rollout(self, execute_dt=True):
        obs = self.env.reset()
        done = False
        rollout = []
        act_dim = self.env.action_space.shape[0]
        n_q_samples = self.n_q_samples
        while not done:
            # oracle_act: action that can be directly used by the environment
            oracle_act, _ = self.oracle.predict(obs, deterministic=True)
            # raw_oracle_act: raw mean action output by the policy network, used to train the DT
            processed_obs, raw_oracle_act = self.oracle.actor.get_sa_pair()
            sampled_obss = processed_obs.repeat(n_q_samples ** act_dim, 1)
            sampled_acts = self.generate_uniform_combinations(n_samples=n_q_samples, act_dim=act_dim)
            q1_values = self.oracle.critic.q1_forward(sampled_obss, sampled_acts)
            loss = torch.max(q1_values, dim=0).values - torch.min(q1_values, dim=0).values

            if execute_dt:
                act = self.dt.predict(processed_obs.cpu().numpy())
            else:
                act = oracle_act

            next_obs, rwd, done, info = self.env.step(act)
            rollout.append((processed_obs.cpu().numpy(), raw_oracle_act.cpu().numpy(), rwd, loss.detach().cpu().numpy()))
            obs = next_obs

        return rollout

    def get_rollouts(self, execute_dt=True):
        rollouts = []
        for n in range(self.n_rollouts):
            rollouts.extend(self.get_rollout(execute_dt))
        return rollouts
    
    def sample_batch_idx(self, probs, max_samples, is_reweight):
        probs = probs / np.sum(probs)
        if is_reweight:
            idx = np.random.choice(probs.shape[0], size=min(max_samples, np.sum(probs > 0)), p=probs)
        else:
            idx = np.random.choice(probs.shape[0], size=min(max_samples, np.sum(probs > 0)), replace=False)   
        return idx

    def train(self, save_path):
        first_batch = self.get_rollouts(execute_dt=False)
        self.dataset_obs.extend((obs for obs, _, _, _ in first_batch))
        self.dataset_act.extend((act for _, act, _, _ in first_batch))
        self.dataset_prob.extend((prob for _, _, _, prob in first_batch))
        best_rwd = -9e5

        for i in range(self.iterations):
            dataset_obs = np.concatenate(self.dataset_obs, axis=0)
            dataset_act = np.concatenate(self.dataset_act, axis=0)
            dataset_prob = np.concatenate(self.dataset_prob, axis=0)

            idx = self.sample_batch_idx(probs=dataset_prob, max_samples=self.max_samples, is_reweight=self.is_reweight)

            self.dt.train(dataset_obs[idx], dataset_act[idx])
            
            later_batch = self.get_rollouts(execute_dt=True)
            self.dataset_obs.extend((obs for obs, _, _, _ in later_batch))
            self.dataset_act.extend((act for _, act, _, _ in later_batch))
            self.dataset_prob.extend((prob for _, _, _, prob in later_batch))

            average_rwd = np.sum((rwd for _, _, rwd, _ in later_batch)) / self.n_rollouts
            print('average reward: ', average_rwd)
            if average_rwd >= best_rwd:
                best_rwd = average_rwd
                print('save the best model!')
                self.best_dt = self.dt.clone()
                self.save_best_dt(save_path)

    def save_best_dt(self, save_path):
        pickle.dump(self.best_dt, open(save_path, 'wb'))

    def load_best_dt(self, load_path):
        self.best_dt = pickle.load(open(load_path, 'rb'))

    def evaluate(self, n_episodes):
        print('number of leaves', self.best_dt.tree.get_n_leaves())
        episode_reward_list = []
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                _, _ = self.oracle.predict(obs, deterministic=True)
                obs, _ = self.oracle.actor.get_sa_pair()
                action = self.best_dt.predict(obs.cpu().numpy())
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            episode_reward_list.append(episode_reward)
        print(episode_reward_list)
        print(np.mean(episode_reward_list))
        print(np.std(episode_reward_list))
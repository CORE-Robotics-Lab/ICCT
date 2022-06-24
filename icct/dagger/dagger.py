# Created by Yaru Niu
# Reference: https://github.com/obastani/viper/blob/master/python/viper/core/rl.py

import numpy as np
import pickle

class DAgger:
    def __init__(self,
                 env,
                 oracle_model, 
                 dt_model,
                 n_rollouts,
                 iterations):
        self.env = env
        self.oracle = oracle_model
        self.dt = dt_model
        self.best_dt = None
        self.n_rollouts = n_rollouts
        self.iterations = iterations
        self.dataset_obs = []
        self.dataset_act = []

    def get_rollout(self, execute_dt=True):
        obs = self.env.reset()
        done = False
        rollout = []
        
        while not done:
            oracle_act, _ = self.oracle.predict(obs, deterministic=True)
            processed_obs, raw_oracle_act = self.oracle.actor.get_sa_pair()

            if execute_dt:
                act = self.dt.predict(processed_obs.cpu().numpy())
            else:
                act = oracle_act

            next_obs, rwd, done, info = self.env.step(act)
            rollout.append((processed_obs.cpu().numpy(), raw_oracle_act.cpu().numpy(), rwd))
            obs = next_obs

        return rollout

    def get_rollouts(self, execute_dt=True):
        rollouts = []
        for n in range(self.n_rollouts):
            rollouts.extend(self.get_rollout(execute_dt))
        return rollouts

    def train(self, save_path):
        first_batch = self.get_rollouts(execute_dt=False)
        self.dataset_obs.extend((obs for obs, _, _ in first_batch))
        self.dataset_act.extend((act for _, act, _ in first_batch))
        best_rwd = -9e5

        for i in range(self.iterations):
            dataset_obs = np.concatenate(self.dataset_obs, axis=0)
            dataset_act = np.concatenate(self.dataset_act, axis=0)
            self.dt.train(dataset_obs, dataset_act)
            
            later_batch = self.get_rollouts(execute_dt=True)
            self.dataset_obs.extend((obs for obs, _, _ in later_batch))
            self.dataset_act.extend((act for _, act, _ in later_batch))

            average_rwd = np.sum((rwd for _, _, rwd in later_batch)) / self.n_rollouts
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
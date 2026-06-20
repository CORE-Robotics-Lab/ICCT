# Created by Rohan Paleja

import numpy as np
import pickle


class ModifiedDT:
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
        average_rwd_ = np.sum((rwd for _, _, rwd in first_batch)) / self.n_rollouts
        print(average_rwd_)
        best_rwd = -9e5

        dataset_obs = np.concatenate(self.dataset_obs, axis=0)
        dataset_act = np.concatenate(self.dataset_act, axis=0)
        self.dt.train(dataset_obs, dataset_act)
        self.best_dt = self.dt.clone()
        self.save_best_dt(save_path)

    def save_best_dt(self, save_path):
        pickle.dump(self.best_dt, open(save_path, 'wb'))

    def load_best_dt(self, load_path):
        self.best_dt = pickle.load(open(load_path, 'rb'))

    def evaluate(self, n_episodes):

        data = self.best_dt.tree.summary()
        self.analyze_linear_tree(data, len(self.best_dt.tree.feature_importances_))
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

    def analyze_linear_tree(self, tree_data, input_space_size):
        """
        Analyzes a linear tree structure to count nodes and parameters.

        Args:
            tree_data (dict): A dictionary representing the linear tree, where keys
                              are node IDs and values are dictionaries of node attributes.
        """
        leaf_nodes = 0
        decision_nodes = 0

        # --- Node Counting ---
        # Iterate through each node in the tree dictionary
        for node_id, node_info in tree_data.items():
            # A node is a leaf if it does not have a 'children' key.
            if 'children' not in node_info:
                leaf_nodes += 1
            # Otherwise, it's a decision node.
            else:
                decision_nodes += 1

        # --- Parameter Calculation ---
        # Parameters for decision nodes: 2 per node (feature 'col' and threshold 'th')
        decision_node_params = decision_nodes * 2

        # Parameters for leaf nodes: 5 per node (input space size + 1 for intercept)
        # As per the user's request.
        leaf_node_params = leaf_nodes * (input_space_size + 1) * self.env.action_space.shape[0]

        # Total parameters is the sum of parameters from both types of nodes
        total_params = decision_node_params + leaf_node_params

        # --- Printing the Results ---
        print("--- Linear Tree Analysis ---")
        print(f"Total Nodes: {len(tree_data)}")
        print(f"Decision Nodes: {decision_nodes}")
        print(f"Leaf Nodes: {leaf_nodes}")
        print("-" * 28)
        print("--- Parameter Calculation ---")
        print(f"Parameters from Decision Nodes: {decision_nodes} nodes * 2 params/node = {decision_node_params}")
        print(f"Parameters from Leaf Nodes: {leaf_nodes} leaves * {input_space_size + 1} params/leaf = {leaf_node_params}")
        print(f"Total Model Parameters: {total_params}")

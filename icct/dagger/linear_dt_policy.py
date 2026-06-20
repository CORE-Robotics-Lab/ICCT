# Created by Yaru Niu

import numpy as np
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor

class LinearDTPolicy:
    def __init__(self, action_space, max_depth):
        self.action_space = action_space
        self.max_depth = max_depth

    def fit(self, obss, acts):
        self.tree = LinearTreeRegressor(base_estimator=LinearRegression())
        self.tree.fit(obss, acts)

    def train(self, obss, acts):
        self.fit(obss, acts)

    def _predict(self, obs):
        return self.tree.predict(obs)

    def predict(self, obs):
        raw_act = self._predict(obs)[0]
        squashed_act = np.tanh(raw_act)
        return self.unscale_action(squashed_act)

    def unscale_action(self, scaled_action):
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def clone(self):
        clone = LinearDTPolicy(self.action_space, self.max_depth)
        clone.tree = self.tree
        return clone

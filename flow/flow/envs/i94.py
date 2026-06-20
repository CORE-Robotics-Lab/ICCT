"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
}


class I94POEnv(Env):
    """Partially observable I94 environment.

    This environment is used to train autonomous vehicles to get to destination as fast as possible while maintaining courtesy for other vehicles.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2

    States
        The observation consists of 4 (number of max lanes) * 2 (front/tail) * 2 (distance/speed) + 3 (ego edge/speed/lane)
        If the lane is less than # max lanes, the distance/speed will be filled with 0.

    Actions
        The action space consists of acceleration and lane changing for the
        autonomous vehicle. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

    Rewards
        The reward function rewards the ego car velocity
        while penalizing emergency breaks for all vehicles (courtesy to other cars).
        The reward also penalizes not going to the exit ramp.

    Termination
        A rollout is terminated if the ego car exits.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.highway_lanes = network.net_params.additional_params["highway_lanes"]
        self.merge_lanes = network.net_params.additional_params["merge_lanes"]
        self.num_lanes = self.highway_lanes + self.merge_lanes
        self.num_ramps = network.net_params.additional_params["number_ramp"]

        # names of the rl vehicles controlled at any step
        self.the_rl_vehicle = None
        self.visible = []
        self.episode_finished = False

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        # return Box(
        #     low=[-abs(self.env_params.additional_params["max_decel"]), -1],
        #     high=[self.env_params.additional_params["max_accel"], 1],
        #     shape=(self.num_rl, ),
        #     dtype=np.float32)
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1]
        ub = [max_accel, 1]

        return Box(np.array(lb, dtype=np.float32), np.array(ub, dtype=np.float32), dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(self.num_lanes * 4 + 3, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        # please implement this in a wrapper to apply the actions
        super(I94POEnv, self)._apply_rl_actions(rl_actions)

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        self.visible = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        observation = [0.0 for _ in range(self.num_lanes * 4 + 3)]
        if self.the_rl_vehicle is None:
            return observation
        if self.the_rl_vehicle not in self.k.vehicle.get_rl_ids():
            return observation

        rl_id = self.the_rl_vehicle

        current_edge = self.k.vehicle.get_edge(rl_id)
        route = self.available_routes["rl"][0][0]
        assert len(route) == 3 * self.num_ramps + 2
        for idx, edge in enumerate(route):
            if edge == current_edge:
                observation[0] = idx / len(route)
                break

        observation[1] = self.k.vehicle.get_speed(rl_id) / max_speed
        observation[2] = self.k.vehicle.get_lane(rl_id) / self.num_lanes

        headway = [0.0] * self.num_lanes
        tailway = [0.0] * self.num_lanes
        lane_headways = self.k.vehicle.get_lane_headways(rl_id)
        lane_tailways = self.k.vehicle.get_lane_tailways(rl_id)
        headway[0:len(lane_headways)] = lane_headways
        tailway[0:len(lane_tailways)] = lane_tailways

        vel_in_front = [0.0] * self.num_lanes
        vel_behind = [0.0] * self.num_lanes
        lane_leaders = self.k.vehicle.get_lane_leaders(rl_id)
        lane_followers = self.k.vehicle.get_lane_followers(rl_id)
        for j, lane_leader in enumerate(lane_leaders):
            if lane_leader != '':
                headway[j] /= max_length
                vel_in_front[j] = self.k.vehicle.get_speed(lane_leader) / max_speed
                self.visible.append(lane_leader)
        for j, lane_follower in enumerate(lane_followers):
            if lane_follower != '':
                tailway[j] /= max_length
                vel_behind[j] = self.k.vehicle.get_speed(lane_follower) / max_speed
                self.visible.append(lane_follower)

        # add the headways, tailways, and speed for all lane leaders and followers
        observation[3:self.num_lanes*4+3] = np.concatenate((headway, tailway, vel_in_front, vel_behind))

        return observation

    def step(self, rl_actions):
        next_observation, reward, done, infos = super(I94POEnv, self).step(rl_actions)
        # if there was rl vehicles and now the vehicle is gone, the episode is done!
        done = done or self.episode_finished
        return next_observation, reward, done, infos

    def reset(self):
        self.episode_finished = False
        self.the_rl_vehicle = None
        return super(I94POEnv, self).reset()

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # return a reward of -100 if a collision occurred
        if kwargs["fail"]:
            return -100

        if self.the_rl_vehicle not in self.k.vehicle.get_rl_ids():
            return 0

        if self.the_rl_vehicle is None:
            return 0

        veh_id = self.the_rl_vehicle
        speed_reward = self.k.vehicle.get_speed(veh_id)
        if abs(speed_reward) > 200:
            # sumo is doing something weird
            print("super large speed", speed_reward)
            speed_reward = 0

        # TODO: penalty for emergency braking
        # penalize small time headways
        too_close_penalty = 0
        t_min = 1  # smallest acceptable time headway
        lead_id = self.k.vehicle.get_leader(veh_id)
        if lead_id not in ["", None] and self.k.vehicle.get_speed(veh_id) > 0:
            t_headway = max(self.k.vehicle.get_headway(veh_id) / self.k.vehicle.get_speed(veh_id), 0)
            too_close_penalty += min((t_headway - t_min) / t_min, 0)
        assert too_close_penalty <= 0

        routing_reward = 0
        edge = self.k.vehicle.get_edge(veh_id)
        if edge not in self.available_routes["rl"][0][0] and ":" not in edge:
            # ":" means it's in a joint
            routing_reward -= 10
        # if edge == "exit_0":
        #     routing_reward += 1

        time_penalty = -1

        # print(speed_reward + too_close_penalty + routing_reward, speed_reward, too_close_penalty, routing_reward)
        return speed_reward + time_penalty + too_close_penalty + routing_reward

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        if self.the_rl_vehicle is None:
            for veh_id in self.k.vehicle.get_rl_ids():
                self.the_rl_vehicle = veh_id
                break
        else:
            if self.the_rl_vehicle not in self.k.vehicle.get_rl_ids():
                self.episode_finished = True
            for veh_id in self.k.vehicle.get_rl_ids():
                if veh_id != self.the_rl_vehicle:
                    self.k.vehicle.set_color(veh_id, (255, 255, 255))
                else:
                    self.k.vehicle.set_color(veh_id, (255, 0, 0))

        # specify observed vehicles
        for veh_id in self.visible:
            self.k.vehicle.set_observed(veh_id)

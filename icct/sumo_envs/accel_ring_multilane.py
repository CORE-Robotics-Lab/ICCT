# Created by Yaru Niu and Chace Ritchie
# Reference: https://github.com/flow-project/flow/tree/master/tutorials

from flow.controllers.car_following_models import SimCarFollowingController
from flow.core import rewards
from flow.envs import AccelEnv as RingAccelEnv
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
#from flow.networks.figure_eight import FigureEightNetwork, ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, ContinuousRouter, SimLaneChangeController
import numpy as np

class LaneChangeAccelEnv_Wrapper(LaneChangeAccelEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

    def _apply_rl_actions(self, actions):
        acceleration = actions[::2]
        direction = actions[1::2]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        
        # discretize the direction values
        lane_changing_plus = \
            [direction[i] >= 0.5 and direction[i] <= 1 for i, veh_id in enumerate(sorted_rl_ids)]
        direction[lane_changing_plus] = \
            np.array([1] * sum(lane_changing_plus))
        
        lane_changing_minus = \
            [direction[i] >= -1 and direction[i] <= -0.5 for i, veh_id in enumerate(sorted_rl_ids)]
        direction[lane_changing_minus] = \
            np.array([-1] * sum(lane_changing_minus))
        
        lane_keeping = \
            [direction[i] > -0.5 and direction[i] < 0.5 for i, veh_id in enumerate(sorted_rl_ids)]
        direction[lane_keeping] = \
            np.array([0] * sum(lane_keeping))
        
        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.k.vehicle.get_last_lc(veh_id)
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))
        if direction[0] != 0 and direction[0] != 1 and direction[0] != -1:
            print('wrong value of direction!', direction[0])
            direction = np.array([0.])
        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # compute the system-level performance of vehicles from a velocity
        # perspective
        reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        return reward



# time horizon of a single rollout
HORIZON = 1500

ADDITIONAL_NET_PARAMS["lanes"] = 2
ADDITIONAL_NET_PARAMS["speed_limit"] = 12


# We place one autonomous vehicle and 21 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0,
        max_speed=12
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)

vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode="no_lc_safe",),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    num_vehicles=1)    


ring_accel_lc_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name=LaneChangeAccelEnv_Wrapper,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1, # seconds per simulation step
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "target_velocity": 5,
            "sort_vehicles": False,
            "max_accel": 3,
            "max_decel": 3,
            "lane_change_duration": 5
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy()
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        bunching=20,
    ),
)
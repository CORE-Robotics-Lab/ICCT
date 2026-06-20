"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
on-ramp merge to a single lane open highway network. File is close to singleagent_merge.py
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig
from flow.core.params import NetParams, InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController
from flow.envs import MergePOEnv
from flow.networks import MergeNetwork
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
import numpy as np
from copy import deepcopy


class MergePOEnv_Wrapper(MergePOEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

    def _apply_rl_actions(self, rl_actions):
        acceleration = rl_actions[::2]
        direction = rl_actions[1::2]

        """See class definition."""
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue

            # discretize the direction values
            lane_changing_plus = \
                [direction[i] >= 0.5 and direction[i] <= 1]
            direction[lane_changing_plus] = \
                np.array([1] * sum(lane_changing_plus))

            lane_changing_minus = \
                [direction[i] >= -1 and direction[i] <= -0.5]
            direction[lane_changing_minus] = \
                np.array([-1] * sum(lane_changing_minus))

            lane_keeping = \
                [direction[i] > -0.5 and direction[i] < 0.5]
            direction[lane_keeping] = \
                np.array([0] * sum(lane_keeping))

            if direction[0] != 0 and direction[0] != 1 and direction[0] != -1:
                print('wrong value of direction!', direction[0])
                direction = np.array([0.])
            # self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])
            self.k.vehicle.apply_acceleration(rl_id, acc=acceleration)
            self.k.vehicle.apply_lane_change(rl_id, direction=direction)


# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0

# time horizon of a single rollout
HORIZON = 600
# number of rollouts per training iteration

# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [1, 5, 13, 17][EXP_NUM]

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 3
additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=5)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE*2,
    depart_lane="free",
    depart_speed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE*.4,

    depart_lane="free",
    depart_speed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=500,
    depart_lane="free",
    depart_speed=1)

merge_params = dict(
    # name of the experiment
    exp_tag="stabilizing_open_network_merges",

    # name of the flow environment the experiment is running on
    env_name=MergePOEnv_Wrapper,

    # name of the network class the experiment is running on
    network=MergeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": NUM_RL,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)

merge_params_visual = deepcopy(merge_params)
merge_params_visual["sim"].render = True

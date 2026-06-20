"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
on-ramp merge to a single lane open highway network. File is close to singleagent_merge.py
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig
from flow.core.params import NetParams, InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.networks.i94 import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController, BaseLaneChangeController, I94Router
from flow.envs import I94POEnv
from flow.networks import I94Network
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
import numpy as np
from copy import deepcopy


class I94SimParams(SumoParams):
    def __init__(self, disable_rl_vehicle_auto_coloring, *args, **kwargs):
        self.disable_rl_vehicle_auto_coloring = disable_rl_vehicle_auto_coloring
        super(I94SimParams, self).__init__(*args, **kwargs)


class I94POEnv_Wrapper(I94POEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

    def _apply_rl_actions(self, rl_actions):
        acceleration = rl_actions[::2]
        direction = rl_actions[1::2]

        if self.the_rl_vehicle is None:
            return

        """See class definition."""
        for i, rl_id in enumerate([self.the_rl_vehicle]):
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


# class RampInflowLaneController(BaseLaneChangeController):
#     def get_lane_change_action(self, env):
#         current_lane = env.k.vehicle.get_lane(self.veh_id)
#         current_edge = env.k.vehicle.get_edge(self.veh_id)
#         current_route = env.k.vehicle.get_route(self.veh_id)
#         num_lanes = env.k.network.num_lanes(current_edge)
#         if "Runway" in current_edge:
#             if current_lane == 0:
#                 return 1
#         elif "After_Merge" in current_edge:
#             if "After_Exit" in current_route[-1]:
#                 return 0
#             exit_number = int(current_route[-1][5:])
#             current_edge_number = int(current_edge[12:])
#             if exit_number == current_edge_number:
#                 if current_lane > 0:
#                     return -1
#         return 0
#

class MergeAndExitLaneController(BaseLaneChangeController):
    def get_lane_change_action(self, env):
        current_lane = env.k.vehicle.get_lane(self.veh_id)
        current_edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)
        num_lanes = env.k.network.num_lanes(current_edge)
        if "Runway" in current_edge:
            if current_lane < num_lanes - 1:
                return 1
        elif "After_Merge" in current_edge:
            if "After_Exit" in current_route[-1]:
                return 0
            exit_number = int(current_route[-1][5:])
            current_edge_number = int(current_edge[12:])
            if exit_number == current_edge_number:
                if current_lane > 0:
                    return -1
        return 0


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

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 3
additional_net_params["number_ramp"] = 3

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    lane_change_controller=(MergeAndExitLaneController, {}),
    # lane_change_params=SumoLaneChangeParams(
    #     model="SL2015",
    #     lc_sublane=1.0,
    # ),
    num_vehicles=2)
vehicles.add(
    veh_id="human_merging",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    lane_change_controller=(MergeAndExitLaneController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=2)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    routing_controller=(I94Router, {}),
    num_vehicles=0)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="Inflow_Highway",
    vehs_per_hour=1000,
    depart_lane="free",
    depart_speed=10)
inflow.add(
    veh_type="rl",
    edge="Spawn_1",
    vehs_per_hour=10,
    # number=1,
    depart_lane="free",
    depart_speed=10,
    name="rl")
for i in range(1, additional_net_params["number_ramp"] + 1):
    inflow.add(
        veh_type="human_merging",
        edge=f"Spawn_{i}",
        vehs_per_hour=100,
        depart_lane="free",
        depart_speed=1)

i94_params = dict(
    # name of the experiment
    exp_tag="stabilizing_open_network_merges",

    # name of the flow environment the experiment is running on
    env_name=I94POEnv_Wrapper,

    # name of the network class the experiment is running on
    network=I94Network,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=I94SimParams(
        disable_rl_vehicle_auto_coloring=True,
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

i94_params_visual = deepcopy(i94_params)
i94_params_visual["sim"].render = True

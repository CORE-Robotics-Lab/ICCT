from flow.controllers.car_following_models import SimCarFollowingController
from flow.core import rewards
from flow.envs import AccelEnv as RingAccelEnv
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
# from flow.networks.figure_eight import FigureEightNetwork, ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams, InFlows
from flow.controllers import RLController, IDMController, ContinuousRouter, SimLaneChangeController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController
from flow.envs import BottleneckDesiredVelocityEnv
from flow.networks import BottleneckNetwork
import numpy as np
from copy import deepcopy

# Rohan additions to modify to highway
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS

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
HORIZON = 1000

# ADDITIONAL_NET_PARAMS["lanes"] = 2
# ADDITIONAL_NET_PARAMS["speed_limit"] = 12

# We place one autonomous vehicle and 21 human-driven vehicles in the network
vehicles = VehicleParams()
# vehicles.add(
#     veh_id="human",
#     acceleration_controller=(SimCarFollowingController, {
#         "noise": 0.2
#     }),
#     car_following_params=SumoCarFollowingParams(
#         min_gap=0,
#         max_speed=12
#     ),
#     routing_controller=(ContinuousRouter, {}),
#     num_vehicles=9) # was 21
SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.10
vehicles.add(
    veh_id="human",
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)
# vehicles.add(
#     veh_id='rl',
#     acceleration_controller=(RLController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     lane_change_params=SumoLaneChangeParams(lane_change_mode="no_lc_safe", ),
#     car_following_params=SumoCarFollowingParams(
#         speed_mode="obey_safe_speed",
#         decel=1.5,
#     ),
#     num_vehicles=1)
vehicles.add(
    veh_id="followerstopper",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)

controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
additional_env_params = {
    "target_velocity": 40,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "controlled_segments": controlled_segments,
    "symmetric": False,
    "observed_segments": num_observed_segments,
    "reset_inflow": False,
    "lane_change_duration": 5,
    "max_accel": 3,
    "max_decel": 3,
    "inflow_range": [1000, 2000]
}

# flow rate
flow_rate = 2300 * SCALING

# percentage of flow coming out of each lane
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    depart_lane="random",
    depart_speed=10)
inflow.add(
    veh_type="followerstopper",
    edge="1",
    vehs_per_hour=flow_rate * AV_FRAC,
    depart_lane="random",
    depart_speed=10)

traffic_lights = TrafficLightParams()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING, "speed_limit": 23}
net_params = NetParams(
    inflows=inflow,
    additional_params=additional_net_params)

bottleneck_params = dict(
    # name of the experiment
    exp_tag="DesiredVelocity",

    # name of the flow environment the experiment is running on
    env_name=BottleneckDesiredVelocityEnv,

    # name of the network class the experiment is running on
    network=BottleneckNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps=40,
        sims_per_step=1,
        horizon=HORIZON,
        additional_params=additional_env_params,
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
    initial=InitialConfig(
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"],
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=traffic_lights,
)

bottleneck_params_visual = deepcopy(bottleneck_params)
bottleneck_params_visual["sim"].render = True

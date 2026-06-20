from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.networks.figure_eight import FigureEightNetwork
from flow.networks.bottleneck import BottleneckNetwork
from flow.networks.merge import MergeNetwork
from flow.controllers import SimLaneChangeController, ContinuousRouter, IDMController
from flow.envs.bottleneck import BottleneckEnv
from icct.sumo_envs.merge import MergePOEnv_Wrapper
from flow.envs.loop.loop_accel import AccelEnv
from flow.core.experiment import Experiment

import logging
import os
from copy import deepcopy


def run_exp_bottleneck(flow_rate,
            scaling=1,
            disable_tb=True,
            disable_ramp_meter=True,
            n_crit=1000,
            feedback_coef=20):
    # Set up SUMO to render the results, take a time_step of 0.5 seconds per simulation step
    sim_params = SumoParams(
        sim_step=0.5,
        render=True,
        overtake_right=False,
        restart_instance=False)

    vehicles = VehicleParams()

    # Add a few vehicles to initialize the simulation. The vehicles have all lane changing enabled,
    # which is mode 1621
    vehicles.add(
        veh_id="human",
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=25,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=1621,
        ),
        num_vehicles=1)

    # These are additional params that configure the bottleneck experiment. They are explained in more
    # detail below.
    additional_env_params = {
        "target_velocity": 40,
        "max_accel": 1,
        "max_decel": 1,
        "lane_change_duration": 5,
        "add_rl_if_exit": False,
        "disable_tb": disable_tb,
        "disable_ramp_metering": disable_ramp_meter,
        "n_crit": n_crit,
        "feedback_coeff": feedback_coef,
    }
    # Set up the experiment to run for 1000 time steps i.e. 500 seconds (1000 * 0.5)
    env_params = EnvParams(
        horizon=1000, additional_params=additional_env_params)

    # Add vehicle inflows at the front of the bottleneck. They enter with a flow_rate number of vehicles
    # per hours and with a speed of 10 m/s
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehsPerHour=flow_rate,
        departLane="random",
        departSpeed=10)

    # Initialize the traffic lights. The meanings of disable_tb and disable_ramp_meter are discussed later.
    traffic_lights = TrafficLightParams()
    if not disable_tb:
        traffic_lights.add(node_id="2")
    if not disable_ramp_meter:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": scaling, "speed_limit": 23}
    net_params = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"])

    flow_params = dict(
        exp_tag='try_domain',
        env_name=BottleneckEnv,
        network=BottleneckNetwork,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
        tls=traffic_lights,
    )

    # number of time steps
    flow_params['env'].horizon = 1000
    exp = Experiment(flow_params)

    # run the sumo simulation
    _ = exp.run(1)


def run_exp_figure8():
    from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS

    # time horizon of a single rollout
    HORIZON = 1500
    ADDITIONAL_NET_PARAMS["speed_limit"] = 12

    # We place 1 autonomous vehicle and 13 human-driven vehicles in the network
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            max_speed=12
        ),
        num_vehicles=13)
    # vehicles.add(
    #     veh_id="rl",
    #     acceleration_controller=(RLController, {}),
    #     routing_controller=(ContinuousRouter, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         speed_mode="obey_safe_speed",
    #     ),
    #     num_vehicles=1)

    fig8_params = dict(
        # name of the experiment
        exp_tag="figure_eight",

        # name of the flow environment the experiment is running on
        env_name=AccelEnv,

        # name of the network class the experiment is running on
        network=FigureEightNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=False,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 5,
                "max_accel": 3,
                "max_decel": 3,
                "sort_vehicles": False
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )

    fig8_params_visual = dict(
        # name of the experiment
        exp_tag="figure_eight",

        # name of the flow environment the experiment is running on
        env_name=AccelEnv,

        # name of the network class the experiment is running on
        network=FigureEightNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 5,
                "max_accel": 3,
                "max_decel": 3,
                "sort_vehicles": False
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )

    exp = Experiment(fig8_params_visual)
    exp.run(1)


def run_exp_merge():
    from flow.networks.merge import ADDITIONAL_NET_PARAMS

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
    # vehicles.add(
    #     veh_id="rl",
    #     acceleration_controller=(RLController, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         speed_mode="obey_safe_speed",
    #     ),
    #     num_vehicles=0)

    # Vehicles are introduced from both sides of merge, with RL vehicles entering
    # from the highway portion as well
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE * 2,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="rl",
        edge="inflow_highway",
        vehs_per_hour=RL_PENETRATION * FLOW_RATE * .4,

        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=500,
        departLane="free",
        departSpeed=1)

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

    merge_params_visual = dict(
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
            render=True,
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

    exp = Experiment(merge_params_visual)
    exp.run(1)


def run_exp_i94():
    from icct.sumo_envs.i94 import i94_params_visual
    exp = Experiment(i94_params_visual)
    exp.run(1)


sumo_home = os.environ.get("SUMO_HOME", "")
if sumo_home:
    os.environ["PATH"] = os.path.join(sumo_home, "bin") + ":" + os.environ["PATH"]
# run_exp_bottleneck(flow_rate=1000, scaling=1, disable_tb=True, disable_ramp_meter=True)
# run_exp_merge()
run_exp_i94()

from flow.envs import TestEnv

# the Experiment class is used for running simulations
from flow.core.experiment import Experiment

# all other imports are standard
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams

from flow.networks import Network


net_params = NetParams(
    osm_path='networks/bay_bridge.osm'
)
# create the remainding parameters
env_params = EnvParams()
sim_params = SumoParams(render=True)
initial_config = InitialConfig()
vehicles = VehicleParams()
vehicles.add('human', num_vehicles=100)

flow_params = dict(
    exp_tag='bay_bridge',
    env_name=TestEnv,
    network=Network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

# number of time steps
flow_params['env'].horizon = 1000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1)


# we define an EDGES_DISTRIBUTION variable with the edges within
# the westbound Bay Bridge
EDGES_DISTRIBUTION = [
    "11197898",
    # "123741311",
    # "123741303",
    # "90077193#0",
    # "90077193#1",
    # "340686922",
    # "236348366",
    # "340686911#0",
    # "340686911#1",
    # "340686911#2",
    # "340686911#3",
    # "236348361",
    # "236348360#0",
    # "236348360#1"
]

# the above variable is added to initial_config
new_initial_config = InitialConfig(
    edges_distribution=EDGES_DISTRIBUTION
)



# we create a new network class to specify the expected routes
class BayBridgeOSMNetwork(Network):

    def specify_routes(self, net_params):
        return {
            "11197898": [
                "11197898", "123741311", "123741303", "90077193#0", "90077193#1",
                "340686922", "236348366", "340686911#0", "340686911#1",
                "340686911#2", "340686911#3", "236348361", "236348360#0", "236348360#1",
            ],
            "123741311": [
                "123741311", "123741303", "90077193#0", "90077193#1", "340686922",
                "236348366", "340686911#0", "340686911#1", "340686911#2",
                "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "123741303": [
                "123741303", "90077193#0", "90077193#1", "340686922", "236348366",
                "340686911#0", "340686911#1", "340686911#2", "340686911#3", "236348361",
                "236348360#0", "236348360#1"
            ],
            "90077193#0": [
                "90077193#0", "90077193#1", "340686922", "236348366", "340686911#0",
                "340686911#1", "340686911#2", "340686911#3", "236348361", "236348360#0",
                "236348360#1"
            ],
            "90077193#1": [
                "90077193#1", "340686922", "236348366", "340686911#0", "340686911#1",
                "340686911#2", "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "340686922": [
                "340686922", "236348366", "340686911#0", "340686911#1", "340686911#2",
                "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "236348366": [
                "236348366", "340686911#0", "340686911#1", "340686911#2", "340686911#3",
                "236348361", "236348360#0", "236348360#1"
            ],
            "340686911#0": [
                "340686911#0", "340686911#1", "340686911#2", "340686911#3", "236348361",
                "236348360#0", "236348360#1"
            ],
            "340686911#1": [
                "340686911#1", "340686911#2", "340686911#3", "236348361", "236348360#0",
                "236348360#1"
            ],
            "340686911#2": [
                "340686911#2", "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "340686911#3": [
                "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "236348361": [
                "236348361", "236348360#0", "236348360#1"
            ],
            "236348360#0": [
                "236348360#0", "236348360#1"
            ],
            "236348360#1": [
                "236348360#1"
            ]
        }


flow_params = dict(
    exp_tag='bay_bridge',
    env_name=TestEnv,
    network=BayBridgeOSMNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=new_initial_config,
)

# number of time steps
flow_params['env'].horizon = 10000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1)

#############################################################
####### Replace this with the environment you created #######
#############################################################
from flow.envs import i94 as myEnv

import json
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter


# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2


# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name=myEnv,  # <------ here we replace the environment with our new environment

    # name of the network class the experiment is running on
    network=RingNetwork,

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
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "target_velocity": 20,
            "sort_vehicles": False,
            "max_accel": 1,
            "max_decel": 1,
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


def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [3, 3]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


alg_run, gym_name, config = setup_exps()
ray.init(num_cpus=N_CPUS + 1)
trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 20,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": 200,
        },
    }
})





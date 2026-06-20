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
    # osm_path='/home/rohanpaleja/Downloads/map_correct_region.osm'
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
    "365215817#0",
    "365215817#1",
    "517478545#0",
    "517478545#1",
    "517479714#0",
    "517479714#1",
    "517479714#2",
]

# the above variable is added to initial_config
new_initial_config = InitialConfig(
    edges_distribution=EDGES_DISTRIBUTION
)

# we create a new network class to specify the expected routes
class BayBridgeOSMNetwork(Network):

    def specify_routes(self, net_params):
        return {
            "365215817#0": [
                "517478545", "517478544", "62476049", "517479714"
            ],
            "365215817#1": [
                "123741311", "123741303", "90077193#0", "90077193#1", "340686922",
                "236348366", "340686911#0", "340686911#1", "340686911#2",
                "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "517478545#0": [
                "123741303", "90077193#0", "90077193#1", "340686922", "236348366",
                "340686911#0", "340686911#1", "340686911#2", "340686911#3", "236348361",
                "236348360#0", "236348360#1"
            ],
            "517478545#1": [
                "90077193#0", "90077193#1", "340686922", "236348366", "340686911#0",
                "340686911#1", "340686911#2", "340686911#3", "236348361", "236348360#0",
                "236348360#1"
            ],
            "517479714#0": [
                "90077193#1", "340686922", "236348366", "340686911#0", "340686911#1",
                "340686911#2", "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "517479714#1": [
                "340686922", "236348366", "340686911#0", "340686911#1", "340686911#2",
                "340686911#3", "236348361", "236348360#0", "236348360#1"
            ],
            "517479714#2": [
                "236348366", "340686911#0", "340686911#1", "340686911#2", "340686911#3",
                "236348361", "236348360#0", "236348360#1"
            ], # done
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
    network=Network,
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











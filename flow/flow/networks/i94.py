"""Contains the merge network class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 30  # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5

ADDITIONAL_NET_PARAMS = {
    # length of the merge and exit edge
    "Ramp_Length": 100,
    # length of the highway leading to the merge
    "Pre_Merge_Length": 100,
    # length of the merging runway
    "Merge_Runway_Length": 100,
    # length of the highway past the merge
    "Post_Merge_Length": 200,
    # length of the highway past the exit
    "Post_Exit_Length": 150,
    # number of lanes in the merge
    "merge_lanes": 1,
    # number of lanes in the highway
    "highway_lanes": 3,
    # number of ramp in the network
    "number_ramp": 3,
    # max speed limit of the network
    "speed_limit": 30,
}


class I94Network(Network):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a merge network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        angle = pi / 4
        ramp_length = net_params.additional_params["Ramp_Length"]
        pre_merge_length = net_params.additional_params["Pre_Merge_Length"]
        merge_runway_length = net_params.additional_params["Merge_Runway_Length"]
        post_merge_length = net_params.additional_params["Post_Merge_Length"]
        post_exit_length = net_params.additional_params["Post_Exit_Length"]
        number_ramp = net_params.additional_params["number_ramp"]

        nodes = [
            {
                "id": "Inflow_Highway_Point",
                "x": -INFLOW_EDGE_LEN,
                "y": 0
            },
            {
                "id": "Left_Point",
                "y": 0,
                "x": 0
            },
            {
                "id": "Right_Point",
                "y": 0,
                "x": pre_merge_length + (merge_runway_length + post_merge_length + post_exit_length) * number_ramp
            },
        ]
        base_x = pre_merge_length
        for i in range(1, number_ramp + 1):
            nodes.append({
                "id": f"Merge_Point_{i}",
                "y": 0,
                "x": base_x,
                "radius": 10
            })
            nodes.append({
                "id": f"Merge_Finish_Point_{i}",
                "y": 0,
                "x": base_x + merge_runway_length,
                "type": "priority",
                "radius": 10
            })
            nodes.append({
                "id": f"Exit_Point_{i}",
                "y": 0,
                "x": base_x + merge_runway_length + post_merge_length,
                "radius": 10
            })
            nodes.append({
                "id": f"Bottom_Point_{i}",
                "y": -ramp_length * sin(angle),
                "x": base_x - ramp_length * cos(angle)
            })
            nodes.append({
                "id": f"Inflow_Ramp_Point_{i}",
                "y": -(ramp_length + INFLOW_EDGE_LEN) * sin(angle),
                "x": base_x - (ramp_length + INFLOW_EDGE_LEN) * cos(angle)
            })
            nodes.append({
                "id": f"Exit_Finish_Point_{i}",
                "x": base_x + merge_runway_length + post_merge_length + ramp_length * cos(angle),
                "y": -ramp_length * sin(angle)
            })
            base_x += merge_runway_length + post_merge_length + post_exit_length

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        ramp_length = net_params.additional_params["Ramp_Length"]
        pre_merge_length = net_params.additional_params["Pre_Merge_Length"]
        merge_runway_length = net_params.additional_params["Merge_Runway_Length"]
        post_merge_length = net_params.additional_params["Post_Merge_Length"]
        post_exit_length = net_params.additional_params["Post_Exit_Length"]
        number_ramp = net_params.additional_params["number_ramp"]

        edges = [{
            "id": "Inflow_Highway",
            "type": "highwayType",
            "from": "Inflow_Highway_Point",
            "to": "Left_Point",
            "length": INFLOW_EDGE_LEN
        }, {
            "id": "Pre_Merge",
            "type": "highwayType",
            "from": "Left_Point",
            "to": "Merge_Point_1",
            "length": pre_merge_length
        }]
        for i in range(1, number_ramp + 1):
            edges.append({
                "id": f"Runway_{i}",
                "type": "runwayType",
                "from": f"Merge_Point_{i}",
                "to": f"Merge_Finish_Point_{i}",
                "length": merge_runway_length
            })
            edges.append({
                "id": f"After_Merge_{i}",
                "type": "highwayType",
                "from": f"Merge_Finish_Point_{i}",
                "to": f"Exit_Point_{i}",
                "length": post_merge_length
            })
            if i == number_ramp:
                edges.append({
                    "id": f"After_Exit_{i}",
                    "type": "highwayType",
                    "from": f"Exit_Point_{i}",
                    "to": "Right_Point",
                    "length": post_exit_length
                })
            else:
                edges.append({
                    "id": f"After_Exit_{i}",
                    "type": "highwayType",
                    "from": f"Exit_Point_{i}",
                    "to": f"Merge_Point_{i+1}",
                    "length": post_exit_length
                })
            edges.append({
                "id": f"Bottom_{i}",
                "type": "rampType",
                "from": f"Bottom_Point_{i}",
                "to": f"Merge_Point_{i}",
                "length": ramp_length
            })
            edges.append({
                "id": f"Spawn_{i}",
                "type": "rampType",
                "from": f"Inflow_Ramp_Point_{i}",
                "to": f"Bottom_Point_{i}",
                "length": INFLOW_EDGE_LEN
            })
            edges.append({
                "id": f"Exit_{i}",
                "type": "rampType",
                "from": f"Exit_Point_{i}",
                "to": f"Exit_Finish_Point_{i}",
                "length": ramp_length
            })

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        h_lanes = net_params.additional_params["highway_lanes"]
        m_lanes = net_params.additional_params["merge_lanes"]
        speed = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": h_lanes,
            "speed": speed
        }, {
            "id": "rampType",
            "numLanes": m_lanes,
            "speed": speed
        }, {
            "id": "runwayType",
            "numLanes": h_lanes + m_lanes,
            "speed": speed
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        number_ramp = net_params.additional_params["number_ramp"]

        default_route = ["Inflow_Highway", "Pre_Merge"]
        for i in range(1, number_ramp + 1):
            default_route.extend([f"Runway_{i}", f"After_Merge_{i}", f"After_Exit_{i}"])

        rts = {}

        # for vehicles from Inflow_Highway, they may take any exit or highway_exit
        possible_routes = []
        probability_for_each_exit = 0.2
        for j in range(1, number_ramp + 1):
            default_route_end_idx = default_route.index(f"After_Exit_{j}")
            possible_routes.append((default_route[:default_route_end_idx] + [f"Exit_{j}"], probability_for_each_exit))
        possible_routes.append((default_route, 1 - probability_for_each_exit * number_ramp))
        rts["Inflow_Highway"] = possible_routes

        for idx, start_edge in enumerate(default_route[1:]):
            rts[start_edge] = default_route[idx:]

        for i in range(1, number_ramp + 1):
            default_route_idx = default_route.index(f"Runway_{i}")

            # for vehicles from Inflow_Ramp, they may take any exit or highway_exit
            possible_routes = []
            for j in range(i, number_ramp + 1):
                default_route_end_idx = default_route.index(f"After_Exit_{j}")
                possible_routes.append(([f"Spawn_{i}", f"Bottom_{i}"] + default_route[default_route_idx:default_route_end_idx] + [f"Exit_{j}"], probability_for_each_exit))
            possible_routes.append(([f"Spawn_{i}", f"Bottom_{i}"] + default_route[default_route_idx:], 1 - probability_for_each_exit * (number_ramp - i + 1)))
            rts[f"Spawn_{i}"] = possible_routes

            rts[f"Bottom_{i}"] = [f"Bottom_{i}"] + default_route[default_route_idx:]
            rts[f"Exit_{i}"] = [f"Exit_{i}"]

            if i == 1:
                rts["rl"] = possible_routes[-2][0]

        return rts

    # def specify_edge_starts(self):
    #     """See parent class."""
    #     premerge = self.net_params.additional_params["pre_merge_length"]
    #     postmerge = self.net_params.additional_params["post_merge_length"]
    #     merge_runway_length = self.net_params.additional_params["merge_runway_length"]
    #     postexit = self.net_params.additional_params["post_exit_length"]
    #
    #     edgestarts = [("inflow_highway", 0), ("left", INFLOW_EDGE_LEN),
    #                   ("merge_0", INFLOW_EDGE_LEN + premerge),
    #                   ("merge_finish_0", INFLOW_EDGE_LEN + premerge + merge_runway_length),
    #                   ("exit_0", INFLOW_EDGE_LEN + premerge + merge_runway_length + postmerge),
    #                   ("inflow_merge_0",
    #                    INFLOW_EDGE_LEN + premerge + merge_runway_length + postmerge + postexit),
    #                   ("bottom",
    #                    2 * INFLOW_EDGE_LEN + premerge + merge_runway_length + postmerge + postexit),
    #                   ]
    #
    #     return edgestarts

    # def specify_internal_edge_starts(self):
    #     """See parent class."""
    #     premerge = self.net_params.additional_params["pre_merge_length"]
    #     postmerge = self.net_params.additional_params["post_merge_length"]
    #
    #     internal_edgestarts = [
    #         (":left", INFLOW_EDGE_LEN), (":center",
    #                                      INFLOW_EDGE_LEN + premerge + 0.1),
    #         (":bottom", 2 * INFLOW_EDGE_LEN + premerge + postmerge + 22.6)
    #     ]
    #
    #     return internal_edgestarts

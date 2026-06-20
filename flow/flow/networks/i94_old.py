"""Contains the Bay Bridge network class."""

from flow.networks.base import Network

# Use this to ensure that vehicles are only placed in the edges of the Bay
# Bridge moving from Oakland to San Francisco.
EDGES_DISTRIBUTION = [
    "1046877960",
    "517479714",
]


class BayBridgeNetwork(Network):
    """A network used to simulate the Bay Bridge.

    The bay bridge was originally imported from OpenStreetMap and subsequently
    modified to more closely match the network geometry of the actual Bay
    Bridge. Vehicles are only allowed to exist of and traverse the edges
    leading up to and which the westbound Bay Bridge.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import BayBridgeNetwork
    >>>
    >>> network = BayBridgeNetwork(
    >>>     name='bay_bridge',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams()
    >>> )
    """

    def specify_routes(self, net_params):
        """See parent class.

        Routes for vehicles moving through the bay bridge from Oakland to San
        Francisco.
        """
        rts = {
            "517479714": [
                "517479714",
                "1046877960",
            ],
            "1046877960": [
                "1046877960",
                "124433729#0",
                "124433729#1",
            ],
            "124433729#0": ["124433729#0","124433729#1"],
            "124433729#1": ["124433729#1"]
        }

        return rts

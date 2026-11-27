"""
Publishes the world of an avocado's pieces on ROS 2, so that another process, such as
KnowRob, can follow the pieces, what they are and where they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.node import Node
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

from fruit_world import FruitWorld

# %% publishing the world


@dataclass(kw_only=True)
class FruitWorldPublisher:
    """
    Publishes a :class:`FruitWorld` on a ROS 2 node for as long as it lives: every
    change of its world, the pieces and their annotations included, on the world's
    synchronization topic, a frame for each body on ``tf``, and the whole world to
    whoever asks the fetch service, so a process may join it at any time.
    """

    fruit_world: FruitWorld
    """
    The world of the pieces.
    """

    node: Node
    """
    The node the world is published on.
    """

    synchronizer: WorldSynchronizer = field(init=False)
    """
    Publishes every change of the world.
    """

    tf_publisher: TFPublisher = field(init=False)
    """
    Publishes the frames of the world's bodies.
    """

    fetch_world_server: FetchWorldServer = field(init=False)
    """
    Serves the whole world to a process joining it.
    """

    def __post_init__(self) -> None:
        """
        Start publishing the world.
        """
        world = self.fruit_world.world
        self.synchronizer = WorldSynchronizer(node=self.node, _world=world)
        self.tf_publisher = TFPublisher(node=self.node, _world=world)
        self.fetch_world_server = FetchWorldServer(node=self.node, world=world)

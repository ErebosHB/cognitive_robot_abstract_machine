"""
Runs a demonstration with the solid avocado while publishing the world of its pieces on
ROS 2, for KnowRob or any other process to follow, and keeps publishing the pieces it
ends with until Enter is pressed.
"""

from __future__ import annotations

import argparse

from semantic_digital_twin.adapters.ros.background_node import (
    get_node,
    init_node,
    shutdown,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TfTopic

from demonstrations import Demonstration
from fruit_world_publisher import FruitWorldPublisher

# %% entry point


def main(
    demonstration: Demonstration,
    headless: bool,
    playback_speed: float,
    pit_seed: int | None,
    firmness: float | None = None,
    node_name: str = "solid_avocado",
) -> None:
    """
    Run a demonstration while its world of pieces is published on ROS 2.

    :param demonstration: The demonstration.
    :param headless: Whether to run without a viewer.
    :param playback_speed: Simulated seconds shown per second in the viewer.
    :param pit_seed: The seed of the avocado's random pit; None draws one.
    :param firmness: How firm the avocado is, in newtons; None draws a firmness from the
        seed.
    :param node_name: The name of the ROS 2 node the world is published on.
    """
    init_node(node_name)
    running = demonstration.create(
        headless=headless,
        playback_speed=playback_speed,
        pit_seed=pit_seed,
        firmness=firmness,
    )
    publisher = FruitWorldPublisher(fruit_world=running.fruit_world, node=get_node())
    print(
        f"Publishing the avocado's world on the node {node_name}: its changes on "
        f"{publisher.synchronizer.topic_name}, the whole world on the service "
        f"{publisher.fetch_world_server.service_name}, and its frames on "
        f"{publisher.tf_publisher.tf_topic} and {TfTopic.STATIC}."
    )
    running.run()
    input(
        f"The {demonstration} demonstration has ended with "
        f"{len(running.fruit.pieces)} pieces. Press Enter to stop publishing them."
    )
    shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a solid avocado demonstration and publish its pieces on ROS 2."
    )
    parser.add_argument(
        "demonstration",
        type=Demonstration,
        choices=list(Demonstration),
        help="the demonstration to run",
    )
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="simulated seconds shown per second; 1 is real time, 0.5 half speed",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="the seed of the random pit"
    )
    parser.add_argument(
        "--firmness",
        type=float,
        default=None,
        help="how firm the avocado is in newtons, from 200 (hard) to 7 (overripe)",
    )
    arguments = parser.parse_args()
    main(
        arguments.demonstration,
        arguments.headless,
        arguments.speed,
        arguments.seed,
        arguments.firmness,
    )

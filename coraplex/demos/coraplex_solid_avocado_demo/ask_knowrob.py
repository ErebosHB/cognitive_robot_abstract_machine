"""
Joins the avocado's world that ``publish_on_ros.py`` publishes, keeps KnowRob knowing
what its pieces are while they are cut, and answers the KnowRob queries typed in, one
per line, until an empty line or the end of the input.
"""

from __future__ import annotations

import sys

from semantic_digital_twin.adapters.knowrob.bridge import KnowRobBridge, KnowRobSession
from semantic_digital_twin.adapters.ros.background_node import (
    get_node,
    init_node,
    shutdown,
)
from semantic_digital_twin.adapters.ros.world_fetcher import join_shared_world


# %% entry point


def main(node_name: str = "knowrob_bridge") -> None:
    """
    Join the published world, bridge it to KnowRob and answer queries typed in.

    :param node_name: The name of the ROS 2 node the world is joined on.
    """
    init_node(node_name)
    world = join_shared_world(get_node())
    bridge = KnowRobBridge(_world=world, session=KnowRobSession.start())
    print(
        "KnowRob follows the avocado's pieces. Type a query per line, for example\n"
        "  rdf:type(?piece, sdt:'AvocadoPiece')\n"
        "  dul:hasPart(?piece, ?pit), rdf:type(?pit, sdt:'Pit')\n"
        "  rdf:type(?food, sdt:'Food'), knowrob:frameName(?food, ?frame)\n"
        "  dul:hasRegion(?avocado, ?ripeness), rdf:type(?ripeness, sdt:'Ripe')\n"
        "  soma:hasDisposition(?part, ?d), rdf:type(?d, soma:'Cuttability')\n"
        "  soma:isLinkedTo(?piece, ?other)\n"
        "and an empty line to stop."
    )
    for line in sys.stdin:
        query = line.strip()
        if not query:
            break
        answers = bridge.session.ask(query=query)
        for answer in answers:
            print(
                "  " + ", ".join(f"{name} = {value}" for name, value in answer.items())
            )
        print(f"{len(answers)} answers.")
    bridge.stop()
    shutdown()


if __name__ == "__main__":
    main()

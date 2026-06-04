import argparse

import rclpy


from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from interactive_viz_marker import InteractiveVizMarkerPublisher


def main():
    parser = argparse.ArgumentParser(
        description="Lade und visualisiere eine URDF-Datei."
    )
    parser.add_argument("urdf_path", type=str, help="Absoluter Pfad zur URDF-Datei")
    args = parser.parse_args()

    urdf_path = args.urdf_path

    print(f"Lade URDF von: {urdf_path}")
    robot_world = URDFParser.from_file(urdf_path).parse()

    # (Optional) You can uncomment this to view the kinematic tree in a popup window
    # robot_world.visualize_world_structure().show()

    # 2. Setup the semantic world with a designated "map" root
    world = World()
    map_root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(map_root)

    world.merge_world(robot_world)

    # 3. Setup ROS2 and publish markers/TF so it can be seen in RViz2
    rclpy.init()
    node = rclpy.create_node("urdf_visualizer")

    # VizMarkerPublisher publishes the meshes/collision shapes and TF frames
    viz = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
    interactive_viz = InteractiveVizMarkerPublisher(_world=world, node=node)

    print(
        "URDF loaded and being published! Open RViz2 to review it. Press Ctrl+C to stop."
    )
    try:
        # Keep the node alive to continuously broadcast TF frames
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

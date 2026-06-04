import rclpy
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
)
from interactive_markers import InteractiveMarkerServer
from semantic_digital_twin.world import World
from rclpy.node import Node


class InteractiveVizMarkerPublisher:
    def __init__(self, _world: World, node: Node):
        self._world = _world
        self.node = node
        self.server = InteractiveMarkerServer(self.node, "interactive_markers")

        for joint in self._world.joints:
            if joint.type == "revolute" or joint.type == "prismatic":
                self.create_interactive_marker(joint)

    def create_interactive_marker(self, joint):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = joint.parent.name.name
        int_marker.name = joint.name.name
        int_marker.description = joint.name.name

        # Create a control that allows rotation around the joint axis
        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        control.name = "move_axis"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS

        int_marker.controls.append(control)

        self.server.insert(int_marker, self.feedback_callback)
        self.server.applyChanges()

    def feedback_callback(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            joint_name = feedback.marker_name
            joint = self._world.get_joint_by_name(joint_name)
            if joint:
                # This is a simplified example. The actual joint update logic
                # might be more complex depending on your URDF and world representation.
                # Here we just print the pose. In a real application, you would
                # update the joint state in the world model.
                self.node.get_logger().info(
                    f"Joint {joint_name} moved to pose: {feedback.pose.position}"
                )

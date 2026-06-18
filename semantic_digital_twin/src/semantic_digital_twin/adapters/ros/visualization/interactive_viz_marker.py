import math
from dataclasses import field, dataclass

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
)

from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    RevoluteConnection,
    PrismaticConnection,
)


@dataclass
class InteractiveVizMarkerPublisher:
    _world: World
    node: Node
    server: InteractiveMarkerServer = field(init=False)
    update_timer: object = field(init=False)
    joint_pub: object = field(init=False)
    joint_timer: object = field(init=False)

    def __post_init__(self):
        self.server = InteractiveMarkerServer(self.node, "interactive_markers")

        marker_count = 0
        for connection in self._world.connections:
            if isinstance(connection, RevoluteConnection) or isinstance(
                connection, PrismaticConnection
            ):
                int_marker = self.create_interactive_marker(connection)
                self.server.insert(int_marker, feedback_callback=self.process_feedback)
                marker_count += 1

        self.server.applyChanges()
        print(
            f"[INFO] InteractiveMarkerServer initialized with {marker_count} markers."
        )

        # Update-Timer für die interaktiven Marker (ca. 30 FPS für flüssige Bedienung)
        self.update_timer = self.node.create_timer(0.033, self.server.applyChanges)

        # Eigener Publisher für die Joint-States, damit RViz die Bewegung flüssig rendert
        self.joint_pub = self.node.create_publisher(JointState, "joint_states", 10)
        self.joint_timer = self.node.create_timer(0.033, self.publish_joint_states)

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()

        for connection in self._world.connections:
            if isinstance(connection, RevoluteConnection) or isinstance(
                connection, PrismaticConnection
            ):
                try:
                    # connection.position (sichtbarer Wert mit multiplier/offset)
                    # statt des rohen DOF-Werts. RViz dreht das Mesh um genau diesen
                    # Wert um die URDF-Achse; bei einer gespiegelten Doppeltür
                    # (multiplier=-1) öffnet der zweite Flügel so in die richtige
                    # Richtung, konsistent mit dem Setzen in process_feedback.
                    current_position = connection.position
                    msg.name.append(str(connection.name))
                    msg.position.append(float(current_position))
                except KeyError:
                    pass

        self.joint_pub.publish(msg)

    def create_interactive_marker(
        self, connection: ActiveConnection1DOF
    ) -> InteractiveMarker:
        int_marker = InteractiveMarker()

        # Frame-ID direkt vom Parent übernehmen (enthält bereits "kitchen/")
        int_marker.header.frame_id = str(connection.parent.name)

        int_marker.name = str(connection.name)
        int_marker.description = f"Control for {str(connection.name)}"

        pose = PoseToRos2Converter.convert(connection.origin.to_pose())
        int_marker.pose = pose

        control = InteractiveMarkerControl()

        # Richte die Kontroll-Achse (Standard: +X) auf die TATSÄCHLICHE Gelenkachse
        # aus der URDF aus. Dadurch dreht/schiebt der Griff immer entlang der echten
        # Achse des Gelenks (z.B. +Z links, -Z rechts) und ist nicht an eine fixe
        # Richtung oder an den Namen des Schranks gebunden.
        ax, ay, az = self._axis_xyz(connection)
        qw, qx, qy, qz = self._control_orientation_for_axis(ax, ay, az)
        control.orientation.w = qw
        control.orientation.x = qx
        control.orientation.y = qy
        control.orientation.z = qz

        if isinstance(connection, RevoluteConnection):
            control.name = "rotate_axis"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        else:
            control.name = "move_axis"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS

        int_marker.controls.append(control)
        return int_marker

    def _axis_xyz(self, connection: ActiveConnection1DOF):
        """Liefert die (normierte) Gelenkachse als (x, y, z) direkt aus der URDF."""
        flat = [float(v) for v in connection.axis.to_np().flatten().tolist()]
        ax, ay, az = flat[0], flat[1], flat[2]
        norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        return ax / norm, ay / norm, az / norm

    @staticmethod
    def _control_orientation_for_axis(ax, ay, az):
        """
        Quaternion (w, x, y, z), das die Standard-Kontrollachse (+X) auf die
        Gelenkachse (ax, ay, az) dreht. So liegt der Griff immer auf der echten
        Achse des Gelenks.
        """
        fx, fy, fz = 1.0, 0.0, 0.0  # Standard-Kontrollachse von RViz
        dot = fx * ax + fy * ay + fz * az
        if dot > 0.999999:  # bereits entlang +X
            return 1.0, 0.0, 0.0, 0.0
        if dot < -0.999999:  # entgegengesetzt zu +X -> 180° um Z
            return 0.0, 0.0, 0.0, 1.0
        # Rotationsachse = X x Achse, Winkel = arccos(dot)
        cx = fy * az - fz * ay
        cy = fz * ax - fx * az
        cz = fx * ay - fy * ax
        cn = math.sqrt(cx * cx + cy * cy + cz * cz) or 1.0
        cx, cy, cz = cx / cn, cy / cn, cz / cn
        angle = math.acos(max(-1.0, min(1.0, dot)))
        s = math.sin(angle / 2.0)
        return math.cos(angle / 2.0), cx * s, cy * s, cz * s

    def process_feedback(self, feedback: InteractiveMarkerFeedback):
        # Akzeptiere MOUSE_UP (Loslassen) ODER POSE_UPDATE (flüssiges Ziehen)
        valid_events = [
            InteractiveMarkerFeedback.MOUSE_UP,
            InteractiveMarkerFeedback.POSE_UPDATE,
        ]

        if feedback.event_type in valid_events:
            connection_name = feedback.marker_name
            # String-Matching-Fix (löst den PyCRAM-Typ-Mismatch-Fehler)
            connection = next(
                (c for c in self._world.connections if str(c.name) == connection_name),
                None,
            )

            if connection:
                with self._world.modify_world():
                    if isinstance(connection, RevoluteConnection):
                        try:
                            # 1. Gelenkachse direkt aus der URDF holen (z.B. +Z / -Z).
                            ax, ay, az = self._axis_xyz(connection)

                            # 2. Vorzeichenbehafteten Drehwinkel des Markers UM DIESE
                            #    Achse aus dem Feedback-Quaternion berechnen. Da der
                            #    Griff auf der Gelenkachse liegt, ergibt das direkt die
                            #    DOF-Position (positiv = öffnen in URDF-Richtung).
                            q = feedback.pose.orientation
                            dot_va = q.x * ax + q.y * ay + q.z * az
                            angle_rad = 2.0 * math.atan2(dot_va, q.w)

                            # 3. Limits direkt aus der URDF/dem DOF lesen -
                            #    KEIN Raten über Gelenknamen mehr.
                            dof = connection.dof
                            lower_limit = dof.limits.lower.position
                            upper_limit = dof.limits.upper.position

                            # 4. Auf die tatsächlichen Limits einklemmen (falls gesetzt).
                            if lower_limit is not None:
                                angle_rad = max(lower_limit, angle_rad)
                            if upper_limit is not None:
                                angle_rad = min(upper_limit, angle_rad)

                            # 5. Auf den Digital Twin anwenden.
                            #    connection.position (statt state[dof_id]) berücksichtigt
                            #    multiplier/offset. Bei einer Doppeltür mit geteiltem DOF
                            #    und gespiegeltem (negativem) multiplier öffnet so jeder
                            #    Flügel in seine eigene, korrekte Richtung.
                            connection.position = angle_rad

                        except Exception as e:
                            print(
                                f"[ERROR] Fehler bei der Tür-Rotation ({connection.name}): {e}"
                            )

                    elif isinstance(connection, PrismaticConnection):
                        try:
                            # Verschiebung des Markers auf die ECHTE Gelenkachse
                            # projizieren, statt eine feste Z-Achse anzunehmen.
                            ax, ay, az = self._axis_xyz(connection)
                            start_pose = PoseToRos2Converter.convert(
                                connection.origin.to_pose()
                            )
                            dx = feedback.pose.position.x - start_pose.position.x
                            dy = feedback.pose.position.y - start_pose.position.y
                            dz = feedback.pose.position.z - start_pose.position.z
                            displacement = dx * ax + dy * ay + dz * az

                            # Auch hier: echte Limits aus dem DOF verwenden.
                            dof = connection.dof
                            lower_limit = dof.limits.lower.position

                            upper_limit = dof.limits.upper.position
                            if lower_limit is not None:
                                displacement = max(lower_limit, displacement)
                            if upper_limit is not None:
                                displacement = min(upper_limit, displacement)

                            connection.position = displacement
                        except Exception as e:
                            print(
                                f"[ERROR] Fehler bei der Schublade ({connection.name}): {e}"
                            )

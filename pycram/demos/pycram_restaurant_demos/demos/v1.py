import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from time import sleep
import pycram.external_interfaces.robokudo
import semantic_digital_twin
from pycram_suturo_demos.helper_methods_and_useful_classes.nlp_human_robot_interaction import (
    TalkingNode,
)
from pycram_suturo_demos.helper_methods_and_useful_classes.waving_detection import (
    ContinuousWavingDetector,
)
from geometry_msgs.msg import PointStamped
from pycram.external_interfaces import nav2_move, robokudo
from pycram.datastructures.enums import Arms
from pycram.external_interfaces.nlp_interface import NlpInterface, FilterOptions
from pycram.plans.factories import sequential
from pycram.robot_plans import ParkArmsActionDescription, LookAtActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from pycram_restaurant_demos.demos.world_setup import world, robot_view, context


import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TtsPublisher:
    def __init__(self, node_name="tts_text_publisher", topic_name="/tts_text"):
        self.node = Node(node_name)
        self.publisher = self.node.create_publisher(String, topic_name, 10)

    def pub(self, text: str):
        msg = String()
        msg.data = text
        self.publisher.pub(msg)
        self.node.get_logger().info(f"Published: '{text}'")
        rclpy.spin_once(self.node, timeout_sec=0.1)

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()


logging.getLogger(semantic_digital_twin.world.__name__).setLevel(logging.WARN)

MIN_DISTANCE_M: float = 0.8
STOP_DISTANCE_M: float = 0.8
WAVING_TIMEOUT_PER_DIRECTION: float = 10.0
BUFFER_SWITCH = False

bartender_pose: Optional[PointStamped] = None
customer_poses: list[PointStamped] = []


class Direction(Enum):
    LEFT = [0.1, 1, 1]
    RIGHT = [0.1, -1, 1]
    BACK = [-1, 0, 1]
    FRONT = [1, 0, 1]
    FRONT_SOFA = [1, 0, 0.65]
    FRONT_DOWN = [1, 0, 0.5]



def transform_perception_to_map(perception_point: PointStamped) -> HomogeneousTransformationMatrix:
    pose_in_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(perception_point.point.x),
        y=float(perception_point.point.y),
        z=float(perception_point.point.z),
        reference_frame=world.get_body_by_name("head_rgbd_sensor_link"),
    )
    return world.transform(pose_in_camera, world.root)



def look_in_direction(direction: Direction):
    look_at_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=direction.value[0],
        y=direction.value[1],
        z=direction.value[2],
        reference_frame=robot_view.root,
    )
    look_at_pose_in_map = world.transform(look_at_pose, world.root)
    sequential(
        [LookAtActionDescription([look_at_pose_in_map.to_pose()])],
        context=context,
    ).plan.perform()


def _pose_distance(a: PointStamped, b: PointStamped) -> float:
    return math.sqrt((a.point.x - b.point.x) ** 2 + (a.point.y - b.point.y) ** 2)


def find_bartender_by_waving() -> Optional[PointStamped]:
    global bartender_pose
    talking = TalkingNode()
    talking.pub("Bartender, please wave your hand so I can identify you.", delay=3)
    bartender_pose = scan_for_waving_human()
    return bartender_pose


def scan_for_waving_human():
    detector = ContinuousWavingDetector(retry_interval=1.0)
    s_human = detector.wait_for_waving_human(timeout=WAVING_TIMEOUT_PER_DIRECTION)
    for direction in [
        Direction.FRONT,
        Direction.RIGHT,
        Direction.BACK,
        Direction.LEFT,
        Direction.FRONT,
    ]:
        if s_human is not None:
            break
        look_in_direction(direction)
        s_human = detector.wait_for_waving_human(timeout=WAVING_TIMEOUT_PER_DIRECTION)
    look_in_direction(Direction.FRONT)
    return s_human

def scan_for_waving_customers() -> list[PointStamped]:
    global customer_poses
    customer_poses = []
    detector = ContinuousWavingDetector(retry_interval=1.0)
    for direction in [
        Direction.FRONT,
        Direction.RIGHT,
        Direction.BACK,
        Direction.LEFT,
    ]:
        look_in_direction(direction)
        pose = detector.wait_for_waving_human(timeout=WAVING_TIMEOUT_PER_DIRECTION)
        if pose is None:
            continue
        if bartender_pose is not None and _pose_distance(pose, bartender_pose) < MIN_DISTANCE_M:
            continue
        customer_poses.append(pose)
    look_in_direction(Direction.FRONT)
    return customer_poses


def _approach_pose(target_pose: HomogeneousTransformationMatrix) -> HomogeneousTransformationMatrix:
    robot_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.0, y=0.0, z=0.0, reference_frame=robot_view.root
    )
    robot_in_map = world.transform(robot_origin, world.root)
    robot_geom = robot_in_map.to_pose()
    target_geom = target_pose.to_pose()

    dx = target_geom.position.x - robot_geom.position.x
    dy = target_geom.position.y - robot_geom.position.y
    dist = math.sqrt(dx ** 2 + dy ** 2)

    if dist <= STOP_DISTANCE_M:
        return target_pose

    scale = (dist - STOP_DISTANCE_M) / dist
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=robot_geom.position.x + dx * scale,
        y=robot_geom.position.y + dy * scale,
        z=0.0,
        reference_frame=world.root,
    )


def drive_to_pose(target_pose: HomogeneousTransformationMatrix):
    sequential(
        [NavigateAction(target_pose.to_pose())],
        context=context,
    ).plan.perform()



def drive_to_customer(customer_point: PointStamped):
    map_pose = transform_perception_to_map(customer_point)
    drive_to_pose(_approach_pose(map_pose))


@dataclass
class CustomerOrder:
    customer_point: PointStamped
    items: list[str] = field(default_factory=list)


class RestaurantNlpInterface(NlpInterface):
    def filter_response(self, response: list, filter_for: FilterOptions):
        return super().filter_response(response, filter_for)

    def get_order_items(self) -> list[str]:
        items = []
        for response in self.all_last_outputs:
            entities = response[2] if response and len(response) > 2 else []
            for entity in entities:
                value = entity[1]
                if value is not None:
                    items.append(value)
        return items


customer_orders: list[CustomerOrder] = []


def take_order(customer_point: PointStamped) -> CustomerOrder:
    nlp = RestaurantNlpInterface()
    nlp.tts.pub("Hello! What would you like to order? Please tell me your drinks and food.", delay=5)
    nlp.input_confirmation_loop(tries=3)
    order = CustomerOrder(
        customer_point=customer_point,
        items=nlp.get_order_items(),
    )
    customer_orders.append(order)
    return order


def _order_to_speech(order: CustomerOrder) -> str:
    return ", ".join(order.items) if order.items else "nothing"


def drive_to_bartender():
    if bartender_pose is None:
        return
    map_pose = transform_perception_to_map(bartender_pose)
    drive_to_pose(_approach_pose(map_pose))


def report_order_to_bartender(order: CustomerOrder):
    talking = TalkingNode()
    talking.pub(
        f"The customer ordered {_order_to_speech(order)}.",
        delay=5,
    )


def deliver_order_to_bartender(order: CustomerOrder):
    drive_to_bartender()
    report_order_to_bartender(order)


def take_orders(total: int = 3):
    while len(customer_orders) < total:
        customer_point = scan_for_waving_human()
        if customer_point is None:
            continue
        drive_to_customer(customer_point)
        order = take_order(customer_point)
        deliver_order_to_bartender(order)

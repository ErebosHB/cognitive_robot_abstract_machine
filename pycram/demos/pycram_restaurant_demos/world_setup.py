import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.ros import world_fetcher, world_synchronizer
from semantic_digital_twin.robots.hsrb import HSRB
from suturo_resources.suturo_map_rody import load_environment

rclpy.init()
node = rclpy.create_node("restaurant_demo")

_executor = SingleThreadedExecutor()
_executor.add_node(node)
_executor_thread = threading.Thread(target=_executor.spin, daemon=True, name="rclpy-executor")
_executor_thread.start()
time.sleep(0.1)

world = world_fetcher.fetch_world_from_service(node)
world_synchronizer.StateSynchronizer(world, node)
world_synchronizer.ModelSynchronizer(world, node)

_environment = load_environment()
with world.modify_world():
    world.merge_world(_environment)

robot_view = HSRB.from_world(world)
context = Context.from_world(world)
context.ros_node = node
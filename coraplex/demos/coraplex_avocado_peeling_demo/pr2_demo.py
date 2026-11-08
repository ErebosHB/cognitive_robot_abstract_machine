"""
The PR2 puts an avocado on a cutting board in the kitchen of the IAI apartment and peels
a strip of its skin, in MuJoCo.

The avocado starts on the kitchen island beside the board. The PR2 stands at the island,
takes the avocado with its right hand from above and lays it on the board, stem to the
robot's right. Its left hand then holds the fruit down at its base end, on the left. The
right hand pinches the tip, at the stem, of the upper half's strip facing the robot, and
pulls it out toward the robot and away from the fruit, tearing the strip off; the rest of the skin stays
on. The right hand carries the peeled strip to the island beside the board and lays it
down there.

The PR2 is the one of the semantic digital twin, exported to MuJoCo when the
demonstration starts; its meshes come from ``iai_pr2_description``. The avocado is seed 0
of the peelable avocado of ``private_meshes``, which comes with CRAM. Run it from this
folder::

    python pr2_demo.py

or pass another avocado's folder, or set ``AVOCADO_DIRECTORY`` to it.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from demo import (
    AvocadoPeelingDemonstration,
    SceneConstraint,
    avocado_folder,
)
from exceptions import (
    ArmsCollided,
    FruitNotPlaced,
    NeighbouringStripsTorn,
    PeelNotPlaced,
    SkinTornAtRest,
    StripNotPeeled,
)
from peeling import FruitSide, SkinTile
from pr2 import ArmSide, Pr2Arm, Pr2Export, Pr2Name

# %% motions


@dataclass
class ToolWaypoint:
    """
    A pose the tool frame of a PR2 hand passes.
    """

    position: np.ndarray
    """
    Where the point between the fingertips goes, in world coordinates.
    """

    travel_time: float
    """
    How long in seconds the hand takes to get there from the previous waypoint.
    """


@dataclass
class Pr2PeelingOutcome:
    """
    What the PR2 achieved.
    """

    placement_error: float
    """
    How far in metres from the board's centre the avocado came to rest.
    """

    peeled_strip: str
    """
    The name of the strip that was pulled.
    """

    peeled_tiles: list[str]
    """
    The names of the strip's tiles that came off while it was peeled.
    """

    peel_placement_error: float
    """
    How far in metres from where it was laid down, on the island, the farthest piece of
    the peeled strip came to rest.
    """


# %% the demonstration


@dataclass
class Pr2AvocadoPeelingDemonstration(AvocadoPeelingDemonstration):
    """
    The PR2 puts the avocado from the kitchen island onto the cutting board and peels
    one strip of its skin, checking that each stage ends as it should.
    """

    countertop_spot: tuple[float, float] = (2.47, 2.45)
    """
    The point (x, y) in metres on the kitchen island, in front of the PR2, that the
    cutting board is centred on.
    """

    fruit_heading: float = -math.pi / 2
    """
    The direction in radians, about the vertical from world x, in which the stem of the
    lying fruit points: to the PR2's right, so each hand works on the end on its own
    side, the left holding the base end and the right peeling from the stem.
    """

    peeled_side: tuple[float, float, float] = (-1.0, 0.0, 1.0)
    """
    The direction, in world axes, the peeled strip faces: up and toward the PR2, which
    the right hand pulls it out to, away from the holding hand.
    """

    fruit_start_offset: tuple[float, float] = (0.0, -0.33)
    """
    Where on the kitchen island, relative to the board's centre, the avocado starts,
    in metres: to the PR2's right of the board.
    """

    viewer_distance: float = 1.2
    """
    How far in metres the viewer's camera starts from the cutting board: far enough to
    see the PR2's arms.
    """

    viewer_azimuth: float = 150.0
    """
    The direction in degrees the viewer's camera starts looking in: from across the
    island, to the PR2's right.
    """

    robot_position: tuple[float, float] = (1.86, 2.45)
    """
    Where in metres the PR2's base stands, in front of the kitchen island and facing it.
    """

    torso_height: float = 0.3
    """
    How far in metres the PR2's torso is raised.
    """

    head_tilt: float = 0.9
    """
    How far in radians the PR2 tilts its head down, to look at the board.
    """

    robot_prefix: str = "pr2/"
    """
    The prefix of the PR2's names in the scene.
    """

    initial_arm_joints: dict[ArmSide, tuple[float, ...]] = field(
        default_factory=lambda: {
            ArmSide.RIGHT: (-0.7, -0.2, -0.5, -1.8, 0.0, -1.0, 0.0),
            ArmSide.LEFT: (0.7, -0.2, 0.5, -1.8, 0.0, -1.0, 0.0),
        }
    )
    """
    The arm joint positions, shoulder to wrist, the PR2 starts with: hands up in front
    of it, clear of the island.
    """

    open_gripper: float = 0.5
    """
    The finger angle in radians of an open gripper.
    """

    holding_clearance: float = 0.004
    """
    How far in metres above the skin's tiles at the top of the fruit the left hand's
    closed fingertips stop to hold it down.
    """

    finger_force: float = 1.0
    """
    The most force in newtons a finger closes with; enough to hold, gentle on the skin.
    """

    pinch_opening: float = 0.12
    """
    The finger angle in radians the right hand comes in with to pinch the skin.
    """

    pinch_collision_bit: int = 128
    """
    The contact bit, unused by the scene, by which the pinching fingers touch only the
    pinched strip.
    """

    retreat_distance: float = 0.06
    """
    How far in metres the right hand backs off toward the stem, and up, after letting go
    of the strip.
    """

    peel_drop_offset: tuple[float, float] = (0.0, -0.27)
    """
    Where on the kitchen island, relative to the board's centre, the peeled strip is
    laid down, in metres: beside the board, to the PR2's right.
    """

    peel_drop_height: float = 0.02
    """
    How far in metres above the island the fingertips let go of the peeled strip.
    """

    peel_spread_margin: float = 0.03
    """
    How much farther in metres than the strip is long any piece of the peeled strip may
    come to rest from where it was laid down, on the island; the strip hangs from its
    pinched tip, so it lies out from there.
    """

    peeled_strip_length: float = field(init=False, default=0.0)
    """
    The length in metres of the strip being peeled, measured on the fruit.
    """

    ready_offset: float = 0.25
    """
    How far in metres to its own side of the board a hand waits while the other works.
    """

    hover_height: float = 0.08
    """
    How far in metres above a grasp the hand comes in from and leaves to.
    """

    reach_time: float = 1.5
    """
    How long in seconds a hand takes to get above a grasp.
    """

    approach_time: float = 0.8
    """
    How long in seconds a hand takes to go down onto a grasp, or up from it.
    """

    carry_time: float = 2.0
    """
    How long in seconds the right hand takes to carry the avocado to the board.
    """

    gripper_time: float = 0.6
    """
    How long in seconds a gripper takes to open or close.
    """

    control_period: int = 5
    """
    How many simulation steps pass between two solutions of the inverse kinematics.
    """

    holding_inset: float = 0.015
    """
    How far in metres from the fruit's base end the left hand holds it.
    """

    peeling_waypoints: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.012, 0.0, 0.0),
            (0.02, 0.01, 0.0),
            (0.035, 0.018, 0.0),
            (0.055, 0.022, 0.0),
            (0.08, 0.026, 0.0),
            (0.11, 0.03, 0.0),
            (0.14, 0.033, 0.0),
            (0.17, 0.036, 0.0),
            (0.2, 0.04, 0.0),
        ]
    )
    """
    The points the right hand passes after pinching the skin, as distances in metres out
    from the skin, along the fruit toward its base and down, from the pinched tile: a
    little along the fruit, then mostly out from it, away from the holding hand, while
    the strip tears off behind the fingers.
    """

    peeling_time: float = 1.0
    """
    How long in seconds the right hand takes between two peeling waypoints.
    """

    minimum_peeled_share: float = 0.6
    """
    The share of the pulled strip's tiles that must come off; the base end of the strip
    stays under the holding hand.
    """

    maximum_placement_error: float = 0.05
    """
    How far in metres from the board's centre the avocado may come to rest.
    """

    pinched_geom_ids: np.ndarray | None = field(init=False, default=None)
    """
    The ids of the tile geoms of the strip being pinched, while it is.
    """

    left_arm_geom_ids: np.ndarray = field(init=False)
    """
    The ids of the geoms of the left arm.
    """

    geom_arm_side: np.ndarray = field(init=False)
    """
    For every geom of the scene, -1 if it belongs to the left arm, 1 if to the right
    arm, 0 otherwise.
    """

    arms: dict[ArmSide, Pr2Arm] = field(init=False)
    """
    The PR2's arms in the compiled scene.
    """

    model_directory: tempfile.TemporaryDirectory = field(init=False)
    """
    The folder the PR2 is exported into for the run.
    """

    # %% scene

    def fruit_start(self) -> np.ndarray:
        """
        :return: Where the fruit's origin starts: on the island beside the board.
        """
        dx, dy = self.fruit_start_offset
        return super().fruit_start() + [dx, dy, 0.0]

    def fruit_resting_height(self) -> float:
        """
        :return: The height of the kitchen island, which the fruit starts on.
        """
        return self.board_top - self.cutting_board_size[2]

    def add_hands(self, spec: mujoco.MjSpec) -> list[tuple[SceneConstraint, str]]:
        """
        Put the PR2 in front of the kitchen island, facing it; its right hand grips, its
        left hand holds.

        :param spec: The scene.
        :return: Each weld the hands take hold with, and the hand's body it starts on.
        """
        self.model_directory = tempfile.TemporaryDirectory(prefix="pr2_avocado_")
        export = Pr2Export()
        robot = export.standing_spec(export.write(Path(self.model_directory.name)))
        x, y = self.robot_position
        spec.attach(
            robot,
            prefix=self.robot_prefix,
            frame=spec.worldbody.add_frame(pos=[x, y, 0.0]),
        )
        return [
            (
                SceneConstraint.GRIP,
                f"{self.robot_prefix}{ArmSide.RIGHT}{Pr2Name.TOOL_FRAME}",
            ),
            (
                SceneConstraint.HOLD,
                f"{self.robot_prefix}{ArmSide.LEFT}{Pr2Name.TOOL_FRAME}",
            ),
        ]

    def prepare_hands(self) -> None:
        """
        Raise the torso, tilt the head toward the board and bring the arms up, open.
        """
        for joint, position in (
            (Pr2Name.TORSO, self.torso_height),
            (Pr2Name.HEAD_TILT, self.head_tilt),
        ):
            name = self.robot_prefix + joint
            self.data.qpos[self.model.jnt_qposadr[self.joint_id(name)]] = position
            self.data.ctrl[self.actuator_id(name + Pr2Name.SERVO_SUFFIX)] = position
        self.arms = {
            side: Pr2Arm(self.model, self.data, side, self.robot_prefix)
            for side in ArmSide
        }
        arm_prefixes = {
            f"{self.robot_prefix}{ArmSide.LEFT}_": -1,
            f"{self.robot_prefix}{ArmSide.RIGHT}_": 1,
        }
        self.geom_arm_side = np.array(
            [
                next(
                    (
                        side
                        for prefix, side in arm_prefixes.items()
                        if self.model.body(
                            self.model.geom_bodyid[geom]
                        ).name.startswith(prefix)
                    ),
                    0,
                )
                for geom in range(self.model.ngeom)
            ]
        )
        self.left_arm_geom_ids = np.flatnonzero(self.geom_arm_side == -1)
        for side, arm in self.arms.items():
            arm.place_joints(np.array(self.initial_arm_joints[side]))
            arm.set_gripper(self.open_gripper)
            self.model.actuator_forcerange[arm.finger_actuator_id] = [
                -self.finger_force,
                self.finger_force,
            ]
        mujoco.mj_forward(self.model, self.data)

    def joint_id(self, name: str) -> int:
        """
        :param name: The full name of a joint of the scene.
        :return: Its id.
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def actuator_id(self, name: str) -> int:
        """
        :param name: The full name of an actuator of the scene.
        :return: Its id.
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    # %% the run

    def run(self) -> Pr2PeelingOutcome:
        """
        :return: What the PR2 achieved.
        """
        self.build_scene()
        try:
            with self.viewer_context():
                torn_tiles = self.simulate(self.rest_duration)
                if torn_tiles:
                    raise SkinTornAtRest([t.description.body for t in torn_tiles])
                placement_error = self.put_avocado_on_board()
                self.hold_avocado()
                strip, peeled_tiles = self.peel_strip_facing_robot()
                peel_placement_error = self.put_peel_aside(strip)
                self.linger()
        finally:
            self.model_directory.cleanup()
        return Pr2PeelingOutcome(
            placement_error, strip, peeled_tiles, peel_placement_error
        )

    def after_step(self) -> None:
        """
        Keep the strip being pinched touchable by the fingers, also once its tiles tore
        off, and check that the arms stay clear of each other.

        :raises ArmsCollided: If a body of one arm touches a body of the other.
        """
        if self.pinched_geom_ids is not None:
            self.model.geom_conaffinity[self.pinched_geom_ids] |= (
                self.pinch_collision_bit
            )
        geoms = self.data.contact.geom[: self.data.ncon]
        sides = self.geom_arm_side[geoms]
        touching = np.flatnonzero(sides[:, 0] * sides[:, 1] == -1)
        if len(touching):
            first, second = geoms[touching[0]]
            left, right = sorted(
                (first, second), key=lambda geom: self.geom_arm_side[geom]
            )
            raise ArmsCollided(
                self.model.body(self.model.geom_bodyid[left]).name,
                self.model.body(self.model.geom_bodyid[right]).name,
                float(self.data.time),
            )

    # %% stages

    def put_avocado_on_board(self) -> float:
        """
        Take the avocado from above with the right hand, carry it over the board, lay it
        down there, let go of it and take the hand out of the way.

        :return: How far in metres from the board's centre the avocado came to rest.
        """
        right = self.arms[ArmSide.RIGHT]
        held_id = self.fruit_body(self.fruit.description.pit_side, 1)
        grasp = self.fruit_centre()
        torn = self.reach_down_to(right, grasp)
        torn += self.close_gripper(right, 0.0)
        self.attach(SceneConstraint.GRIP, right, held_id)
        x, y = self.countertop_spot
        drop = np.array([x, y, grasp[2] + self.cutting_board_size[2] + 0.004])
        torn += self.move_tool(
            right,
            [
                ToolWaypoint(grasp + [0, 0, self.hover_height], self.approach_time),
                ToolWaypoint(drop + [0, 0, self.hover_height], self.carry_time),
                ToolWaypoint(drop, self.approach_time),
            ],
        )
        self.release_weld(SceneConstraint.GRIP)
        torn += self.close_gripper(right, self.open_gripper)
        torn += self.move_tool(
            right,
            [
                ToolWaypoint(drop + [0, 0, self.hover_height], self.approach_time),
                ToolWaypoint(self.ready_position(ArmSide.RIGHT), self.reach_time),
            ],
        )
        if torn:
            raise NeighbouringStripsTorn([t.description.body for t in torn])
        placement_error = float(
            np.linalg.norm(self.fruit_centre()[:2] - np.array([x, y]))
        )
        if placement_error > self.maximum_placement_error:
            raise FruitNotPlaced(placement_error)
        return placement_error

    def hold_avocado(self) -> None:
        """
        Hold the avocado down at its base end with the left hand: the closed fingertips
        come down onto the top of the skin, so nothing of the hand reaches round to the
        side being peeled.
        """
        left = self.arms[ArmSide.LEFT]
        axis = self.fruit_axis_now()
        point = self.fruit_end(toward_stem=False) + self.holding_inset * axis
        tile_ids = [
            self.body_id(self.fruit_prefix + t.description.body)
            for t in self.fruit.tiles
        ]
        tiles = self.data.xipos[tile_ids]
        nearby = np.abs((tiles - point) @ axis) < self.holding_inset / 2
        point[2] = tiles[nearby, 2].max() + self.holding_clearance
        torn = self.reach_down_to(left, point, gripper=0.0)
        self.attach(SceneConstraint.HOLD, left, self.upper_half_ids()[-1])
        if torn:
            raise NeighbouringStripsTorn([t.description.body for t in torn])

    def peel_strip_facing_robot(self) -> tuple[str, list[str]]:
        """
        Pinch the tip of the upper half's strip facing the PR2 at the stem between the
        fingertips, and draw it back along the fruit toward the base, at a low angle so
        the strip comes off a little at a time; the hand keeps holding the strip.

        :return: The name of the peeled strip and of its tiles that came off.
        """
        right = self.arms[ArmSide.RIGHT]
        facing = np.array(self.peeled_side)
        # A strip of the half the left hand holds.
        pinched = max(
            (
                tile
                for tile in self.fruit.tiles
                if tile.description.index == 0
                and tile.flesh_body_id in self.upper_half_ids()
            ),
            key=lambda tile: self.side_normal(tile) @ facing,
        )
        strip = pinched.description.strip
        pinched_id = self.body_id(self.fruit_prefix + pinched.description.body)
        start = self.data.xipos[pinched_id].copy()
        outward = self.side_normal(pinched)
        toward_base = -self.fruit_axis_now()
        strip_ids = [
            self.body_id(self.fruit_prefix + t.description.body)
            for t in sorted(self.fruit.tiles, key=lambda t: t.description.index)
            if t.description.strip == strip
        ]
        self.peeled_strip_length = float(
            np.linalg.norm(np.diff(self.data.xipos[strip_ids], axis=0), axis=1).sum()
        )
        # The fingers come in from the side the strip faces, across the fruit's axis, and
        # close across the strip. The tips of all strips meet at the stem, so while pinching the fingers
        # touch only the pinched strip.
        rotation = self.pinch_grasp(outward, toward_base)
        self.pinched_geom_ids = np.array(
            [
                tile.geom_id
                for tile in self.fruit.tiles
                if tile.description.strip == strip
            ]
        )
        right.set_hand_collisions(self.pinch_collision_bit)
        # The other arm stays solid to the pinching hand, so the arms still cannot meet.
        self.model.geom_conaffinity[self.left_arm_geom_ids] |= self.pinch_collision_bit
        right.set_gripper(self.pinch_opening)
        torn = self.move_tool(
            right,
            [
                ToolWaypoint(start + self.hover_height * outward, self.reach_time),
                ToolWaypoint(start, self.approach_time),
            ],
            rotation,
        )
        torn += self.close_gripper(right, 0.0)
        self.attach(SceneConstraint.GRIP, right, pinched_id)
        # The weld holds the strip now: the fingers stay where they closed to and stop
        # pressing on its light tip, which they would otherwise fight the weld over.
        right.set_gripper(right.gripper_opening())
        self.model.geom_conaffinity[self.pinched_geom_ids] &= ~self.pinch_collision_bit
        self.pinched_geom_ids = None
        torn += self.move_tool(
            right,
            [
                ToolWaypoint(
                    start + out * outward + along * toward_base - [0.0, 0.0, down],
                    self.peeling_time,
                )
                for out, along, down in self.peeling_waypoints
            ],
            rotation,
        )
        peeled = [t.description.body for t in torn if t.description.strip == strip]
        others = [t.description.body for t in torn if t.description.strip != strip]
        strip_size = sum(t.description.strip == strip for t in self.fruit.tiles)
        if len(peeled) < self.minimum_peeled_share * strip_size:
            raise StripNotPeeled(strip, len(peeled), strip_size)
        if len(others) > self.maximum_neighbouring_tiles_torn:
            raise NeighbouringStripsTorn(others)
        return strip, peeled

    def put_peel_aside(self, strip: str) -> float:
        """
        Carry the peeled strip, still pinched, off the fruit, tearing off what is left
        of it, to :attr:`peel_drop_offset` beside the board, lay it down on the island
        and let go of it.

        :param strip: The name of the strip being held.
        :return: How far in metres from where it was laid down, on the island, the
            farthest piece of the strip came to rest.
        """
        right = self.arms[ArmSide.RIGHT]
        rotation = right.tool_pose()[1]
        toward_base = -self.fruit_axis_now()
        x, y = self.countertop_spot
        dx, dy = self.peel_drop_offset
        countertop = self.board_top - self.cutting_board_size[2]
        drop = np.array([x + dx, y + dy, countertop + self.peel_drop_height])
        held = right.tool_pose()[0]
        # Back toward the stem and up first, away from the holding hand at the base.
        torn = self.move_tool(
            right,
            [
                ToolWaypoint(
                    held
                    - self.retreat_distance * toward_base
                    + [0.0, 0.0, self.retreat_distance],
                    self.approach_time,
                )
            ],
            rotation,
        )
        # Turn the hand to point down on the way, which the arm can reach the spot with.
        torn += self.move_tool(
            right,
            [ToolWaypoint(drop + [0.0, 0.0, self.hover_height], self.carry_time)],
            rotation,
            turn_to=self.downward_grasp(),
        )
        torn += self.move_tool(right, [ToolWaypoint(drop, self.approach_time)])
        self.release_weld(SceneConstraint.GRIP)
        torn += self.close_gripper(right, self.open_gripper)
        torn += self.move_tool(
            right,
            [ToolWaypoint(drop + [0.0, 0.0, self.hover_height], self.approach_time)],
        )
        right.set_hand_collisions(None)
        self.model.geom_conaffinity[self.left_arm_geom_ids] &= ~self.pinch_collision_bit
        self.pinched_geom_ids = None
        torn += self.move_tool(
            right, [ToolWaypoint(self.ready_position(ArmSide.RIGHT), self.reach_time)]
        )
        others = [t.description.body for t in torn if t.description.strip != strip]
        if len(others) > self.maximum_neighbouring_tiles_torn:
            raise NeighbouringStripsTorn(others)
        strip_ids = [
            self.body_id(self.fruit_prefix + t.description.body)
            for t in self.fruit.tiles
            if t.description.strip == strip
        ]
        # The strip hangs from its pinched tip, so it lies out from where it was let go.
        on_island = np.array([drop[0], drop[1], countertop])
        distance = float(
            np.linalg.norm(self.data.xipos[strip_ids] - on_island, axis=1).max()
        )
        if distance > self.peeled_strip_length + self.peel_spread_margin:
            raise PeelNotPlaced(distance)
        return distance

    # %% moving the arms

    def ready_position(self, side: ArmSide) -> np.ndarray:
        """
        :param side: An arm.
        :return: Where its hand waits, out of the other hand's way: above the island,
            to its own side of the board.
        """
        x, y = self.countertop_spot
        sideways = -1.0 if side == ArmSide.RIGHT else 1.0
        return np.array(
            [x - 0.05, y + sideways * self.ready_offset, self.board_top + 0.2]
        )

    def reach_down_to(
        self, arm: Pr2Arm, point: np.ndarray, gripper: float | None = None
    ) -> list[SkinTile]:
        """
        Bring a hand, open and pointing down, above *point* and then down onto it.

        :param arm: The arm to move.
        :param point: Where the point between the fingertips goes.
        :param gripper: The finger angle on the way; :attr:`open_gripper` if None.
        :return: The skin tiles released meanwhile.
        """
        arm.set_gripper(self.open_gripper if gripper is None else gripper)
        return self.move_tool(
            arm,
            [
                ToolWaypoint(point + [0, 0, self.hover_height], self.reach_time),
                ToolWaypoint(point, self.approach_time),
            ],
        )

    def move_tool(
        self,
        arm: Pr2Arm,
        waypoints: list[ToolWaypoint],
        rotation: np.ndarray | None = None,
        turn_to: np.ndarray | None = None,
    ) -> list[SkinTile]:
        """
        Move a hand through waypoints in straight lines, by solving the inverse
        kinematics along the way.

        :param arm: The arm to move.
        :param waypoints: The poses to pass.
        :param rotation: The rotation the hand keeps, or starts from; pointing down with
            its fingers closing across the fruit if None.
        :param turn_to: The rotation the hand turns to evenly along the way, if any.
        :return: The skin tiles released meanwhile.
        """
        start, _ = arm.tool_pose()
        path = np.array([start, *(w.position for w in waypoints)])
        arrival_times = np.cumsum([0.0, *(w.travel_time for w in waypoints)])
        if rotation is None:
            rotation = self.downward_grasp()
        step = [0]

        def follow(elapsed: float) -> None:
            if step[0] % self.control_period == 0:
                target = np.array(
                    [np.interp(elapsed, arrival_times, path[:, a]) for a in range(3)]
                )
                turned = rotation
                if turn_to is not None:
                    turned = self.turned(
                        rotation, turn_to, min(1.0, elapsed / arrival_times[-1])
                    )
                arm.command(arm.solve(target, turned))
            step[0] += 1

        return self.simulate(arrival_times[-1] + self.settle_time, follow)

    @staticmethod
    def turned(start: np.ndarray, end: np.ndarray, share: float) -> np.ndarray:
        """
        :param start: A rotation matrix.
        :param end: Another rotation matrix.
        :param share: How far from *start* to *end*, 0 to 1.
        :return: The rotation that far along the shortest turn between them.
        """
        quaternions = []
        for matrix in (start, end):
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, matrix.flatten())
            quaternions.append(quaternion)
        first, last = quaternions
        if first @ last < 0:
            last = -last
        angle = math.acos(min(1.0, float(first @ last)))
        if angle < 1e-6:
            return end
        blend = (
            math.sin((1 - share) * angle) * first + math.sin(share * angle) * last
        ) / math.sin(angle)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, blend / np.linalg.norm(blend))
        return matrix.reshape(3, 3)

    def close_gripper(self, arm: Pr2Arm, opening: float) -> list[SkinTile]:
        """
        :param arm: The arm whose gripper moves.
        :param opening: The finger angle in radians to go to.
        :return: The skin tiles released while the gripper moved.
        """
        arm.set_gripper(opening)
        return self.simulate(self.gripper_time)

    def downward_grasp(self) -> np.ndarray:
        """
        :return: The tool rotation pointing the fingers down, closing them across the
            lying fruit.
        """
        approach = np.array([0.0, 0.0, -1.0])
        across = np.cross(self.fruit_axis(), approach)
        return np.column_stack([approach, across, np.cross(approach, across)])

    def pinch_grasp(self, outward: np.ndarray, along: np.ndarray) -> np.ndarray:
        """
        :param outward: The skin's outward normal where it is pinched.
        :param along: The direction along the strip.
        :return: The tool rotation pointing the fingers into the skin, closing them
            across the strip.
        """
        approach = -outward / np.linalg.norm(outward)
        across = np.cross(approach, along)
        across /= np.linalg.norm(across)
        return np.column_stack([approach, across, np.cross(approach, across)])

    def attach(self, constraint: SceneConstraint, arm: Pr2Arm, body_id: int) -> None:
        """
        Weld a body to a hand in the pose they are in now.

        :param constraint: The weld to switch on.
        :param arm: The hand.
        :param body_id: The body it takes hold of.
        """
        constraint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, constraint
        )
        self.model.eq_obj2id[constraint_id] = body_id
        hand_rotation = self.data.xmat[arm.tool_id].reshape(3, 3)
        relative_position = hand_rotation.T @ (
            self.data.xpos[body_id] - self.data.xpos[arm.tool_id]
        )
        inverse_hand = np.zeros(4)
        relative_quaternion = np.zeros(4)
        mujoco.mju_negQuat(inverse_hand, self.data.xquat[arm.tool_id])
        mujoco.mju_mulQuat(relative_quaternion, inverse_hand, self.data.xquat[body_id])
        # The weld's relative pose is that of its second body in the frame of its first.
        self.model.eq_data[constraint_id][3:10] = [
            *relative_position,
            *relative_quaternion,
        ]
        self.data.eq_active[constraint_id] = 1

    # %% the fruit where it is

    def side_normal(self, tile: SkinTile) -> np.ndarray:
        """
        :param tile: A tile of the fruit.
        :return: The direction out of the fruit at the tile, across the fruit's axis: at
            the tips of the strips the skin faces mostly along the axis, which would lay
            a hand coming in along it over the whole fruit.
        """
        axis = self.fruit_axis_now()
        normal = self.outward_normal(tile)
        across = normal - (normal @ axis) * axis
        return across / np.linalg.norm(across)

    def fruit_body(self, side: FruitSide, slice_index: int) -> int:
        """
        :param side: A half of the fruit.
        :param slice_index: The slice's position from the stem; negative from the base.
        :return: The id of the flesh slice.
        """
        name = self.fruit.description.flesh_slices[side][slice_index]
        return self.body_id(self.fruit_prefix + name)

    def flesh_ids(self) -> list[int]:
        """
        :return: The ids of all flesh slices.
        """
        return [
            self.body_id(self.fruit_prefix + name)
            for names in self.fruit.description.flesh_slices.values()
            for name in names
        ]

    def fruit_centre(self) -> np.ndarray:
        """
        :return: The fruit's centre of mass, in world coordinates.
        """
        ids = self.flesh_ids()
        masses = self.model.body_mass[ids]
        return masses @ self.data.xipos[ids] / masses.sum()

    def fruit_axis_now(self) -> np.ndarray:
        """
        :return: The direction from base to stem of the fruit as it lies now.
        """
        return self.data.xmat[self.flesh_ids()[0]].reshape(3, 3)[:, 2].copy()

    def fruit_end(self, toward_stem: bool) -> np.ndarray:
        """
        :param toward_stem: Whether the stem end is wanted, rather than the base end.
        :return: The point on the fruit's long axis at that end, in world coordinates.
        """
        axis = self.fruit_axis_now()
        centre = self.fruit_centre()
        points = self.data.xipos[
            [
                self.body_id(self.fruit_prefix + t.description.body)
                for t in self.fruit.tiles
            ]
        ]
        reach = (points - centre) @ axis
        return centre + axis * (reach.max() if toward_stem else reach.min())


# %% entry point


def main(
    avocado_directory: Path | None = None, headless: bool = False
) -> Pr2PeelingOutcome:
    """
    Run the demonstration and report what the PR2 achieved.

    :param avocado_directory: The avocado model's folder; defaults to the folder named
        by :attr:`EnvironmentVariable.AVOCADO_DIRECTORY`, or else the avocado that comes
        with CRAM.
    :param headless: Whether the physics runs without a viewer.
    :return: What the PR2 achieved.
    """
    outcome = Pr2AvocadoPeelingDemonstration(
        avocado_folder(avocado_directory), headless=headless
    ).run()
    print(
        f"The PR2 laid the avocado {outcome.placement_error * 1000:.0f} mm from the "
        f"board's centre and peeled {len(outcome.peeled_tiles)} tiles of strip "
        f"{outcome.peeled_strip}, then laid the strip down beside the board, all of it "
        f"within {outcome.peel_placement_error * 1000:.0f} mm of where it let go."
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The PR2 puts an avocado on a cutting board and peels it, in MuJoCo."
    )
    parser.add_argument(
        "avocado_directory",
        type=Path,
        nargs="?",
        help="the avocado model's folder; by default the one that comes with CRAM",
    )
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    arguments = parser.parse_args()
    main(arguments.avocado_directory, arguments.headless)

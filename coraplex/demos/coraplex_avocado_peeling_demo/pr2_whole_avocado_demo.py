"""
The PR2 puts an avocado on a cutting board in the kitchen of the IAI apartment and peels
all of its skin off, strip by strip, in MuJoCo.

The avocado starts on the kitchen island beside the board. The PR2 stands at the island,
takes the avocado with its right hand from above and lays it on the board, stem to the
robot's right. Its left hand picks the fruit up by its base end and holds it out above
the board. The right hand pinches the skin facing up near the held end, peels a strip
off toward the free end and lays it on the board; the left wrist then turns the fruit
until skin faces up again, and so on round the fruit. The left hand then lays the fruit
back down, the right hand turns it end for end, the left hand picks it up by its peeled
stem end, and the right hand peels the base end the same way.

The PR2 is the one of the semantic digital twin, exported to MuJoCo when the
demonstration starts; its meshes come from ``iai_pr2_description``. The avocado is seed 0
of the peelable avocado of ``private_meshes``, which comes with CRAM. Run it from this
folder::

    python pr2_whole_avocado_demo.py

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
    FruitFile,
    SceneConstraint,
    avocado_folder,
)
from exceptions import (
    ArmsCollided,
    FruitNotPlaced,
    NeighbouringStripsTorn,
    SkinTornAtRest,
    StripNotPeeled,
)
from peeling import FruitSide, PeelingDescription, SkinTile
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

    strip_lengths: list[int]
    """
    For each strip the PR2 peeled, how many tiles it held when it laid the strip down.
    """

    peeled_tile_count: int
    """
    How many of the avocado's skin tiles came off in all.
    """

    tile_count: int
    """
    How many skin tiles the avocado has.
    """


# %% the demonstration


@dataclass
class Pr2WholeAvocadoPeelingDemonstration(AvocadoPeelingDemonstration):
    """
    The PR2 puts the avocado from the kitchen island onto the cutting board and peels
    all of its skin, checking that each stage ends as it should.
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

    finger_force: float = 1.0
    """
    The most force in newtons a finger closes with; enough to hold, gentle on the skin.
    """

    pinch_tilt: float = math.radians(35.0)
    """
    How far in radians the right hand's approach leans from the skin's normal toward the
    fruit's base, so the hand stays on the stem side of the pinch, away from the hand
    holding the base end.
    """

    pinch_opening: float = 0.07
    """
    The finger angle in radians the right hand comes in with to pinch the skin.
    """

    pinch_collision_bit: int = 1 << 28
    """
    The contact bit, unused by the scene and the fruit, by which the pinching fingers
    touch only the pinched tile and the other arm.
    """

    strip_drop_offset: tuple[float, float] = (0.0, -0.11)
    """
    Where on the cutting board, relative to its centre, the right hand lays each peeled
    strip down, in metres: beside the held fruit, to the PR2's right.
    """

    strip_drop_height: float = 0.02
    """
    How far in metres above the board the fingertips let go of a peeled strip.
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

    base_grip_inset: float = 0.006
    """
    How far in metres in from the fruit's base end the left hand's fingertips take it.
    """

    holding_chunks: int = 12
    """
    How many of the fruit's heaviest flesh chunks the left hand is welded to while it
    holds the fruit.
    """

    hold_height: float = 0.1
    """
    How far in metres above the board the left hand holds the fruit while it is peeled.
    """

    roll_speed: float = math.radians(60.0)
    """
    How fast in radians per second the left hand turns the fruit about its axis.
    """

    rows_off_held_end: int = 10
    """
    How many rows of skin tiles, counted from the held end, the right hand keeps off:

    there its fingers would meet the holding ones. A strip is pinched at the first tile
    still on past them.
    """

    pinched_width: int = 3
    """
    How many neighbouring tiles of one row the fingertips take between them: a strip
    about as wide as the fingertips.
    """

    maximum_strips_per_end: int = 16
    """
    How many strips the right hand peels at most while the fruit is held by one end.
    """

    roll_margin: float = math.radians(30.0)
    """
    How far in radians past a full turn the left hand may turn the fruit while looking
    for skin facing up, before the end it holds is done.
    """

    minimum_strip_start: int = 3
    """
    How many tiles of a column must still be on for the right hand to peel there; the
    odd tile left over between strips is not worth a strip of its own.

    The cap at the tip has only a few rows.
    """

    minimum_useful_strip: int = 5
    """
    How many tiles must come off with a pinch for the right hand to try that column
    again: what is left of a column that came off in bits only snaps again.
    """

    maximum_tiles_lost_turning: int = 20
    """
    How many loose tiles, left over between strips, may come off while the fruit is
    turned end for end, and again while it is picked up by its other end.
    """

    drop_gap: float = 0.003
    """
    How far in metres above the board a hand lets go of the fruit it lays down.
    """

    turning_height: float = 0.12
    """
    How far in metres above the board the right hand lifts the fruit to turn it end for
    end.
    """

    pinching_lean: float = math.radians(15.0)
    """
    How far in radians from straight up toward the PR2 the skin the right hand pinches
    best faces.
    """

    roll_back_limit: float = math.radians(40.0)
    """
    How far in radians the left hand turns the fruit back at most, rather than on round.
    """

    minimum_facing_up: float = 0.8
    """
    How directly a tile must face :meth:`pinching_direction`, as the cosine of the angle
    between them, for the right hand to pinch it from above.
    """

    peeling_waypoints: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.015, 0.0),
            (0.035, 0.02),
            (0.05, 0.05),
            (0.05, 0.09),
            (0.05, 0.13),
            (0.04, 0.17),
            (0.03, 0.21),
        ]
    )
    """
    The points the right hand passes after pinching the skin, as distances in metres out
    from the skin and along the fruit toward its stem, away from the holding hand: off
    the skin, then back along the fruit, steeply enough to peel rather than drag.
    """

    peeling_time: float = 0.8
    """
    How long in seconds the right hand takes between two peeling waypoints.
    """

    minimum_peeled_share: float = 0.9
    """
    The share of the avocado's skin tiles that must have come off at the end.
    """

    maximum_placement_error: float = 0.05
    """
    How far in metres from the board's centre the avocado may come to rest.
    """

    held_at_stem: bool = field(init=False, default=False)
    """
    Whether the left hand holds the fruit by its stem end, rather than its base end.
    """

    row_widths: dict[int, int] = field(init=False, default_factory=dict)
    """
    How many skin tiles each row has, by row.
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
        # Further welds of the left hand, so it holds the fruit by many of its chunks:
        # one light chunk alone gives way under the whole fruit held out sideways.
        description = PeelingDescription.from_file(
            self.avocado_directory / FruitFile.PEELING_DESCRIPTION
        )
        # And of the right hand, so its fingertips take a strip several tiles wide.
        for side, names, body in (
            (ArmSide.LEFT, self.holding_weld_names(), description.flesh[0].body),
            (ArmSide.RIGHT, self.pinching_weld_names(), description.tiles[0].body),
        ):
            for name in names:
                weld = spec.add_equality(
                    type=mujoco.mjtEq.mjEQ_WELD,
                    name=name,
                    name1=f"{self.robot_prefix}{side}{Pr2Name.TOOL_FRAME}",
                    name2=self.fruit_prefix + body,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,
                    active=False,
                    solref=[0.004, 1],
                )
                weld.data[0:10] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
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
                self.pick_up_by_end(toward_stem=False)
                strip_lengths = self.peel_all_around()
                self.turn_end_for_end()
                # Loose tiles left between the strips may come off as it is taken again.
                self.pick_up_by_end(
                    toward_stem=True, allowed_losses=self.maximum_tiles_lost_turning
                )
                strip_lengths += self.peel_all_around()
                self.check_peeled()
                self.linger()
        finally:
            self.model_directory.cleanup()
        return Pr2PeelingOutcome(
            placement_error,
            strip_lengths,
            len(self.fruit.peeled_tiles()),
            len(self.fruit.tiles),
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
        held_id = self.heaviest_chunk(self.fruit.description.pit.side)
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

    def pick_up_by_end(self, toward_stem: bool, allowed_losses: int = 0) -> None:
        """
        Take the avocado by one end with the left hand, the fingers pointing along the
        fruit's axis and closing across it, and lift it above the board: turning the
        wrist then turns the fruit about its own axis.

        :param toward_stem: Whether to take it by the stem end, rather than the base
            end.
        :param allowed_losses: How many skin tiles may come off meanwhile.
        :raises NeighbouringStripsTorn: If more tiles came off.
        """
        left = self.arms[ArmSide.LEFT]
        self.held_at_stem = toward_stem
        inward = self.away_from_hand()
        grip = self.fruit_end(toward_stem) + self.base_grip_inset * inward
        rotation = self.along_axis_grasp(inward)
        left.set_gripper(self.open_gripper)
        torn = self.move_tool(
            left,
            [
                ToolWaypoint(grip - self.hover_height * inward, self.reach_time),
                ToolWaypoint(grip, self.approach_time),
            ],
            rotation,
        )
        torn += self.close_gripper(left, 0.0)
        # The hand holds the whole fruit rigidly: weld it, as it is now, to the fruit's
        # heaviest chunks, deep in the flesh. Held by one light chunk, the fruit would
        # swing down on it and sag on the welds between its chunks.
        heaviest = sorted(
            self.fruit.description.flesh, key=lambda chunk: chunk.mass, reverse=True
        )
        for name, chunk in zip(self.all_holding_weld_names(), heaviest):
            self.attach(name, left, self.body_id(self.fruit_prefix + chunk.body))
        lifted = grip.copy()
        lifted[2] = self.board_top + self.hold_height
        torn += self.move_tool(left, [ToolWaypoint(lifted, self.reach_time)], rotation)
        if len(torn) > allowed_losses:
            raise NeighbouringStripsTorn([t.description.body for t in torn])

    def peel_all_around(self) -> list[int]:
        """
        Peel the strip facing up off, lay it on the board, turn the fruit until skin
        faces up again and go on, once round the fruit.

        :return: For each strip, how many tiles it held when it was laid down.
        """
        strip_lengths = []
        tried: set[int] = set()
        given_up: set[tuple[int, int]] = set()
        rolled = 0.0
        while len(strip_lengths) < self.maximum_strips_per_end:
            candidates = [
                tile
                for tile in self.tiles_to_pinch()
                if tile.index not in tried
                and self.column_of(tile) not in given_up
                and self.skin_left_in_column(tile) >= self.minimum_strip_start
            ]
            if not candidates:
                break
            facing_up = [
                tile
                for tile in candidates
                if self.facing_pinch(tile) >= self.minimum_facing_up
            ]
            if not facing_up:
                # Turn the fruit by as little as brings skin up.
                angle = min(self.roll_to_pinch(tile) for tile in candidates)
                if rolled + abs(angle) > 2 * math.pi + self.roll_margin:
                    break
                self.roll_fruit(angle)
                rolled += abs(angle)
                continue
            # The column with the most skin left on, of those facing up.
            best = max(
                facing_up,
                key=lambda tile: (
                    self.skin_left_in_column(tile),
                    self.facing_pinch(tile),
                ),
            )
            tried.add(best.index)
            strip_lengths.append(self.peel_and_lay_down(best))
            if strip_lengths[-1] < self.minimum_useful_strip:
                given_up.add(self.column_of(best))
        return strip_lengths

    def check_peeled(self) -> None:
        """
        :raises StripNotPeeled: If too little of the skin came off in all.
        """
        peeled, total = len(self.fruit.peeled_tiles()), len(self.fruit.tiles)
        if peeled < self.minimum_peeled_share * total:
            raise StripNotPeeled(
                "the whole avocado",
                peeled,
                math.ceil(self.minimum_peeled_share * total),
            )

    def tiles_to_pinch(self) -> list[SkinTile]:
        """
        :return: For each column of tiles, the tile still on nearest the held end, past
            the :attr:`rows_off_held_end` rows the right hand keeps off, if any.
        """
        last_row = max(tile.description.row for tile in self.fruit.tiles)
        nearest: dict[tuple[int, int], SkinTile] = {}
        for tile in self.fruit.tiles:
            row = tile.description.row
            from_hand = row if self.held_at_stem else last_row - row
            if from_hand < self.rows_off_held_end or not self.fruit.is_attached(tile):
                continue
            key = self.column_of(tile)
            known = nearest.get(key)
            if known is None or from_hand < (
                known.description.row
                if self.held_at_stem
                else last_row - known.description.row
            ):
                nearest[key] = tile
        return list(nearest.values())

    def column_of(self, tile: SkinTile) -> tuple[int, int]:
        """
        :param tile: A tile of the fruit.
        :return: Its column, told apart from the columns of the rows with fewer tiles,
            which are counted separately: the width of its row and its column there.
        """
        return self.row_width(tile.description.row), tile.description.column

    def skin_left_in_column(self, tile: SkinTile) -> int:
        """
        :param tile: A tile of the fruit.
        :return: How many tiles of its column are still on.
        """
        column = self.column_of(tile)
        return sum(
            1
            for other in self.fruit.tiles
            if self.column_of(other) == column and self.fruit.is_attached(other)
        )

    def pinching_direction(self) -> np.ndarray:
        """
        :return: The direction across the held fruit's axis the right hand pinches skin
            facing best: up, leaning toward the PR2 by :attr:`pinching_lean`, so the hand
            comes in over the near side of the fruit, never reaching round its far side
            past the holding hand.
        """
        axis = self.away_from_hand()
        x, y = self.countertop_spot
        toward_robot = np.array([*(np.array(self.robot_position) - [x, y]), 0.0])
        toward_robot /= np.linalg.norm(toward_robot)
        direction = (
            math.cos(self.pinching_lean) * np.array([0.0, 0.0, 1.0])
            + math.sin(self.pinching_lean) * toward_robot
        )
        direction -= (direction @ axis) * axis
        return direction / np.linalg.norm(direction)

    def facing_pinch(self, tile: SkinTile) -> float:
        """
        :param tile: A tile of the fruit.
        :return: The cosine of the angle between the way the tile faces and
            :meth:`pinching_direction`.
        """
        return float(self.side_normal(tile) @ self.pinching_direction())

    def roll_to_pinch(self, tile: SkinTile) -> float:
        """
        :param tile: A tile of the fruit.
        :return: How far in radians the left wrist must turn the fruit for the tile to
            face :meth:`pinching_direction`: back by up to :attr:`roll_back_limit`, which
            straightens a tile the fruit's sag turned a little too far, or else forward.
        """
        axis = self.away_from_hand()
        wanted = self.pinching_direction()
        normal = self.side_normal(tile)
        # The wrist turns the hand, and the fruit with it, about the fingers' direction.
        angle = -math.atan2(
            float(np.cross(wanted, normal) @ axis), float(wanted @ normal)
        )
        return angle if angle > -self.roll_back_limit else angle + 2 * math.pi

    def row_width(self, row: int) -> int:
        """
        :param row: A row of skin tiles.
        :return: How many tiles the row has.
        """
        if not self.row_widths:
            for tile in self.fruit.tiles:
                self.row_widths[tile.description.row] = (
                    self.row_widths.get(tile.description.row, 0) + 1
                )
        return self.row_widths[row]

    def pinched_tiles(self, pinched: SkinTile) -> list[SkinTile]:
        """
        :param pinched: The tile in the middle of the fingertips.
        :return: It and, of the :attr:`pinched_width` tiles of its row around it, those
            still on.
        """
        row = pinched.description.row
        width = self.row_width(row)
        by_column = {
            tile.description.column: tile
            for tile in self.fruit.tiles
            if tile.description.row == row
        }
        offsets = [
            sign * (k // 2 + 1)
            for k, sign in zip(
                range(self.pinched_width - 1), [1, -1] * self.pinched_width
            )
        ]
        tiles = [pinched]
        for offset in offsets:
            tile = by_column[(pinched.description.column + offset) % width]
            if self.fruit.is_attached(tile) and tile not in tiles:
                tiles.append(tile)
        return tiles

    def peel_and_lay_down(self, pinched: SkinTile) -> int:
        """
        Pinch a tile and its neighbours with the right hand, peel the strip they start
        toward the free end of the fruit, lay the strip on the board beside the fruit
        and let go of it.

        :param pinched: The tile to pinch in the middle of the fingertips.
        :return: How many tiles the strip held when it was laid down.
        """
        right = self.arms[ArmSide.RIGHT]
        start = self.data.xipos[pinched.body_id].copy()
        outward = self.side_normal(pinched)
        away = self.away_from_hand()
        # The fingers come in from the side the tile faces, across the fruit's axis, and
        # close along the strip, touching only the pinched tiles and the other arm.
        rotation = self.pinch_grasp(outward, away)
        approach = rotation[:, 0]
        tiles = self.pinched_tiles(pinched)
        self.pinched_geom_ids = np.array(
            [geom for tile in tiles for geom in (tile.shell_id, tile.geom_id)]
        )
        right.set_hand_collisions(self.pinch_collision_bit)
        self.model.geom_conaffinity[self.left_arm_geom_ids] |= self.pinch_collision_bit
        right.set_gripper(self.pinch_opening)
        self.move_tool(
            right,
            [
                ToolWaypoint(start - self.hover_height * approach, self.reach_time),
                ToolWaypoint(start, self.approach_time),
            ],
            rotation,
        )
        self.close_gripper(right, 0.0)
        names = self.all_pinching_weld_names()
        for name, tile in zip(names, tiles):
            self.attach(name, right, tile.body_id)
        # The welds hold the strip now: the fingers stay where they closed to and stop
        # pressing on the light tiles, which they would otherwise fight the welds over.
        right.set_gripper(right.gripper_opening())
        self.model.geom_conaffinity[self.pinched_geom_ids] &= ~self.pinch_collision_bit
        self.pinched_geom_ids = None
        self.move_tool(
            right,
            [
                ToolWaypoint(start + out * outward + along * away, self.peeling_time)
                for out, along in self.peeling_waypoints
            ],
            rotation,
        )
        x, y = self.countertop_spot
        dx, dy = self.strip_drop_offset
        drop = np.array([x + dx, y + dy, self.board_top + self.strip_drop_height])
        # Turn the hand to point down on the way, which the arm reaches the board with.
        self.move_tool(
            right,
            [ToolWaypoint(drop + [0.0, 0.0, self.hover_height], self.reach_time)],
            rotation,
            turn_to=self.downward_grasp(),
        )
        self.move_tool(right, [ToolWaypoint(drop, self.approach_time)])
        strip = {
            carried.index for tile in tiles for carried in self.fruit.carried_with(tile)
        }
        for name in names:
            self.release_weld(name)
        self.close_gripper(right, self.open_gripper)
        self.move_tool(
            right,
            [ToolWaypoint(drop + [0.0, 0.0, self.hover_height], self.approach_time)],
        )
        right.set_hand_collisions(None)
        self.model.geom_conaffinity[self.left_arm_geom_ids] &= ~self.pinch_collision_bit
        return len(strip)

    def turn_end_for_end(self) -> None:
        """
        Lay the fruit back on the board with the left hand and let go of it; take it
        from above with the right hand, turn it end for end over the board, lay it down
        again and take the right hand out of the way.

        :raises NeighbouringStripsTorn: If more than :attr:`maximum_tiles_lost_turning`
            tiles came off meanwhile.
        """
        left, right = self.arms[ArmSide.LEFT], self.arms[ArmSide.RIGHT]
        x, y = self.countertop_spot
        held, rotation = left.tool_pose()
        # Down until the fruit's lowest point is just above the board, wherever the held
        # fruit hangs lowest: the hand, lowered to where it took the fruit, would press
        # the drooping free end into the board and tip the fruit up.
        lying = held - [0.0, 0.0, self.fruit_bottom() - self.board_top - self.drop_gap]
        torn = self.move_tool(left, [ToolWaypoint(lying, self.reach_time)], rotation)
        for name in self.all_holding_weld_names():
            self.release_weld(name)
        torn += self.close_gripper(left, self.open_gripper)
        # Up and back out along the fruit, the open fingers clear of it.
        torn += self.move_tool(
            left,
            [
                ToolWaypoint(lying + [0.0, 0.0, self.drop_gap * 3], self.approach_time),
                ToolWaypoint(
                    lying
                    - self.hover_height * rotation[:, 0]
                    + [0.0, 0.0, self.drop_gap * 3],
                    self.approach_time,
                ),
                ToolWaypoint(self.ready_position(ArmSide.LEFT), self.reach_time),
            ],
            rotation,
        )
        axis = self.fruit_axis_now()
        grasp = self.fruit_centre()
        across = self.downward_grasp(axis)
        right.set_gripper(self.open_gripper)
        torn += self.move_tool(
            right,
            [
                ToolWaypoint(grasp + [0, 0, self.hover_height], self.reach_time),
                ToolWaypoint(grasp, self.approach_time),
            ],
            across,
        )
        torn += self.close_gripper(right, 0.0)
        self.attach(
            SceneConstraint.GRIP,
            right,
            self.heaviest_chunk(self.fruit.description.pit.side),
        )
        drop = np.array([x, y, grasp[2]])
        # About the vertical until the stem points the other way from where it pointed
        # when the fruit was laid down, about half a turn: also taking back how far the
        # fruit turned as it was let go of.
        wanted = -self.fruit_axis()
        angle = math.atan2(
            axis[0] * wanted[1] - axis[1] * wanted[0],
            axis[0] * wanted[0] + axis[1] * wanted[1],
        )
        about_vertical = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        turned = about_vertical @ across
        torn += self.move_tool(
            right,
            [
                ToolWaypoint(grasp + [0, 0, self.turning_height], self.approach_time),
                ToolWaypoint(drop + [0, 0, self.turning_height], self.carry_time),
            ],
            across,
            turn_to=turned,
        )
        torn += self.move_tool(right, [ToolWaypoint(drop, self.approach_time)], turned)
        self.release_weld(SceneConstraint.GRIP)
        # Wide open, and touching only the other arm on the way up: the peeled flesh,
        # about as wide as the open fingers, would otherwise stick between them.
        torn += self.close_gripper(
            right, float(self.model.actuator_ctrlrange[right.finger_actuator_id][1])
        )
        right.set_hand_collisions(self.pinch_collision_bit)
        self.model.geom_conaffinity[self.left_arm_geom_ids] |= self.pinch_collision_bit
        torn += self.move_tool(
            right,
            [
                ToolWaypoint(drop + [0, 0, self.hover_height], self.approach_time),
                ToolWaypoint(self.ready_position(ArmSide.RIGHT), self.reach_time),
            ],
            turned,
        )
        right.set_hand_collisions(None)
        self.model.geom_conaffinity[self.left_arm_geom_ids] &= ~self.pinch_collision_bit
        if len(torn) > self.maximum_tiles_lost_turning:
            raise NeighbouringStripsTorn([t.description.body for t in torn])

    def roll_fruit(self, angle: float) -> None:
        """
        Turn the fruit about its axis by turning the left wrist, whose roll axis the
        fruit's axis lies along.

        :param angle: How far to turn it, in radians.
        """
        left = self.arms[ArmSide.LEFT]
        actuator = left.actuator_ids[-1]
        start = float(self.data.ctrl[actuator])
        duration = max(abs(angle) / self.roll_speed, self.model.opt.timestep)
        step = [0]
        steps = max(1, round(duration / self.model.opt.timestep))

        def turn(elapsed: float) -> None:
            self.data.ctrl[actuator] = start + angle * min(1.0, step[0] / steps)
            step[0] += 1

        self.simulate(duration + self.settle_time, turn)
        left.solution[-1] = start + angle
        # The wrist's rest goes round with it: the inverse kinematics would otherwise
        # pull the turned wrist back toward its old rest, and the hand off its target.
        left.posture[-1] += angle

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

    def downward_grasp(self, axis: np.ndarray | None = None) -> np.ndarray:
        """
        :param axis: The direction the lying fruit's axis points in; as it was laid down
            if None.
        :return: The tool rotation pointing the fingers down, closing them across the
            lying fruit.
        """
        approach = np.array([0.0, 0.0, -1.0])
        if axis is None:
            axis = self.fruit_axis()
        across = np.cross(axis, approach)
        across /= np.linalg.norm(across)
        return np.column_stack([approach, across, np.cross(approach, across)])

    @staticmethod
    def along_axis_grasp(axis: np.ndarray) -> np.ndarray:
        """
        :param axis: The fruit's axis, from base to stem.
        :return: The tool rotation pointing the fingers along the axis, toward the stem,
            and closing them horizontally across the fruit.
        """
        approach = axis / np.linalg.norm(axis)
        across = np.cross([0.0, 0.0, 1.0], approach)
        across /= np.linalg.norm(across)
        return np.column_stack([approach, across, np.cross(approach, across)])

    def pinch_grasp(self, outward: np.ndarray, along: np.ndarray) -> np.ndarray:
        """
        :param outward: The skin's outward normal where it is pinched.
        :param along: The direction along the strip.
        :return: The tool rotation pointing the fingers into the skin, tilted by
            :attr:`pinch_tilt` away from *along*, so the hand stays on that side of the
            pinch, and closing them along the strip: on the tile's edges toward the
            stem and the base. The hand is narrow that way.
        """
        normal = outward / np.linalg.norm(outward)
        tangent = along - (along @ normal) * normal
        tangent /= np.linalg.norm(tangent)
        approach = (
            -math.cos(self.pinch_tilt) * normal - math.sin(self.pinch_tilt) * tangent
        )
        closing = along - (along @ approach) * approach
        closing /= np.linalg.norm(closing)
        return np.column_stack([approach, closing, np.cross(approach, closing)])

    def holding_weld_names(self) -> list[str]:
        """
        :return: The names of the left hand's welds besides :attr:`SceneConstraint.HOLD`.
        """
        return [f"{SceneConstraint.HOLD}_{k}" for k in range(1, self.holding_chunks)]

    def all_holding_weld_names(self) -> list[str]:
        """
        :return: The names of all the welds the left hand holds the fruit by.
        """
        return [SceneConstraint.HOLD, *self.holding_weld_names()]

    def pinching_weld_names(self) -> list[str]:
        """
        :return: The names of the right hand's welds besides :attr:`SceneConstraint.GRIP`.
        """
        return [f"{SceneConstraint.GRIP}_{k}" for k in range(1, self.pinched_width)]

    def all_pinching_weld_names(self) -> list[str]:
        """
        :return: The names of all the welds the right hand pinches a strip by.
        """
        return [SceneConstraint.GRIP, *self.pinching_weld_names()]

    def attach(self, constraint: str, arm: Pr2Arm, body_id: int) -> None:
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

    def away_from_hand(self) -> np.ndarray:
        """
        :return: The direction along the fruit's axis away from the left hand holding it
            by one end.
        """
        axis = self.fruit_axis_now()
        return -axis if self.held_at_stem else axis

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

    def heaviest_chunk(self, side: FruitSide) -> int:
        """
        :param side: A half of the fruit.
        :return: The id of the half's heaviest flesh chunk, deep in the flesh near the
            middle of the fruit.
        """
        heaviest = max(
            (chunk for chunk in self.fruit.description.flesh if chunk.side == side),
            key=lambda chunk: chunk.mass,
        )
        return self.body_id(self.fruit_prefix + heaviest.body)

    def fruit_centre(self) -> np.ndarray:
        """
        :return: The fruit's centre of mass, in world coordinates.
        """
        ids = self.fruit.flesh_body_ids
        masses = self.model.body_mass[ids]
        return masses @ self.data.xipos[ids] / masses.sum()

    def fruit_points(self) -> np.ndarray:
        """
        :return: The corners of the meshes on the fruit's flesh chunks, in world
            coordinates: the flesh and, over it, the skin's shells, whether the skin is
            still on there or not. Peeled tiles lying elsewhere are left out.
        """
        geoms = np.flatnonzero(
            np.isin(self.model.geom_bodyid, self.fruit.flesh_body_ids)
        )
        points = []
        for geom in geoms:
            mesh = self.model.geom_dataid[geom]
            first = self.model.mesh_vertadr[mesh]
            vertices = self.model.mesh_vert[
                first : first + self.model.mesh_vertnum[mesh]
            ]
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            points.append(self.data.geom_xpos[geom] + vertices @ rotation.T)
        return np.concatenate(points)

    def fruit_bottom(self) -> float:
        """
        :return: The height in metres of the lowest point of the fruit.
        """
        return float(self.fruit_points()[:, 2].min())

    def fruit_end(self, toward_stem: bool) -> np.ndarray:
        """
        :param toward_stem: Whether the stem end is wanted, rather than the base end.
        :return: The point on the fruit's long axis at that end, in world coordinates.
        """
        axis = self.fruit_axis_now()
        centre = self.fruit_centre()
        reach = (self.fruit_points() - centre) @ axis
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
    outcome = Pr2WholeAvocadoPeelingDemonstration(
        avocado_folder(avocado_directory), headless=headless
    ).run()
    print(
        f"The PR2 laid the avocado {outcome.placement_error * 1000:.0f} mm from the "
        f"board's centre and peeled {len(outcome.strip_lengths)} strips of "
        f"{outcome.strip_lengths} tiles; {outcome.peeled_tile_count} of its "
        f"{outcome.tile_count} skin tiles came off."
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "The PR2 puts an avocado on a cutting board and peels all of it, in MuJoCo."
        )
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

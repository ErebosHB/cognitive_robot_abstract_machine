"""
An avocado is peeled, cut, pitted and sliced in MuJoCo, in the kitchen of the IAI
apartment.

The avocado lies on a cutting board on the kitchen island, the half holding the pit
down, and is held by its upper half. A gripper grabs the skin at the stem end of the
strip facing up and draws it back over the fruit, tearing that strip off tile by tile
while the rest of the skin stays on. The fruit is then cut, its upper half laid down
beside the lower one, the pit lifted out of its cavity and the stem slice of the lower
half cut off and set aside.

The avocado is the peelable produce model of ``private_meshes``; seed 0 comes with CRAM
in ``coraplex/resources/objects/avocado``. Run it from this folder::

    python demo.py

or pass another avocado's folder, or set ``AVOCADO_DIRECTORY`` to it.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from exceptions import (
    FruitModelNotFound,
    NeighbouringStripsTorn,
    PitStuck,
    SkinTornAtRest,
    SliceStuck,
    StripNotPeeled,
    UnstableSimulation,
)
from krrood.patterns.path_enumeration import PathEnumeration
from peeling import FruitSide, PeelableFruit, PeelingDescription, SkinTile
from typing_extensions import Callable

# %% names


class SceneFile(PathEnumeration):
    """
    The files the scene is built from, in this repository.
    """

    APARTMENT = (
        Path(__file__).resolve().parents[3]
        / "semantic_digital_twin"
        / "resources"
        / "mjcf"
        / "iai_apartment.xml"
    )
    """
    The IAI apartment, whose kitchen island holds the cutting board.
    """

    AVOCADO = Path(__file__).resolve().parents[2] / "resources" / "objects" / "avocado"
    """
    The folder of the peelable avocado, seed 0 of the generator of ``private_meshes``.
    """


class FruitFile(StrEnum):
    """
    The files of a peelable fruit model, inside its folder.
    """

    MODEL = "model.xml"
    PEELING_DESCRIPTION = "mujoco_peel.json"


class SceneBody(StrEnum):
    """
    The bodies the scene adds around the fruit.
    """

    GRIPPER = "gripper"
    """
    The mocap target the gripping hand is dragged after.
    """

    HOLDER = "holder"
    """
    The mocap target the holding hand is dragged after.
    """

    GRIPPER_HAND = "gripper_hand"
    """
    The hand that grips the skin and the pit: a light body, not a mocap.
    """

    HOLDER_HAND = "holder_hand"
    """
    The hand that holds the flesh: a light body, not a mocap.
    """

    CUTTING_BOARD = "cutting_board"


class SceneConstraint(StrEnum):
    """
    The welds the scene adds to hold the fruit and grip its skin.
    """

    GRIP = "grip"
    HOLD = "hold"


class EnvironmentVariable(StrEnum):
    """
    Environment variables the demonstration reads.
    """

    AVOCADO_DIRECTORY = "AVOCADO_DIRECTORY"


# %% the pull


@dataclass
class GripperWaypoint:
    """
    A point the gripper passes while peeling, relative to the grabbed tile's start.
    """

    outward: float
    """
    How far in metres the point lies out from the skin at the grabbed tile.
    """

    toward_base: float
    """
    How far in metres the point lies along the fruit toward its base.
    """

    down: float
    """
    How far in metres the point lies below the start.
    """

    travel_time: float
    """
    How long in seconds the gripper takes to reach the point from the previous one.
    """


@dataclass
class MocapWaypoint:
    """
    A point a mocap body passes, relative to where it starts.
    """

    offset: tuple[float, float, float]
    """
    The point's offset in metres from the start, in world axes.
    """

    travel_time: float
    """
    How long in seconds the body takes to reach the point from the previous one.
    """


# %% the outcome


@dataclass
class PeelingOutcome:
    """
    What the demonstration achieved.
    """

    peeled_strip: str
    """
    The name of the tile the strip was pulled by.
    """

    peeled_tiles: list[str]
    """
    The names of the strip's tiles that came off.
    """

    pit_displacement: float
    """
    How far in metres the pit left its cavity.
    """

    slice_displacement: float
    """
    How far in metres the cut-off stem slice was carried away from its neighbour.
    """


# %% the demonstration


@dataclass
class AvocadoPeelingDemonstration:
    """
    Peel one strip of an avocado's skin, cut the avocado and remove its pit, checking
    that each stage ends as it should.
    """

    avocado_directory: Path = field(default_factory=lambda: Path(SceneFile.AVOCADO))
    """
    The folder of the avocado model, holding its :class:`FruitFile` files; the avocado
    that comes with CRAM by default.
    """

    headless: bool = True
    """
    Whether the physics runs without a viewer.
    """

    fruit_prefix: str = ""
    """
    The prefix of the fruit's names in the scene.

    .. warning:: MuJoCo 3.x does not prefix the bodies a skin follows when attaching, so
        a non-empty prefix fails to compile there; the fruit's own names already start
        with its name.
    """

    countertop_spot: tuple[float, float] = (2.75, 2.45)
    """
    The point (x, y) in metres on the kitchen island, between stove and sink, that the
    cutting board is centred on.
    """

    fruit_heading: float = 0.0
    """
    The direction in radians, about the vertical from world x, in which the stem of the
    lying fruit points.
    """

    cutting_board_size: tuple[float, float, float] = (0.40, 0.30, 0.02)
    """
    The length, width and thickness in metres of the cutting board.
    """

    cutting_board_color: tuple[float, float, float, float] = (0.66, 0.47, 0.29, 1.0)
    """
    The color of the cutting board, a light wood.
    """

    viewer_distance: float = 0.5
    """
    How far in metres the viewer's camera starts from the cutting board.
    """

    viewer_azimuth: float = 200.0
    """
    The direction in degrees, about the vertical, the viewer's camera starts looking in.
    """

    viewer_elevation: float = -30.0
    """
    How far in degrees the viewer's camera starts looking up; negative is down.
    """

    placement_clearance: float = 0.0002
    """
    How far in metres above the cutting board the fruit's lowest point starts.
    """

    rest_duration: float = 1.5
    """
    How long in seconds the fruit rests on the cutting board before it is touched.
    """

    peeling_path: list[GripperWaypoint] = field(
        default_factory=lambda: [
            GripperWaypoint(outward=0.02, toward_base=0.0, down=0.0, travel_time=0.5),
            GripperWaypoint(outward=0.04, toward_base=0.06, down=0.0, travel_time=1.0),
            GripperWaypoint(outward=0.05, toward_base=0.14, down=0.0, travel_time=1.5),
            GripperWaypoint(outward=0.0, toward_base=0.2, down=0.02, travel_time=1.0),
            GripperWaypoint(outward=0.0, toward_base=0.24, down=0.05, travel_time=0.8),
        ]
    )
    """
    The points the gripper passes after grabbing the tile: off the skin, back over the
    fruit and down past its base.
    """

    parting_distance: float = 0.02
    """
    How far in metres the holder draws the upper half off the lower one, along the
    normal of the cut, before carrying it aside.
    """

    parting_time: float = 0.4
    """
    How long in seconds parting the halves takes.
    """

    peeling_clearance: float = 0.02
    """
    How far in metres above the cutting board the gripper stays while peeling.
    """

    half_moving_path: list[MocapWaypoint] = field(
        default_factory=lambda: [
            MocapWaypoint(offset=(0.0, 0.0, 0.04), travel_time=0.4),
            MocapWaypoint(offset=(0.0, 0.09, 0.04), travel_time=0.6),
        ]
    )
    """
    The points the holder carries the upper flesh half through once parted, relative to
    where it starts: up and aside, next to the lower half.
    """

    set_down_clearance: float = 0.003
    """
    How far in metres above the cutting board the cut face of the upper half is let go.
    """

    set_down_time: float = 0.8
    """
    How long in seconds the holder takes to lower the upper half onto the board.
    """

    slice_moving_path: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.02, 0.002, 0.6), (0.04, 0.03, 0.6)]
    )
    """
    The points the holder carries the cut-off stem piece through: how far in metres
    along the lower half's axis toward the stem, away from the rest, and up from where
    it starts, and the seconds taken to get there.
    """

    minimum_slice_displacement: float = 0.03
    """
    How far in metres the cut-off slice must be carried.
    """

    maximum_neighbour_displacement: float = 0.01
    """
    How far in metres the slice next to the cut-off one may move with it.
    """

    board_top: float = field(init=False)
    """
    The height in metres of the cutting board's top.
    """

    pit_moving_path: list[MocapWaypoint] = field(
        default_factory=lambda: [
            MocapWaypoint(offset=(0.0, 0.0, 0.06), travel_time=0.5),
            MocapWaypoint(offset=(0.0, -0.09, 0.06), travel_time=0.6),
            MocapWaypoint(offset=(0.0, -0.09, 0.0), travel_time=0.4),
        ]
    )
    """
    The points the gripper carries the pit through, relative to where it starts: out of
    its cavity, aside and down onto the cutting board.
    """

    approach_height: float = 0.05
    """
    How far in metres above the pit the gripper comes in from.
    """

    approach_time: float = 0.6
    """
    How long in seconds the gripper takes to reach the point above the pit.
    """

    settle_time: float = 0.3
    """
    How long in seconds the scene settles after something is let go.
    """

    grabbed_row: int = 1
    """
    The row of skin tiles, counted from the stem, the strip is grabbed by.
    """

    minimum_peeled_tiles: int = 8
    """
    How many tiles, the grabbed one included, must come off as one strip.
    """

    maximum_neighbouring_tiles_torn: int = 0
    """
    How many tiles that are not part of the peeled strip may come off, while peeling or
    cutting.
    """

    slice_height_share: float = 0.7
    """
    Where the fruit is sliced across, as a share of its length from the base.
    """

    hand_mass: float = 0.05
    """
    The mass in kilograms of each hand.
    """

    hand_drive_time_constant: float = 0.02
    """
    The time constant in seconds of the soft weld dragging each hand after its mocap
    target.
    """

    minimum_pit_displacement: float = 0.02
    """
    How far in metres the pit must leave its cavity.
    """

    model: mujoco.MjModel = field(init=False)
    """
    The compiled scene.
    """

    data: mujoco.MjData = field(init=False)
    """
    The simulation state of :attr:`model`.
    """

    fruit: PeelableFruit = field(init=False)
    """
    The avocado in the simulation.
    """

    linger_time: float = 3.0
    """
    How long in seconds the viewer keeps showing the finished scene before it closes.
    """

    viewer_shutdown_timeout: float = 5.0
    """
    How long in seconds to wait for the viewer's window to close.
    """

    viewer: mujoco.viewer.Handle | None = field(init=False, default=None)
    """
    The viewer showing the simulation, if not :attr:`headless`.
    """

    def __post_init__(self) -> None:
        if not all((self.avocado_directory / file).is_file() for file in FruitFile):
            raise FruitModelNotFound(self.avocado_directory)

    def run(self) -> PeelingOutcome:
        """
        :return: What the demonstration achieved.
        """
        self.build_scene()
        with self.viewer_context():
            self.let_rest()
            grabbed_tile, peeled_tiles = self.peel_strip_facing_up()
            pit_displacement = self.cut_and_remove_pit(grabbed_tile)
            slice_displacement = self.cut_off_stem_slice(grabbed_tile)
            self.linger()
        return PeelingOutcome(
            grabbed_tile.description.body,
            peeled_tiles,
            pit_displacement,
            slice_displacement,
        )

    def linger(self) -> None:
        """
        Keep the finished scene in the viewer for :attr:`linger_time`, so the last stage
        can be seen before the window closes.
        """
        if self.viewer is not None:
            self.simulate(self.linger_time)

    def viewer_context(self) -> ExitStack:
        """
        :return: A context that shows the simulation in a window while it lasts, unless
            :attr:`headless`.
        """
        closing = ExitStack()
        if not self.headless:
            self.open_viewer(closing)
        return closing

    def open_viewer(self, closing: ExitStack) -> None:
        """
        Show the simulation in a window that *closing* closes again.

        :param closing: Closes the window, and waits for its thread, when it unwinds.

        .. note:: The interpreter crashes when it exits while the window's thread is still
            shutting down, so the thread is joined after closing.
        """
        threads_before = set(threading.enumerate())
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        (viewer_thread,) = set(threading.enumerate()) - threads_before
        closing.callback(viewer_thread.join, self.viewer_shutdown_timeout)
        closing.callback(self.viewer.close)
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = self.data.xpos[
                self.body_id(SceneBody.CUTTING_BOARD)
            ]
            self.viewer.cam.distance = self.viewer_distance
            self.viewer.cam.azimuth = self.viewer_azimuth
            self.viewer.cam.elevation = self.viewer_elevation

    # %% scene

    def build_scene(self) -> None:
        """
        Put a cutting board on the kitchen island of the apartment and the avocado on
        its side, with the hands that will work on it.
        """
        spec = mujoco.MjSpec.from_file(str(SceneFile.APARTMENT))
        self.add_cutting_board(spec)
        description = PeelingDescription.from_file(
            self.avocado_directory / FruitFile.PEELING_DESCRIPTION
        )
        self.attach_fruit(spec, self.fruit_start(), description)
        for constraint, hand in self.add_hands(spec):
            weld = spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name=constraint,
                name1=hand,
                name2=self.fruit_prefix + description.tiles[0].flesh,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                active=False,
                solref=[0.004, 1],
            )
            # A zero anchor and identity relative pose; :meth:`weld` sets the pose the
            # bodies are welded in. MjSpec's default anchor is not zero.
            weld.data[0:10] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.fruit = PeelableFruit(
            self.model, self.data, description, name_prefix=self.fruit_prefix
        )
        self.prepare_hands()
        self.lower_fruit_onto(self.fruit_resting_height())

    def add_cutting_board(self, spec: mujoco.MjSpec) -> None:
        """
        Put the cutting board, and a lamp above it, on the kitchen island at
        :attr:`countertop_spot`; sets :attr:`board_top`.

        :param spec: The scene.
        """
        world = spec.worldbody
        x, y = self.countertop_spot
        length, width, thickness = self.cutting_board_size
        self.board_top = self.countertop_height() + thickness
        board = world.add_body(
            name=SceneBody.CUTTING_BOARD, pos=[x, y, self.board_top - thickness / 2]
        )
        board.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2, width / 2, thickness / 2],
            rgba=list(self.cutting_board_color),
            friction=[0.9, 0.01, 0.001],
        )
        world.add_light(
            pos=[x + 0.3, y - 0.4, self.board_top + 1.2],
            dir=[-0.25, 0.35, -1],
            diffuse=[0.8, 0.8, 0.8],
        )

    def fruit_start(self) -> np.ndarray:
        """
        :return: Where the fruit's origin, at its base, starts: on the cutting board,
            with the fruit's middle above the board's centre.
        """
        x, y = self.countertop_spot
        return np.array([x, y, self.board_top + 0.04]) - 0.055 * self.fruit_axis()

    def fruit_axis(self) -> np.ndarray:
        """
        :return: The direction from base to stem of the fruit lying on its side.
        """
        return np.array([math.cos(self.fruit_heading), math.sin(self.fruit_heading), 0])

    def fruit_resting_height(self) -> float:
        """
        :return: The height in metres of the surface the fruit starts on.
        """
        return self.board_top

    def add_hands(self, spec: mujoco.MjSpec) -> list[tuple[SceneConstraint, str]]:
        """
        Add a gripping and a holding hand above the cutting board.

        Each is a light body that a soft weld drags after a mocap target, the way a
        robot's controller drives its fingers; a mocap welded straight to a 0.05 g skin
        tile would fight the tile's stiff constraints and make the simulation unstable.

        :param spec: The scene.
        :return: Each weld the hands take hold with, and the hand's body it starts on.
        """
        x, y = self.countertop_spot
        for target_name, hand_name, color in (
            (SceneBody.GRIPPER, SceneBody.GRIPPER_HAND, [0.2, 0.4, 0.9, 1]),
            (SceneBody.HOLDER, SceneBody.HOLDER_HAND, [0, 0, 0, 0]),
        ):
            position = [x, y, self.board_top + 0.3]
            spec.worldbody.add_body(name=target_name, mocap=True, pos=position)
            hand = spec.worldbody.add_body(name=hand_name, pos=position)
            hand.add_freejoint()
            hand.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.004, 0, 0],
                mass=self.hand_mass,
                contype=0,
                conaffinity=0,
                rgba=color,
            )
            drive = spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name=f"{hand_name}_drive",
                name1=target_name,
                name2=hand_name,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solref=[self.hand_drive_time_constant, 1],
            )
            drive.data[0:10] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        return [
            (SceneConstraint.GRIP, SceneBody.GRIPPER_HAND),
            (SceneConstraint.HOLD, SceneBody.HOLDER_HAND),
        ]

    def prepare_hands(self) -> None:
        """
        Set the hands up in the compiled scene; mocap bodies need nothing.
        """

    def attach_fruit(
        self, spec: mujoco.MjSpec, position: np.ndarray, description: PeelingDescription
    ) -> None:
        """
        Add the avocado to the scene lying on its side, its stem along
        :meth:`fruit_axis`, and the half holding the pit down, so the pit can be lifted
        out once the upper half is taken away.

        :param spec: The scene.
        :param position: Where the fruit's origin, at its base, goes.
        :param description: The fruit's peeling description.
        """
        fruit_spec = mujoco.MjSpec.from_file(
            str(self.avocado_directory / FruitFile.MODEL)
        )
        spec.option.timestep = fruit_spec.option.timestep
        spec.option.integrator = fruit_spec.option.integrator
        # Elliptic friction cones with a high impedance ratio, as the fruit asks for: with
        # the default pyramidal cones a held fruit creeps out of the fingers.
        spec.option.cone = fruit_spec.option.cone
        spec.option.impratio = fruit_spec.option.impratio
        heading = np.array(
            [math.cos(self.fruit_heading / 2), 0, 0, math.sin(self.fruit_heading / 2)]
        )
        lying = np.zeros(4)
        mujoco.mju_mulQuat(
            lying,
            heading,
            np.array([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]),
        )
        if description.pit.side == FruitSide.LEFT:
            # Lying this way, the left half (x < 0 in the fruit) faces up: turn it over.
            mujoco.mju_mulQuat(lying, lying.copy(), np.array([0.0, 0.0, 0.0, 1.0]))
        frame = spec.worldbody.add_frame(pos=position, quat=lying)
        spec.attach(fruit_spec, prefix=self.fruit_prefix, frame=frame)

    def lower_fruit_onto(self, surface_height: float) -> None:
        """
        Move the whole fruit down until its lowest skin point lies
        :attr:`placement_clearance` above a surface, so it settles without a drop.

        :param surface_height: The height in metres of the surface below the fruit.
        """
        mujoco.mj_forward(self.model, self.data)
        lowest_point = min(
            self.lowest_point_of(tile.geom_id) for tile in self.fruit.tiles
        )
        drop = lowest_point - surface_height - self.placement_clearance
        fruit_bodies = {
            *self.fruit.flesh_body_ids.tolist(),
            self.fruit.pit_body_id,
            *(tile.body_id for tile in self.fruit.tiles),
        }
        for joint_id in range(self.model.njnt):
            if (
                self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
                and self.model.jnt_bodyid[joint_id] in fruit_bodies
            ):
                self.data.qpos[self.model.jnt_qposadr[joint_id] + 2] -= drop
        mujoco.mj_forward(self.model, self.data)

    def lowest_point_of(self, geom_id: int) -> float:
        """
        :param geom_id: The id of a mesh geom.
        :return: The height in metres of the geom's lowest vertex in the current state.
        """
        mesh_id = self.model.geom_dataid[geom_id]
        first_vertex = self.model.mesh_vertadr[mesh_id]
        vertices = self.model.mesh_vert[
            first_vertex : first_vertex + self.model.mesh_vertnum[mesh_id]
        ]
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
        return float((vertices @ rotation.T + self.data.geom_xpos[geom_id])[:, 2].min())

    def countertop_height(self) -> float:
        """
        :return: The height in metres of the kitchen island's top at
            :attr:`countertop_spot`.
        """
        apartment = mujoco.MjModel.from_xml_path(str(SceneFile.APARTMENT))
        apartment_state = mujoco.MjData(apartment)
        mujoco.mj_forward(apartment, apartment_state)
        x, y = self.countertop_spot
        ray_start_height = 3.0
        hit_geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            apartment,
            apartment_state,
            np.array([x, y, ray_start_height]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            -1,
            hit_geom,
        )
        return ray_start_height - distance

    def body_id(self, name: str) -> int:
        """
        :param name: The full name of a body of the scene.
        :return: Its id.
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def weld(self, constraint: SceneConstraint, mocap: SceneBody, body_id: int) -> None:
        """
        Put a hand and its mocap target on a body and weld the hand to it there.

        :param constraint: The weld to switch on.
        :param mocap: The mocap target of the hand the weld starts at.
        :param body_id: The id of the body it ends at.
        """
        constraint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, constraint
        )
        self.model.eq_obj2id[constraint_id] = body_id
        mocap_id = self.model.body_mocapid[self.body_id(mocap)]
        self.data.mocap_pos[mocap_id] = self.data.xpos[body_id]
        self.data.mocap_quat[mocap_id] = self.data.xquat[body_id]
        hand_id = self.model.eq_obj1id[constraint_id]
        joint = self.model.body_jntadr[hand_id]
        if joint >= 0:
            address = self.model.jnt_qposadr[joint]
            self.data.qpos[address : address + 3] = self.data.xpos[body_id]
            self.data.qpos[address + 3 : address + 7] = self.data.xquat[body_id]
            velocity = self.model.jnt_dofadr[joint]
            self.data.qvel[velocity : velocity + 6] = 0.0
        self.data.eq_active[constraint_id] = 1

    # %% simulation

    def simulate(
        self, duration: float, move_mocaps: Callable[[float], None] | None = None
    ) -> list[SkinTile]:
        """
        Step the simulation, releasing skin tiles pulled off the flesh.

        :param duration: How long in seconds to simulate.
        :param move_mocaps: Called with the elapsed time before every step.
        :return: The tiles released meanwhile.
        """
        released_tiles = []
        wall_clock_start = time.perf_counter()
        for step in range(round(duration / self.model.opt.timestep)):
            elapsed = step * self.model.opt.timestep
            if move_mocaps is not None:
                move_mocaps(elapsed)
            mujoco.mj_step(self.model, self.data)
            released_tiles += self.fruit.release_pulled_tiles()
            self.after_step()
            if self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
                raise UnstableSimulation(self.data.time)
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(max(0.0, elapsed - (time.perf_counter() - wall_clock_start)))
        return released_tiles

    def after_step(self) -> None:
        """
        Called after every simulation step; nothing to do with mocap hands.
        """

    # %% stages

    def let_rest(self) -> None:
        """
        Let the fruit settle on the cutting board; no skin may come off.
        """
        torn_tiles = self.simulate(self.rest_duration)
        if torn_tiles:
            raise SkinTornAtRest([tile.description.body for tile in torn_tiles])

    def peel_strip_facing_up(self) -> tuple[SkinTile, list[str]]:
        """
        Hold the upper half, grab a tile near the stem of its column facing up and draw
        it back over the fruit, so a strip peels off from the stem toward the base.

        :return: The grabbed tile and the names of the tiles that came off with it.
        """
        upper_half_id = self.upper_flesh_half_id()
        self.weld(SceneConstraint.HOLD, SceneBody.HOLDER, upper_half_id)
        upper_half = set(self.upper_half_ids())
        # A strip of the held half: pulling one of the free lower half drags it along.
        grabbed_tile = max(
            (
                tile
                for tile in self.fruit.tiles
                if tile.description.row == self.grabbed_row
                and tile.flesh_body_id in upper_half
            ),
            key=lambda tile: self.outward_normal(tile)[2],
        )
        start = self.data.xpos[grabbed_tile.body_id].copy()
        outward = self.outward_normal(grabbed_tile)
        toward_base = -self.fruit_axis_now()
        self.weld(SceneConstraint.GRIP, SceneBody.GRIPPER, grabbed_tile.body_id)
        points = [
            start
            + waypoint.outward * outward
            + waypoint.toward_base * toward_base
            - [0, 0, waypoint.down]
            for waypoint in self.peeling_path
        ]
        # Never pull the strip into the board.
        for point in points:
            point[2] = max(point[2], self.board_top + self.peeling_clearance)
        released_tiles = self.move_through(
            SceneBody.GRIPPER,
            points,
            [waypoint.travel_time for waypoint in self.peeling_path],
        )
        self.release_weld(SceneConstraint.GRIP)
        peeled_tiles = [
            tile.description.body for tile in self.fruit.peeled_with(grabbed_tile)
        ]
        if len(peeled_tiles) < self.minimum_peeled_tiles:
            raise StripNotPeeled(
                grabbed_tile.description.body,
                len(peeled_tiles),
                self.minimum_peeled_tiles,
            )
        self.check_only_pulled_strip_torn(released_tiles, grabbed_tile)
        return grabbed_tile, peeled_tiles

    def cut_and_remove_pit(self, grabbed_tile: SkinTile) -> float:
        """
        Halve the fruit, set its upper half down beside the lower one and lift the pit
        out of its cavity onto the cutting board; the skin stays on both halves.

        :param grabbed_tile: The tile the strip was peeled by; what is left of the strip
            may tear off.
        :return: How far in metres the pit left its cavity.
        """
        self.fruit.halve()
        torn_tiles = self.set_upper_half_down()

        pit_id = self.fruit.pit_body_id
        seat_id = self.body_id(self.fruit_prefix + self.fruit.description.pit.flesh)
        pit_in_seat_before = self.position_in_frame_of(pit_id, seat_id)
        self.fruit.remove_pit()
        pit_position = self.data.xpos[pit_id].copy()
        torn_tiles += self.move_through(
            SceneBody.GRIPPER,
            [pit_position + [0, 0, self.approach_height], pit_position],
            [self.approach_time, self.approach_time / 2],
        )
        self.weld(SceneConstraint.GRIP, SceneBody.GRIPPER, pit_id)
        torn_tiles += self.carry(
            SceneBody.GRIPPER, SceneConstraint.GRIP, self.pit_moving_path
        )

        self.check_only_pulled_strip_torn(torn_tiles, grabbed_tile)
        displacement = float(
            np.linalg.norm(
                self.position_in_frame_of(pit_id, seat_id) - pit_in_seat_before
            )
        )
        if displacement < self.minimum_pit_displacement:
            raise PitStuck(displacement)
        return displacement

    def set_upper_half_down(self) -> list[SkinTile]:
        """
        Carry the upper half aside and lay it on its cut face on the cutting board.

        :return: The skin tiles released meanwhile.
        """
        holder_id = self.model.body_mocapid[self.body_id(SceneBody.HOLDER)]
        held_id = self.model.eq_obj2id[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, SceneConstraint.HOLD
            )
        ]
        # Part the halves along the normal of the cut first, so the upper half clears
        # the pit instead of dragging the lower half along by it.
        away = self.data.xmat[held_id].reshape(3, 3)[:, 0] * (
            -1.0 if self.upper_side() == FruitSide.LEFT else 1.0
        )
        torn_tiles = self.move_through(
            SceneBody.HOLDER,
            [self.data.mocap_pos[holder_id] + self.parting_distance * away],
            [self.parting_time],
        )
        start = self.data.mocap_pos[holder_id].copy()
        torn_tiles += self.move_through(
            SceneBody.HOLDER,
            [start + waypoint.offset for waypoint in self.half_moving_path],
            [waypoint.travel_time for waypoint in self.half_moving_path],
        )
        lowest_point = min(
            self.lowest_point_of(geom) for geom in self.geoms_of(self.upper_half_ids())
        )
        lowering = self.board_top + self.set_down_clearance - lowest_point
        torn_tiles += self.move_through(
            SceneBody.HOLDER,
            [self.data.mocap_pos[holder_id] + [0.0, 0.0, lowering]],
            [self.set_down_time],
        )
        torn_tiles += self.simulate(self.settle_time)
        self.release_weld(SceneConstraint.HOLD)
        return torn_tiles + self.simulate(self.settle_time)

    def cut_off_stem_slice(self, grabbed_tile: SkinTile) -> float:
        """
        Slice the fruit across near the stem and carry the lower half's stem-side piece
        away; the rest of the lower half must stay where it is and the skin must stay
        on.

        :param grabbed_tile: The tile the strip was peeled by; what is left of the strip
            may tear off.
        :return: How far in metres the stem-side piece was carried.
        """
        length = max(tile.description.position[2] for tile in self.fruit.tiles)
        height = self.slice_height_share * length
        # Placed by the lower half, which is still where it was cut.
        self.fruit.slice(height, reference=self.fruit.description.pit.flesh)
        lower_side = self.fruit.description.pit.side
        chunks = {chunk.body: chunk for chunk in self.fruit.description.flesh}
        lower_pieces = [
            piece
            for piece in self.fruit.pieces()
            if chunks[piece[0]].side == lower_side
        ]
        stem_piece = max(
            lower_pieces,
            key=lambda piece: np.mean([chunks[name].position[2] for name in piece]),
        )
        rest_piece = next(piece for piece in lower_pieces if piece is not stem_piece)
        stem_id, rest_id = (
            self.body_id(
                self.fruit_prefix + max(piece, key=lambda name: chunks[name].mass)
            )
            for piece in (stem_piece, rest_piece)
        )
        stem_start = self.data.xipos[stem_id].copy()
        rest_start = self.data.xipos[rest_id].copy()
        self.weld(SceneConstraint.HOLD, SceneBody.HOLDER, stem_id)
        # The lower half may have rolled since it was cut: follow its own axis.
        toward_stem = self.data.xmat[rest_id].reshape(3, 3)[:, 2]
        torn_tiles = self.carry(
            SceneBody.HOLDER,
            SceneConstraint.HOLD,
            [
                MocapWaypoint(
                    offset=tuple(along * toward_stem + [0.0, 0.0, up]),
                    travel_time=travel_time,
                )
                for along, up, travel_time in self.slice_moving_path
            ],
        )
        self.check_only_pulled_strip_torn(torn_tiles, grabbed_tile)
        slice_displacement = float(
            np.linalg.norm(self.data.xipos[stem_id] - stem_start)
        )
        neighbour_displacement = float(
            np.linalg.norm(self.data.xipos[rest_id] - rest_start)
        )
        if (
            slice_displacement < self.minimum_slice_displacement
            or neighbour_displacement > self.maximum_neighbour_displacement
        ):
            raise SliceStuck(slice_displacement, neighbour_displacement)
        return slice_displacement

    def check_only_pulled_strip_torn(
        self, torn_tiles: list[SkinTile], grabbed_tile: SkinTile
    ) -> None:
        """
        :param torn_tiles: Skin tiles that came off.
        :param grabbed_tile: The tile a strip was peeled by.
        :raises NeighbouringStripsTorn: If more tiles came off that are not part of the
            peeled strip than :attr:`maximum_neighbouring_tiles_torn`.
        """
        strip = {tile.index for tile in self.fruit.peeled_with(grabbed_tile)}
        neighbouring_tiles = [
            tile.description.body for tile in torn_tiles if tile.index not in strip
        ]
        if len(neighbouring_tiles) > self.maximum_neighbouring_tiles_torn:
            raise NeighbouringStripsTorn(neighbouring_tiles)

    # %% moving things

    def move_through(
        self, mocap: SceneBody, points: list[np.ndarray], travel_times: list[float]
    ) -> list[SkinTile]:
        """
        Move a mocap body from where it is through *points* at constant speed between
        them.

        :param mocap: The body to move.
        :param points: The world positions to pass, in order.
        :param travel_times: The seconds taken to reach each point from the previous
            one.
        :return: The skin tiles released meanwhile.
        """
        mocap_id = self.model.body_mocapid[self.body_id(mocap)]
        path = np.array([self.data.mocap_pos[mocap_id].copy(), *points])
        arrival_times = np.cumsum([0.0, *travel_times])

        def follow(elapsed: float) -> None:
            self.data.mocap_pos[mocap_id] = [
                np.interp(elapsed, arrival_times, path[:, axis]) for axis in range(3)
            ]

        return self.simulate(arrival_times[-1], follow)

    def carry(
        self, mocap: SceneBody, constraint: SceneConstraint, path: list[MocapWaypoint]
    ) -> list[SkinTile]:
        """
        Move a mocap body along *path*, let go of what it holds and let the scene
        settle.

        :param mocap: The body carrying something.
        :param constraint: The weld holding what it carries.
        :param path: The points to pass, relative to where the body starts.
        :return: The skin tiles released meanwhile.
        """
        start = self.data.mocap_pos[self.model.body_mocapid[self.body_id(mocap)]].copy()
        released_tiles = self.move_through(
            mocap,
            [start + waypoint.offset for waypoint in path],
            [waypoint.travel_time for waypoint in path],
        )
        self.release_weld(constraint)
        return released_tiles + self.simulate(self.settle_time)

    def release_weld(self, constraint: SceneConstraint) -> None:
        """
        :param constraint: The scene's weld to switch off.
        """
        self.data.eq_active[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, constraint)
        ] = 0

    def position_in_frame_of(self, body_id: int, frame_body_id: int) -> np.ndarray:
        """
        :param body_id: The body whose position is wanted.
        :param frame_body_id: The body whose frame it is expressed in.
        :return: The position of the first body's origin in the second body's frame.
        """
        rotation = self.data.xmat[frame_body_id].reshape(3, 3)
        return rotation.T @ (self.data.xpos[body_id] - self.data.xpos[frame_body_id])

    # %% fruit geometry

    def upper_side(self) -> FruitSide:
        """
        :return: The half lying on top, the one without the pit.
        """
        if self.fruit.description.pit.side == FruitSide.LEFT:
            return FruitSide.RIGHT
        return FruitSide.LEFT

    def upper_half_ids(self) -> list[int]:
        """
        :return: The ids of the flesh chunks of the half lying on top.
        """
        side = self.upper_side()
        return [
            int(body_id)
            for chunk, body_id in zip(
                self.fruit.description.flesh, self.fruit.flesh_body_ids
            )
            if chunk.side == side
        ]

    def upper_flesh_half_id(self) -> int:
        """
        :return: The id of the chunk the upper half is held by: its heaviest, deep in the
            flesh near the middle of the fruit.
        """
        side = self.upper_side()
        heaviest = max(
            (chunk for chunk in self.fruit.description.flesh if chunk.side == side),
            key=lambda chunk: chunk.mass,
        )
        return self.body_id(self.fruit_prefix + heaviest.body)

    def geoms_of(self, body_ids: list[int]) -> list[int]:
        """
        :param body_ids: Bodies of the scene.
        :return: The ids of their mesh geoms.
        """
        bodies = set(body_ids)
        return [
            geom
            for geom in range(self.model.ngeom)
            if self.model.geom_bodyid[geom] in bodies
            and self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        ]

    def fruit_axis_now(self) -> np.ndarray:
        """
        :return: The direction from base to stem of the fruit as it lies now.
        """
        return self.data.xmat[self.fruit.flesh_body_ids[0]].reshape(3, 3)[:, 2].copy()

    def outward_normal(self, tile: SkinTile) -> np.ndarray:
        """
        :param tile: A tile of the fruit.
        :return: The direction pointing out of the fruit at the tile, in world axes.
        """
        return (
            self.data.xmat[tile.flesh_body_id].reshape(3, 3) @ tile.description.normal
        )


# %% entry point


def avocado_folder(given: Path | None) -> Path:
    """
    :param given: The avocado folder given on the command line, if any.
    :return: That folder, else the one named by
        :attr:`EnvironmentVariable.AVOCADO_DIRECTORY`, else the avocado that comes with
        CRAM.
    """
    if given is not None:
        return given
    return Path(
        os.environ.get(EnvironmentVariable.AVOCADO_DIRECTORY, str(SceneFile.AVOCADO))
    )


def main(
    avocado_directory: Path | None = None, headless: bool = False
) -> PeelingOutcome:
    """
    Run the demonstration and report what it achieved.

    :param avocado_directory: The avocado model's folder; defaults to the folder named
        by :attr:`EnvironmentVariable.AVOCADO_DIRECTORY`, or else the avocado that comes
        with CRAM.
    :param headless: Whether the physics runs without a viewer.
    :return: What the demonstration achieved.
    """
    avocado_directory = avocado_folder(avocado_directory)
    outcome = AvocadoPeelingDemonstration(avocado_directory, headless=headless).run()
    print(
        f"Peeled a strip of {len(outcome.peeled_tiles)} tiles at {outcome.peeled_strip}; "
        f"the pit left its cavity by {outcome.pit_displacement * 1000:.0f} mm and the "
        f"stem slice was carried {outcome.slice_displacement * 1000:.0f} mm away."
    )
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Peel, cut, pit and slice an avocado in MuJoCo."
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

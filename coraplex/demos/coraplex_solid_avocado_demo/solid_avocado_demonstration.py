"""
What every demonstration with the solid avocado shares: a cutting board, a knife on a
mocap body, the viewer, and cutting, carrying and laying pieces of the avocado aside.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import manifold3d
import mujoco
import mujoco.viewer
import numpy as np
from krrood.patterns.path_enumeration import PathEnumeration
from typing_extensions import Iterator

from cutting import (
    BodyPose,
    CuttableFruit,
    CuttingPlane,
    FruitMaterial,
    FruitPiece,
    KnifeBlade,
)
from exceptions import CutOffKnifePlane, PiecesScattered, UnstableSimulation
from fruit_world import FruitWorld
from knife_strokes import KnifeStroke
from random_pit import RandomPit

# %% names


class SceneFile(PathEnumeration):
    """
    The files the scene is built from, in this repository.
    """

    SOLID_AVOCADO = (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "objects"
        / "solid_avocado"
        / "model.xml"
    )
    """
    The avocado of three solid parts.
    """


class SceneName(StrEnum):
    """
    The named things of the scene.
    """

    AVOCADO = "avocado"
    """
    The body of the whole avocado, as its model names it.
    """

    KNIFE = "knife"
    """
    The mocap body of the knife.
    """

    BLADE_EDGE = "knife_blade_edge"
    """
    The site in the middle of the knife's cutting edge.
    """


# %% outcomes of single steps


@dataclass
class CutOff:
    """
    A piece a stroke has cut off, with the way its cut face looks.
    """

    piece: FruitPiece
    """
    The piece.
    """

    face_normal: np.ndarray
    """
    The direction the piece's cut face looks in, in the frame of the piece's body.
    """

    on_normal_side: bool
    """
    Whether the piece lies on the side of the blade's plane its normal points to.
    """


# %% the demonstration


@dataclass
class SolidAvocadoDemonstration(ABC):
    """
    A knife on a mocap body works on the solid avocado above a cutting board.
    """

    headless: bool = False
    """
    Whether the physics runs without a viewer.
    """

    playback_speed: float = 1.0
    """
    How many seconds of simulated time the viewer shows per second of wall-clock time;
    below 1 is slow motion.
    """

    board_size: tuple[float, float, float] = (0.40, 0.28, 0.02)
    """
    The length, width and thickness of the cutting board in metres.
    """

    blade_size: tuple[float, float, float] = (0.20, 0.04, 0.0015)
    """
    The length of the blade's cutting edge, the blade's height from edge to spine and
    its thickness, in metres.
    """

    stroke_speed: float = 0.05
    """
    How fast the knife moves through the fruit, in metres per second.
    """

    stroke_clearance: float = 0.03
    """
    How far from the fruit the cutting edge starts a stroke, in metres.
    """

    stroke_overshoot: float = 0.002
    """
    How far past the lowest point of what it cuts the cutting edge ends a downward
    stroke, in metres.
    """

    pit_clearance: float = 0.0003
    """
    How far in metres the cutting edge keeps from the pit while going round it.
    """

    overturn: float = 0.3
    """
    How far past a full turn, in radians, the knife goes round an outline.
    """

    lift_height: float = 0.03
    """
    How far above where it lay and where it goes a piece laid aside is carried, in
    metres.
    """

    drop_height: float = 0.001
    """
    How far above the board a piece laid aside is let go, in metres.
    """

    lay_aside_duration: float = 1.5
    """
    How long laying a piece aside takes, in seconds.
    """

    rest_duration: float = 0.6
    """
    How long the fruit is left to come to rest before and after a stroke, in seconds.
    """

    cut_tolerance: float = 2e-5
    """
    How far in metres a cut surface may lie from the plane of the blade.
    """

    scatter_tolerance: float = 0.05
    """
    How far in metres a piece may move while the pieces come to rest: far enough for
    a piece to tip over, not to fly off.
    """

    linger_duration: float = 3.0
    """
    How long the viewer stays open after the last step, in seconds.
    """

    viewer_poll_interval: float = 0.01
    """
    How often, in seconds, closing the viewer checks whether its window is gone.
    """

    tool_speed: float = 0.1
    """
    How fast a tool moves through the air between strokes, in metres per second.
    """

    knife_lift: float = 0.3
    """
    How far the knife is lifted out of the way when it is no longer needed, in metres.
    """

    random_pit: RandomPit | None = field(default_factory=RandomPit)
    """
    How the avocado's pit is varied at random; None for the avocado's own pit.
    """

    pit_seed: int | None = None
    """
    The seed of the random pit; the same seed gives the same pit. None draws one, which
    is then kept here.
    """

    fruit_world: FruitWorld | None = field(default_factory=FruitWorld)
    """
    The avocado's pieces as bodies of a semantic digital twin world, kept in step with
    the simulation; None keeps no such world.
    """

    world_update_interval: float = 0.05
    """
    How often in simulated seconds :attr:`fruit_world` is brought up to date.
    """

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.05)
    """
    The point the viewer's camera looks at, in metres.
    """

    fruit: CuttableFruit = field(init=False)
    """
    The avocado.
    """

    blade: KnifeBlade = field(init=False)
    """
    The blade of the knife.
    """

    viewer: mujoco.viewer.Handle | None = field(init=False, default=None)
    """
    The viewer showing the scene, if any.
    """

    simulated_time: float = field(init=False, default=0.0)
    """
    How long the demonstration has simulated so far, in seconds.
    """

    playback_start: float = field(init=False, default=0.0)
    """
    The wall-clock time at which the viewer started showing the simulation, in seconds.
    """

    largest_deviation: float = field(init=False, default=0.0)
    """
    The farthest in metres any cut surface has lain from the plane of the blade that
    made it.
    """

    world_updated_at: float = field(init=False, default=-np.inf)
    """
    When in simulated time :attr:`fruit_world` was last brought up to date, in seconds.
    """

    whole_fruit: BodyPose = field(init=False)
    """
    The pose of the whole avocado once it has come to rest, the frame the strokes are
    given in.
    """

    def __post_init__(self) -> None:
        """
        Draw the pit's seed if none is given, and build the scene.
        """
        if self.pit_seed is None:
            self.pit_seed = int(np.random.default_rng().integers(2**31))
        self.build_scene()

    @contextmanager
    def shown(self) -> Iterator[None]:
        """
        Show the scene, let the avocado come to rest, and keep the viewer open for
        :attr:`linger_duration` after the steps made inside, closing it in the end.
        """
        with ExitStack() as closing:
            closing.callback(self.close_viewer)
            self.show_model()
            self.simulate(self.rest_duration)
            self.whole_fruit = self.fruit.pose_of(self.fruit.pieces[0])
            yield
            self.update_fruit_world()
            self.linger()

    # %% the scene

    def build_scene(self) -> None:
        """
        Put the avocado, with its random pit, above the cutting board, add what else
        the demonstration needs and hang the knife above it all.
        """
        spec = mujoco.MjSpec.from_file(str(SceneFile.SOLID_AVOCADO))
        if self.random_pit is not None:
            self.random_pit.apply_to(spec, SceneName.AVOCADO, self.pit_seed)
        # Pieces at rest sleep, so the many convex shapes of pieces lying on the board
        # cost nothing until something touches them.
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
        world = spec.worldbody
        length, width, thickness = self.board_size
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[1, 1, 0.1],
            rgba=[0.55, 0.57, 0.6, 1],
        )
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2, width / 2, thickness / 2],
            pos=[0, 0, thickness / 2],
            rgba=[0.72, 0.53, 0.33, 1],
            friction=[0.9, 0.01, 0.001],
        )
        world.add_light(pos=[0.3, -0.4, 1.2], dir=[-0.25, 0.35, -1])
        self.place_avocado(spec.body(SceneName.AVOCADO))
        self.add_equipment(spec)
        self.add_knife(spec)
        model = spec.compile()
        self.fruit = CuttableFruit(
            spec,
            model,
            mujoco.MjData(model),
            SceneName.AVOCADO,
            holders=self.holders(),
        )
        self.blade = KnifeBlade(SceneName.BLADE_EDGE, self.blade_size[0])

    @abstractmethod
    def place_avocado(self, avocado: mujoco.MjsBody) -> None:
        """
        :param avocado: The body of the whole avocado, whose frame has its origin at the
            base and its z axis along the long axis to the stem; set its pose.
        """

    def add_equipment(self, spec: mujoco.MjSpec) -> None:
        """
        Add what the demonstration needs besides the board, the avocado and the knife;
        nothing by default.

        :param spec: The scene.
        """

    def holders(self) -> list[manifold3d.Manifold]:
        """
        :return: The solids in the world that hold the pieces of the avocado reaching
            into them in place; none by default.
        """
        return []

    def add_knife(self, spec: mujoco.MjSpec) -> None:
        """
        Add the knife: a mocap body whose frame is that of the blade's site, and which
        touches nothing.

        :param spec: The scene.
        """
        edge_length, height, thickness = self.blade_size
        knife = spec.worldbody.add_body(
            name=SceneName.KNIFE, mocap=True, pos=[0, 0, 0.5]
        )
        knife.add_site(name=SceneName.BLADE_EDGE)
        knife.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[edge_length / 2, height / 2, thickness / 2],
            pos=[0, height / 2, 0],
            rgba=[0.82, 0.84, 0.86, 1],
            contype=0,
            conaffinity=0,
        )
        knife.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.055, 0.012, 0.009],
            pos=[-edge_length / 2 - 0.055, height - 0.012, 0],
            rgba=[0.12, 0.1, 0.09, 1],
            contype=0,
            conaffinity=0,
        )

    def mocap_id(self, body_name: str) -> int:
        """
        :param body_name: The name of a mocap body.
        :return: Its index among the mocap bodies.
        """
        model = self.fruit.model
        return model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        ]

    def move_mocap(
        self, body_name: str, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        """
        :param body_name: The name of a mocap body.
        :param position: Where its frame goes, in the world, in metres.
        :param quaternion: Its orientation, scalar first.
        """
        mocap_id = self.mocap_id(body_name)
        self.fruit.data.mocap_pos[mocap_id] = position
        self.fruit.data.mocap_quat[mocap_id] = quaternion

    def glide(
        self,
        body_name: str,
        position: np.ndarray,
        quaternion: np.ndarray,
        speed: float,
    ) -> None:
        """
        Move a mocap body in a straight line into a pose, turning it on the way.

        :param body_name: The name of the mocap body.
        :param position: Where its frame goes, in the world, in metres.
        :param quaternion: Its orientation there, scalar first.
        :param speed: How fast it moves, in metres per second.
        """
        model, data = self.fruit.model, self.fruit.data
        mocap_id = self.mocap_id(body_name)
        start = data.mocap_pos[mocap_id].copy()
        start_quaternion = data.mocap_quat[mocap_id].copy()
        axis, angle = self.rotation_between(start_quaternion, quaternion)
        distance = float(np.linalg.norm(position - start))
        steps = max(1, round(distance / speed / model.opt.timestep))
        for step in range(1, steps + 1):
            progress = step / steps
            turning = self.turn(axis, progress * angle)[0]
            self.move_mocap(
                body_name,
                start + progress * (position - start),
                self.compose(turning, start_quaternion),
            )
            self.step()

    def put_knife_away(self) -> None:
        """
        Lift the knife straight up out of the way.
        """
        data = self.fruit.data
        mocap_id = self.mocap_id(SceneName.KNIFE)
        position = data.mocap_pos[mocap_id] + np.array([0.0, 0.0, self.knife_lift])
        self.glide(
            SceneName.KNIFE,
            position,
            data.mocap_quat[mocap_id].copy(),
            self.tool_speed,
        )

    # %% the viewer

    def show_model(self) -> None:
        """
        Show the fruit's current model, in the viewer's window, from the same point of
        view as before.
        """
        if self.headless:
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.fruit.model, self.fruit.data
            )
            self.playback_start = time.time()
            self.viewer.cam.lookat = self.camera_target
            self.viewer.cam.distance = 0.6
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            return
        if not self.viewer.is_running():
            return
        # A passive viewer has no public way to load another model; this is how
        # MuJoCo's own viewer loads one.
        self.viewer._sim().load(self.fruit.model, self.fruit.data, "")

    def close_viewer(self) -> None:
        """
        Close the viewer, if one is open, and wait until its window is gone, as Python
        exiting while the window still draws crashes the process.
        """
        if self.viewer is None:
            return
        self.viewer.close()
        while self.viewer._sim() is not None:
            time.sleep(self.viewer_poll_interval)

    def linger(self) -> None:
        """
        Keep the viewer open for :attr:`linger_duration`.
        """
        if self.viewer is None:
            return
        closing_time = time.time() + self.linger_duration
        while self.viewer.is_running() and time.time() < closing_time:
            self.viewer.sync()
            time.sleep(0.02)

    # %% simulating

    def step(self) -> None:
        """
        Advance the simulation by one step, bring :attr:`fruit_world` up to date once
        :attr:`world_update_interval` has passed, and show the step.

        :raises UnstableSimulation: If MuJoCo reports diverging accelerations.
        """
        data = self.fruit.data
        mujoco.mj_step(self.fruit.model, data)
        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise UnstableSimulation(data.time)
        self.simulated_time += self.fruit.model.opt.timestep
        if self.simulated_time - self.world_updated_at >= self.world_update_interval:
            self.update_fruit_world()
        if self.viewer is None:
            return
        self.viewer.sync()
        shown_at = self.playback_start + self.simulated_time / self.playback_speed
        time.sleep(max(0.0, shown_at - time.time()))

    def update_fruit_world(self) -> None:
        """
        Bring :attr:`fruit_world`, if there is one, up to date with the simulation.
        """
        self.world_updated_at = self.simulated_time
        if self.fruit_world is None:
            return
        self.fruit_world.update(self.fruit)

    def simulate(self, duration: float) -> None:
        """
        :param duration: How long to simulate, in seconds.
        """
        for _ in range(round(duration / self.fruit.model.opt.timestep)):
            self.step()

    def let_pieces_rest(self) -> None:
        """
        Simulate for :attr:`rest_duration`.

        :raises PiecesScattered: If a piece moves farther than
            :attr:`scatter_tolerance` meanwhile.
        """
        fruit = self.fruit
        body_ids = [fruit.body_id(piece) for piece in fruit.pieces]
        start = fruit.data.xipos[body_ids].copy()
        self.simulate(self.rest_duration)
        moved = np.linalg.norm(fruit.data.xipos[body_ids] - start, axis=1)
        farthest = int(np.argmax(moved))
        if moved[farthest] > self.scatter_tolerance:
            raise PiecesScattered(
                fruit.pieces[farthest].body_name, float(moved[farthest])
            )

    # %% cutting

    def step_length(self) -> float:
        """
        :return: How far in metres the knife moves in one simulation step.
        """
        return self.stroke_speed * self.fruit.model.opt.timestep

    def cut_along(self, stroke: KnifeStroke) -> list[CutOff]:
        """
        Make a stroke with the knife. It cuts each piece once: what it has cut off may
        shift against the blade without being cut again.

        :param stroke: The stroke, given in the frame of :attr:`whole_fruit`.
        :return: The pieces the stroke made that are still whole.
        """
        fruit = self.fruit
        fruit_pose = self.whole_fruit
        plane = stroke.plane_in(fruit_pose)
        uncut = stroke.pieces_to_cut(fruit)
        materials = stroke.materials_cut(fruit)
        cut_off = []
        for blade_pose in stroke.blade_path(self, fruit_pose):
            self.move_mocap(SceneName.KNIFE, blade_pose.edge, blade_pose.quaternion())
            self.step()
            made = fruit.cut_with(self.blade, uncut, materials)
            uncut = [piece for piece in uncut if piece in fruit.pieces]
            if not made:
                continue
            self.largest_deviation = max(
                self.largest_deviation, self.deviation_from(plane, made, materials)
            )
            for piece in made:
                pose = fruit.pose_of(piece)
                centre = pose.to_world(piece.mass_distribution().centre[None])[0]
                on_normal_side = bool((centre - plane.point) @ plane.normal > 0)
                face_normal = pose.rotation().T @ plane.normal
                if on_normal_side:
                    face_normal = -face_normal
                cut_off.append(CutOff(piece, face_normal, on_normal_side))
            self.show_model()
        return [entry for entry in cut_off if entry.piece in fruit.pieces]

    def deviation_from(
        self,
        plane: CuttingPlane,
        pieces: list[FruitPiece],
        materials: frozenset[FruitMaterial],
    ) -> float:
        """
        :param plane: The plane of the blade that made the pieces, in the world.
        :param pieces: The pieces the blade has just made.
        :param materials: The materials the blade cut.
        :return: The farthest in metres a cut surface of the pieces lies from the plane.
        :raises CutOffKnifePlane: If that is farther than :attr:`cut_tolerance`.
        """
        largest = 0.0
        for piece in pieces:
            pose = self.fruit.pose_of(piece)
            for part in piece.parts:
                if part.material not in materials:
                    continue
                vertices = pose.to_world(part.triangle_mesh_of(part.solid).vertices)
                heights = (vertices - plane.point) @ plane.normal
                # A cut part lies on one side of the plane and reaches it.
                deviation = float(min(abs(heights.min()), abs(heights.max())))
                if deviation > self.cut_tolerance:
                    raise CutOffKnifePlane(piece.body_name, deviation)
                largest = max(largest, deviation)
        return largest

    # %% carrying pieces

    def centre_of(self, piece: FruitPiece) -> np.ndarray:
        """
        :param piece: A piece of the avocado.
        :return: Its centre of mass in the world, in metres.
        """
        return self.fruit.data.xipos[self.fruit.body_id(piece)].copy()

    def lay_aside(
        self,
        piece: FruitPiece,
        offset: np.ndarray,
        face_normal: np.ndarray | None,
        carrier: str | None = None,
        long_side: np.ndarray | None = None,
    ) -> None:
        """
        Release a piece, lift it, turn its cut face up and lay it down elsewhere on the
        board, as a hand would.

        :param piece: The piece.
        :param offset: How far to move it, in the world, in metres; only the horizontal
            part counts.
        :param face_normal: The direction its cut face looks in, in the frame of its
            body; None to lay it down the way it lies.
        :param carrier: The name of a mocap body, such as a spoon, that moves along with
            the piece; None if nothing does.
        :param long_side: The horizontal direction in the world its longest extent on
            the board is to lie along; None to leave it as turning its face up does.
        """
        pose = self.fruit.pose_of(piece)
        up = np.array([0.0, 0.0, 1.0])
        facing = up if face_normal is None else pose.rotation() @ face_normal
        axis = np.cross(facing, up)
        angle = float(np.arctan2(np.linalg.norm(axis), facing[2]))
        if np.linalg.norm(axis) < 1e-9:
            # Facing straight up or down: no turn, or roll over towards where it goes.
            axis = np.cross(up, offset)
        quaternion, rotation = self.turn(axis / np.linalg.norm(axis), angle)
        body_centre = piece.mass_distribution().centre
        relative = (piece.outline.vertices - body_centre) @ (
            rotation @ pose.rotation()
        ).T
        if long_side is not None:
            footprint = relative[:, :2] - relative[:, :2].mean(axis=0)
            longest = np.linalg.svd(footprint, full_matrices=False)[2][0]
            yaw = np.arctan2(long_side[1], long_side[0]) - np.arctan2(
                longest[1], longest[0]
            )
            yaw_quaternion, yaw_rotation = self.turn(up, float(yaw))
            quaternion = self.compose(yaw_quaternion, quaternion)
            rotation = yaw_rotation @ rotation
            relative = relative @ yaw_rotation.T
        quaternion = self.compose(quaternion, pose.quaternion)
        end = self.centre_of(piece) + np.array([offset[0], offset[1], 0.0])
        end[2] = self.board_size[2] + self.drop_height - relative[:, 2].min()
        self.carry(piece, end, quaternion, carrier)

    def carry(
        self,
        piece: FruitPiece,
        centre: np.ndarray,
        quaternion: np.ndarray,
        carrier: str | None = None,
        long_side: np.ndarray | None = None,
    ) -> None:
        """
        Release a piece, lift it, carry it over while turning it and lower it into a
        pose, as a hand would.

        :param piece: The piece.
        :param centre: Where its centre of mass goes, in the world, in metres.
        :param quaternion: The orientation its body ends in, scalar first.
        :param carrier: The name of a mocap body, such as a spoon, that moves along with
            the piece; None if nothing does.
        """
        pose = self.released(piece)
        start = self.centre_of(piece)
        axis, angle = self.rotation_between(pose.quaternion, quaternion)
        # High enough that turning it does not swing it down into what lies below.
        reach = np.linalg.norm(
            piece.outline.vertices - piece.mass_distribution().centre, axis=1
        ).max()
        carried = max(start[2], centre[2]) + max(self.lift_height, reach)
        held_by = self.grip_of(carrier, start, pose.quaternion)
        steps = round(self.lay_aside_duration / self.fruit.model.opt.timestep)
        for step in range(1, steps + 1):
            position, across = self.carried_position(
                start, centre, carried, step / steps
            )
            turning = self.turn(axis, across * angle)[0]
            self.place(piece, position, self.compose(turning, pose.quaternion), held_by)
            self.step()

    @staticmethod
    def carried_position(
        start: np.ndarray, end: np.ndarray, carried: float, progress: float
    ) -> tuple[np.ndarray, float]:
        """
        :param start: Where something carried starts, in the world, in metres.
        :param end: Where it ends.
        :param carried: How high it is carried across, in metres.
        :param progress: How far through the carrying it is, from 0 to 1.
        :return: Where it is then, and how far across, from 0 to 1: up for the first
            quarter, across for the middle half and down for the last quarter.
        """
        across = float(np.clip(2 * progress - 0.5, 0.0, 1.0))
        position = start + across * (end - start)
        position[2] = carried + (start[2] - carried) * max(0.0, 1 - 4 * progress)
        position[2] += (end[2] - carried) * max(0.0, 4 * progress - 3)
        return position, across

    def slide(
        self,
        piece: FruitPiece,
        displacement: np.ndarray,
        duration: float,
        carrier: str | None = None,
    ) -> None:
        """
        Release a piece and move it in a straight line without turning it.

        :param piece: The piece.
        :param displacement: How far to move it, in the world, in metres.
        :param duration: How long the move takes, in seconds.
        :param carrier: The name of a mocap body that moves along with the piece; None
            if nothing does.
        """
        pose = self.released(piece)
        start = self.centre_of(piece)
        held_by = self.grip_of(carrier, start, pose.quaternion)
        steps = round(duration / self.fruit.model.opt.timestep)
        for step in range(1, steps + 1):
            centre = start + step / steps * displacement
            self.place(piece, centre, pose.quaternion, held_by)
            self.step()

    def released(self, piece: FruitPiece) -> BodyPose:
        """
        :param piece: A piece of the avocado, which is released if it is held.
        :return: Where the piece is.
        """
        if self.fruit.is_held(piece):
            self.fruit.release(piece)
            self.show_model()
        return self.fruit.pose_of(piece)

    def grip_of(
        self, carrier: str | None, centre: np.ndarray, quaternion: np.ndarray
    ) -> CarrierGrip | None:
        """
        :param carrier: The name of a mocap body that carries a piece, or None.
        :param centre: The centre of mass of the piece in the world, in metres.
        :param quaternion: The orientation of the piece's body, scalar first.
        :return: Where the carrier is relative to the piece; None without a carrier.
        """
        if carrier is None:
            return None
        data = self.fruit.data
        mocap_id = self.mocap_id(carrier)
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, quaternion)
        offset = np.zeros(3)
        mujoco.mju_rotVecQuat(offset, data.mocap_pos[mocap_id] - centre, inverse)
        return CarrierGrip(
            carrier, offset, self.compose(inverse, data.mocap_quat[mocap_id])
        )

    def place(
        self,
        piece: FruitPiece,
        centre: np.ndarray,
        quaternion: np.ndarray,
        held_by: CarrierGrip | None,
    ) -> None:
        """
        Put a free piece, at rest, where its centre of mass is at a point, and the
        carrier holding it, if any, where it holds the piece.

        :param piece: The piece.
        :param centre: Where its centre of mass goes, in the world, in metres.
        :param quaternion: The orientation of its body, scalar first.
        :param held_by: Where the carrier is relative to the piece; None if none.
        """
        fruit = self.fruit
        joint = fruit.model.body_jntadr[fruit.body_id(piece)]
        first_position = fruit.model.jnt_qposadr[joint]
        first_dof = fruit.model.jnt_dofadr[joint]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        rotation = rotation.reshape(3, 3)
        body_centre = piece.mass_distribution().centre
        fruit.data.qpos[first_position : first_position + 3] = (
            centre - rotation @ body_centre
        )
        fruit.data.qpos[first_position + 3 : first_position + 7] = quaternion
        fruit.data.qvel[first_dof : first_dof + 6] = 0.0
        if held_by is None:
            return
        self.move_mocap(
            held_by.carrier,
            centre + rotation @ held_by.offset,
            self.compose(quaternion, held_by.orientation),
        )

    @staticmethod
    def turn(axis: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """
        :param axis: A unit vector.
        :param angle: An angle in radians.
        :return: The rotation by the angle about the axis, as a quaternion, scalar
            first, and as a matrix.
        """
        quaternion, matrix = np.zeros(4), np.zeros(9)
        mujoco.mju_axisAngle2Quat(quaternion, axis, angle)
        mujoco.mju_quat2Mat(matrix, quaternion)
        return quaternion, matrix.reshape(3, 3)

    @staticmethod
    def rotation_between(
        start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        :param start: An orientation, as a quaternion, scalar first.
        :param end: Another.
        :return: The axis in the world and the angle in radians of the rotation that
            turns *start* into *end*.
        """
        inverse, turn = np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(inverse, start)
        product = SolidAvocadoDemonstration.compose(end, inverse)
        mujoco.mju_quat2Vel(turn, product, 1.0)
        angle = float(np.linalg.norm(turn))
        if angle == 0:
            return np.array([0.0, 0.0, 1.0]), 0.0
        return turn / angle, angle

    @staticmethod
    def quaternion_of(rotation: np.ndarray) -> np.ndarray:
        """
        :param rotation: A rotation matrix.
        :return: The rotation as a quaternion, scalar first.
        """
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        return quaternion

    @staticmethod
    def compose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """
        :param first: A quaternion, scalar first.
        :param second: Another.
        :return: The rotation *second* followed by *first*.
        """
        product = np.zeros(4)
        mujoco.mju_mulQuat(product, first, second)
        return product


@dataclass
class CarrierGrip:
    """
    Where a mocap body carrying a piece is, relative to the piece.
    """

    carrier: str
    """
    The name of the mocap body.
    """

    offset: np.ndarray
    """
    Its position relative to the piece's centre of mass, in the axes of the piece's
    body, in metres.
    """

    orientation: np.ndarray
    """
    Its orientation relative to the piece's body, scalar first.
    """

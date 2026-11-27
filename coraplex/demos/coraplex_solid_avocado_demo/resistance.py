"""
The force a fruit's material puts up against a knife's blade or a spoon's bowl going
through it, as firm as its texture makes it.

The force acts on the tool, where the tool moves under forces, and its reaction on the
piece the tool is in, where that piece is free.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import mujoco
import numpy as np

from cutting import CuttableFruit, CuttingPlane, FruitPiece, KnifeBlade, SpoonBowl
from fruit_material import FruitMaterial

# %% a resistance and where it acts


@dataclass(kw_only=True)
class PushBack:
    """
    A force the fruit's material exerts on a tool, and where.
    """

    piece: FruitPiece
    """
    The piece whose material exerts it.
    """

    force: np.ndarray
    """
    The force on the tool in the world, in newtons.
    """

    point: np.ndarray
    """
    Where it acts, in the world, in metres.
    """


@dataclass(kw_only=True)
class MaterialResistance(ABC):
    """
    The force a fruit's material puts up against a tool on a body of the fruit's scene
    going through it.

    Like dry friction, it opposes the tool's motion through the material with up to its
    full strength, fading out as the tool comes to rest in it.
    """

    fruit: CuttableFruit
    """
    The fruit, whose :attr:`CuttableFruit.texture` gives its strength.
    """

    tool_body: str
    """
    The name of the body the tool is on.
    """

    pushes_pieces: bool = True
    """
    Whether the tool pushes the free piece it is in back with the force the piece puts
    up against it; a fruit nothing holds is then pushed away by the tool.
    """

    resting_speed: float = 0.001
    """
    The speed in metres per second below which the resistance fades out, so that a tool
    pushed less hard than the material holds comes to rest in it.
    """

    force: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    """
    The force in newtons the material exerted on the tool in the world at the last
    :meth:`apply`.
    """

    last_positions: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    """
    Where the tool's body and the pieces' bodies were at the last :meth:`apply`, in the
    world, in metres, by the name of the body.
    """

    last_time: float = field(init=False, default=0.0)
    """
    The simulated time of the last :meth:`apply`, in seconds.
    """

    pushed_bodies: list[int] = field(init=False, default_factory=list)
    """
    The ids of the bodies the last :meth:`apply` exerted a force on.
    """

    @abstractmethod
    def push_back(self, *, velocities: dict[str, np.ndarray]) -> list[PushBack]:
        """
        :param velocities: How fast the tool's body and each piece's body have moved
            since the last :meth:`apply`, by the name of the body.
        :return: The forces the fruit's pieces exert on the tool now; none where the
            fruit has no texture or the tool is in no piece.
        """

    def velocities_since_last(self) -> dict[str, np.ndarray]:
        """
        :return: The linear velocity in metres per second, in the world, of the tool's
            body and of each piece's body, by the name of the body: its own where it
            moves under forces, else how fast it has been moved since the last
            :meth:`apply`, as a body led by hand has no velocity of its own.
        """
        model, data = self.fruit.model, self.fruit.data
        names = [self.tool_body, *(piece.body_name for piece in self.fruit.pieces)]
        time = float(data.time)
        elapsed = time - self.last_time
        velocities, positions = {}, {}
        own = np.zeros(6)
        for name in names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            positions[name] = data.xpos[body_id].copy()
            if model.body_dofnum[body_id] > 0:
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, own, 0
                )
                velocities[name] = own[3:].copy()
            elif name in self.last_positions and elapsed > 0:
                velocities[name] = (
                    positions[name] - self.last_positions[name]
                ) / elapsed
            else:
                velocities[name] = np.zeros(3)
        self.last_positions, self.last_time = positions, time
        return velocities

    def stable_force(self, *, piece: FruitPiece, speed: float) -> float:
        """
        :param piece: A piece the tool goes through.
        :param speed: How fast the tool moves through it, in metres per second.
        :return: The largest force in newtons that brings the tool and the piece no
            further than to the same speed within one step, as dry friction does;
            unbounded if neither moves under forces.
        """
        model = self.fruit.model
        bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.tool_body)]
        if self.pushes_pieces:
            bodies.append(self.fruit.body_id(piece))
        inverse_mass = sum(
            1.0 / model.body_mass[body_id]
            for body_id in bodies
            if model.body_dofnum[body_id] > 0
        )
        if inverse_mass == 0.0:
            return math.inf
        return speed / (inverse_mass * model.opt.timestep)

    def fading(self, *, speed: float) -> float:
        """
        :param speed: How fast the tool moves into the material, in metres per second.
        :return: The share of its full strength the material resists with at that
            speed.
        """
        return math.tanh(max(speed, 0.0) / self.resting_speed)

    def apply(self) -> np.ndarray:
        """
        Exert the forces of :meth:`push_back` for the next step: on the tool where its
        body moves under forces, and their reactions on the pieces that are free.

        :return: The force in newtons the material exerts on the tool, in the world.
        """
        model, data = self.fruit.model, self.fruit.data
        for body_id in self.pushed_bodies:
            if body_id < model.nbody:
                data.xfrc_applied[body_id] = 0.0
        self.pushed_bodies = []
        pushes = self.push_back(velocities=self.velocities_since_last())
        tool_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.tool_body)
        for push in pushes:
            piece_id = self.fruit.body_id(push.piece)
            pushed = [(tool_id, 1.0)] + [(piece_id, -1.0)] * self.pushes_pieces
            for body_id, sign in pushed:
                if model.body_dofnum[body_id] == 0:
                    continue
                arm = push.point - data.xipos[body_id]
                data.xfrc_applied[body_id, :3] += sign * push.force
                data.xfrc_applied[body_id, 3:] += sign * np.cross(arm, push.force)
                self.pushed_bodies.append(body_id)
        self.force = sum((push.force for push in pushes), np.zeros(3))
        return self.force


# %% the knife


@dataclass(kw_only=True)
class CuttingResistance(MaterialResistance):
    """
    The force a fruit's skin and flesh put up against a knife's cutting edge pushing
    through them: their cutting resistance times the length of edge in each, less as
    the blade draws along its edge while it pushes.
    """

    blade: KnifeBlade
    """
    The blade, on :attr:`tool_body`.
    """

    edge_spacing: float = 0.001
    """
    The distance in metres between the points of the cutting edge at which it is told
    which material it is in.
    """

    face_margin: float = 0.001
    """
    How far in metres the blade's plane may reach into a piece and still lie along one
    of its faces, in a cut made before, where there is nothing left to cut.
    """

    def push_back(self, *, velocities: dict[str, np.ndarray]) -> list[PushBack]:
        texture = self.fruit.texture
        if texture is None:
            return []
        model, data = self.fruit.model, self.fruit.data
        site_id = self.blade.site_id(model)
        axes = data.site_xmat[site_id].reshape(3, 3)
        along_edge, cutting_direction = axes[:, 0], -axes[:, 1]
        offsets = np.arange(
            -self.blade.edge_length / 2, self.blade.edge_length / 2, self.edge_spacing
        )
        edge = data.site_xpos[site_id] + offsets[:, None] * along_edge
        resistance = {
            FruitMaterial.SKIN: texture.skin_cutting_resistance,
            FruitMaterial.FLESH: texture.flesh_cutting_resistance,
        }
        plane = self.blade.plane(model, data)
        pushes = []
        for piece in self.fruit.pieces:
            through = velocities[self.tool_body] - velocities[piece.body_name]
            pushing = float(through @ cutting_direction)
            if pushing <= 0.0:
                continue
            drawing = abs(float(through @ along_edge))
            strength = self.fading(speed=pushing) * (
                1.0 - texture.texture.slicing_relief * drawing / (drawing + pushing)
            )
            if self.lies_along_face(piece=piece, plane=plane):
                continue
            inside = self.fruit.materials_along(piece=piece, plane=plane, points=edge)
            held_back = sum(
                resistance[material] * np.count_nonzero(points) * self.edge_spacing
                for material, points in inside.items()
                if material in resistance
            )
            if held_back <= 0.0:
                continue
            within = np.any(list(inside.values()), axis=0)
            magnitude = min(
                strength * held_back, self.stable_force(piece=piece, speed=pushing)
            )
            pushes.append(
                PushBack(
                    piece=piece,
                    force=-magnitude * cutting_direction,
                    point=edge[within].mean(axis=0),
                )
            )
        return pushes

    def lies_along_face(self, *, piece: FruitPiece, plane: CuttingPlane) -> bool:
        """
        :param piece: A piece of the fruit.
        :param plane: The blade's plane, in the world.
        :return: Whether the plane reaches no farther than :attr:`face_margin` into
            the piece on one of its sides.
        """
        local = plane.seen_from(self.fruit.pose_of(piece))
        heights = (piece.outline.vertices - local.point) @ local.normal
        return min(heights.max(), -heights.min()) < self.face_margin


# %% the spoon


@dataclass(kw_only=True)
class ScoopingResistance(MaterialResistance):
    """
    The force a fruit's flesh puts up against a spoon's bowl pushing through it: its
    scooping resistance over the bowl's face across its way, as much of it as pushes
    into flesh.

    A scoop, lying in the bowl, puts up nothing.
    """

    bowl: SpoonBowl
    """
    The bowl, on :attr:`tool_body`.
    """

    lead: float = 0.001
    """
    How far ahead in metres of the bowl the room its face pushes into is taken.
    """

    def push_back(self, *, velocities: dict[str, np.ndarray]) -> list[PushBack]:
        texture = self.fruit.texture
        if texture is None:
            return []
        bowl_pose = self.bowl.pose(self.fruit.model, self.fruit.data)
        flesh = frozenset({FruitMaterial.FLESH})
        semi_axes = np.asarray(self.bowl.half_size)
        pushes = []
        for piece in self.fruit.pieces:
            if piece.body_name in self.fruit.scoops:
                continue
            through = velocities[self.tool_body] - velocities[piece.body_name]
            speed = float(np.linalg.norm(through))
            strength = self.fading(speed=speed)
            if strength <= 0.0:
                continue
            across = bowl_pose.rotation().T @ (through / speed)
            bowl = self.bowl.solid()
            ahead = bowl.translate((self.lead * across).tolist()) - bowl
            pushed_into = self.fruit.filling(
                piece,
                ahead.transform(
                    bowl_pose.placement_in(frame=self.fruit.pose_of(piece))
                ),
                flesh,
            )
            if pushed_into <= 0.0:
                continue
            face = (
                math.pi
                * float(np.prod(semi_axes))
                * float(np.linalg.norm(across / semi_axes))
            )
            magnitude = min(
                strength * texture.scooping_resistance * face * pushed_into,
                self.stable_force(piece=piece, speed=speed),
            )
            pushes.append(
                PushBack(
                    piece=piece,
                    force=-magnitude * through / speed,
                    point=bowl_pose.position,
                )
            )
        return pushes

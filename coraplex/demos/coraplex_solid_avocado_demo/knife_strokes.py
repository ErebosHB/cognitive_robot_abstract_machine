"""
The strokes a knife makes through a solid fruit, each in one plane, as the poses its
blade passes through.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import mujoco
import numpy as np
from typing_extensions import TYPE_CHECKING, Iterator

from cutting import BodyPose, CuttableFruit, CuttingPlane, FruitPiece
from fruit_material import FruitMaterial

if TYPE_CHECKING:
    from solid_avocado_demonstration import SolidAvocadoDemonstration

# %% the blade


@dataclass
class BladePose:
    """
    Where the knife's blade is.
    """

    edge: np.ndarray
    """
    The middle of the cutting edge in the world, in metres.
    """

    along_edge: np.ndarray
    """
    The direction the cutting edge runs in, in the world.
    """

    into_blade: np.ndarray
    """
    The direction from the cutting edge into the blade, opposite to the one it cuts in,
    in the world.
    """

    def quaternion(self) -> np.ndarray:
        """
        :return: The orientation of the blade's site in the world, scalar first.
        """
        across_blade = np.cross(self.along_edge, self.into_blade)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(
            quaternion,
            np.column_stack([self.along_edge, self.into_blade, across_blade]).ravel(),
        )
        return quaternion


# %% strokes


@dataclass
class KnifeStroke(ABC):
    """
    One stroke of the knife through the fruit, in one plane.
    """

    through: np.ndarray
    """
    A point the blade's plane goes through, in the frame of the whole fruit, in metres.
    """

    across_blade: np.ndarray
    """
    The normal of the blade's plane, in the frame of the whole fruit.
    """

    def plane_in(self, fruit_pose: BodyPose) -> CuttingPlane:
        """
        :param fruit_pose: The pose of the whole fruit before the first stroke.
        :return: The blade's plane in the world.
        """
        return CuttingPlane(
            fruit_pose.to_world(self.through[None])[0],
            fruit_pose.rotation() @ self.across_blade,
        )

    def materials_cut(self, fruit: CuttableFruit) -> frozenset[FruitMaterial]:
        """
        :param fruit: The fruit.
        :return: The materials the stroke cuts through.
        """
        return fruit.materials_cut(None)

    @abstractmethod
    def pieces_to_cut(self, fruit: CuttableFruit) -> list[FruitPiece]:
        """
        :param fruit: The fruit before the stroke.
        :return: The pieces the stroke may cut.
        """

    @abstractmethod
    def blade_path(
        self, demonstration: SolidAvocadoDemonstration, fruit_pose: BodyPose
    ) -> Iterator[BladePose]:
        """
        :param demonstration: The demonstration the stroke is made in, which simulates
            a step after each pose.
        :param fruit_pose: The pose of the whole fruit before the first stroke.
        :return: The poses the blade passes through, one per simulation step.
        """


@dataclass
class DownwardStroke(KnifeStroke):
    """
    The knife moves straight down through the pieces it cuts until just below them, or
    until its edge reaches a part no knife cuts.
    """

    pieces: list[FruitPiece] | None = None
    """
    The pieces the stroke may cut; every piece the fruit is in when it starts if None.
    """

    def pieces_to_cut(self, fruit: CuttableFruit) -> list[FruitPiece]:
        return list(fruit.pieces if self.pieces is None else self.pieces)

    def blade_path(
        self, demonstration: SolidAvocadoDemonstration, fruit_pose: BodyPose
    ) -> Iterator[BladePose]:
        plane = self.plane_in(fruit_pose)
        along_edge = np.cross([0.0, 0.0, 1.0], plane.normal)
        along_edge /= np.linalg.norm(along_edge)
        into_blade = np.cross(plane.normal, along_edge)
        fruit = demonstration.fruit
        fruit_top = max(
            fruit.pose_of(piece).to_world(piece.outline.vertices)[:, 2].max()
            for piece in fruit.pieces
        )
        bottom = min(
            fruit.pose_of(piece).to_world(piece.outline.vertices)[:, 2].min()
            for piece in self.pieces_to_cut(fruit)
        )
        # The edge's middle starts above the fruit, over the plane's given point.
        climb = (
            fruit_top + demonstration.stroke_clearance - plane.point[2]
        ) / into_blade[2]
        edge = plane.point + climb * into_blade
        step_length = demonstration.step_length()
        while edge[2] > bottom - demonstration.stroke_overshoot:
            yield BladePose(edge, along_edge, into_blade)
            room = demonstration.blade.room_ahead(
                fruit.model, fruit.data, fruit.section_outline(plane, fruit.uncuttable)
            )
            if room < step_length:
                return
            edge = edge - step_length * into_blade


# %% strokes round an outline


@dataclass
class OutlineCircuit:
    """
    The way of the knife's cutting edge once round an outline in the blade's plane,
    facing the outline's middle at a fixed distance from the outline.
    """

    centre: np.ndarray
    """
    The middle of the outline, in the world, in metres.
    """

    normal: np.ndarray
    """
    The normal of the blade's plane, in the world.
    """

    first_direction: np.ndarray
    """
    The direction in the plane from :attr:`centre` to where the edge starts going round.
    """

    outline: np.ndarray
    """
    The corners of the outline in the plane, in the world, one row each.
    """

    clearance: float
    """
    How far in metres the edge keeps outside the outline; below zero, how far inside.
    """

    def direction(self, angle: float) -> np.ndarray:
        """
        :param angle: How far round the edge has gone, in radians.
        :return: The direction from :attr:`centre` to the edge there.
        """
        sideways = np.cross(self.normal, self.first_direction)
        return np.cos(angle) * self.first_direction + np.sin(angle) * sideways

    def edge_distance(self, direction: np.ndarray) -> float:
        """
        :param direction: A direction in the plane from :attr:`centre`.
        :return: How far from the centre in metres the edge, facing the centre from that
            direction, keeps :attr:`clearance` from the outline.
        """
        reach = ((self.outline - self.centre) @ direction).max()
        return float(reach) + self.clearance

    def blade_pose(self, direction: np.ndarray, distance: float) -> BladePose:
        """
        :param direction: A direction in the plane from :attr:`centre`.
        :param distance: How far from the centre the edge is, in metres.
        :return: The blade there, its edge facing the centre.
        """
        return BladePose(
            self.centre + distance * direction,
            np.cross(direction, self.normal),
            direction,
        )


@dataclass
class CircuitStroke(KnifeStroke, ABC):
    """
    The knife comes in from one side until its edge reaches an outline of the fruit in
    the blade's plane, goes once round it, as a cook turns the fruit against the knife,
    and comes out again.
    """

    approach: np.ndarray
    """
    The direction, in the frame of the whole fruit, from the outline to where the knife
    comes in from.
    """

    @abstractmethod
    def circled_materials(self, fruit: CuttableFruit) -> frozenset[FruitMaterial]:
        """
        :param fruit: The fruit.
        :return: The materials of the parts whose outline the edge goes round.
        """

    @abstractmethod
    def clearance(self, demonstration: SolidAvocadoDemonstration) -> float:
        """
        :param demonstration: The demonstration the stroke is made in.
        :return: How far in metres the edge keeps outside the outline; below zero, how
            far inside.
        """

    def blade_path(
        self, demonstration: SolidAvocadoDemonstration, fruit_pose: BodyPose
    ) -> Iterator[BladePose]:
        circuit = self.circuit(demonstration, fruit_pose)
        fruit = demonstration.fruit
        outline = np.vstack(
            [
                fruit.pose_of(piece).to_world(piece.outline.vertices)
                for piece in self.pieces_to_cut(fruit)
            ]
        )
        start = np.linalg.norm(outline - circuit.centre, axis=1).max()
        start += demonstration.stroke_clearance
        step_length = demonstration.step_length()
        direction = circuit.first_direction
        distance = start
        while distance > circuit.edge_distance(direction):
            yield circuit.blade_pose(direction, distance)
            distance = max(distance - step_length, circuit.edge_distance(direction))
        angle = 0.0
        while angle < 2 * np.pi + demonstration.overturn:
            direction = circuit.direction(angle)
            distance = circuit.edge_distance(direction)
            yield circuit.blade_pose(direction, distance)
            angle += step_length / distance
        while distance < start:
            distance += step_length
            yield circuit.blade_pose(direction, distance)

    def circuit(
        self, demonstration: SolidAvocadoDemonstration, fruit_pose: BodyPose
    ) -> OutlineCircuit:
        """
        :param demonstration: The demonstration the stroke is made in.
        :param fruit_pose: The pose of the whole fruit before the first stroke.
        :return: The way of the cutting edge round the outline as it lies now.
        """
        plane = self.plane_in(fruit_pose)
        fruit = demonstration.fruit
        outline = fruit.section_outline(plane, self.circled_materials(fruit))
        first_direction = fruit_pose.rotation() @ self.approach
        first_direction -= (first_direction @ plane.normal) * plane.normal
        return OutlineCircuit(
            outline.mean(axis=0),
            plane.normal,
            first_direction / np.linalg.norm(first_direction),
            outline,
            self.clearance(demonstration),
        )


@dataclass
class StrokeAroundPit(CircuitStroke):
    """
    The knife cuts through the piece holding the pit, going round the pit just clear of
    it.
    """

    def circled_materials(self, fruit: CuttableFruit) -> frozenset[FruitMaterial]:
        return fruit.uncuttable

    def clearance(self, demonstration: SolidAvocadoDemonstration) -> float:
        return demonstration.pit_clearance

    def pieces_to_cut(self, fruit: CuttableFruit) -> list[FruitPiece]:
        return [fruit.piece_with(FruitMaterial.PIT)]


@dataclass
class ScoringStroke(CircuitStroke):
    """
    The knife scores the skin of every piece along the outline of the fruit, just deep
    enough to cut through the skin and no other material.
    """

    depth: float
    """
    How far under the fruit's surface the cutting edge goes, in metres.
    """

    def circled_materials(self, fruit: CuttableFruit) -> frozenset[FruitMaterial]:
        return frozenset(FruitMaterial)

    def clearance(self, demonstration: SolidAvocadoDemonstration) -> float:
        return -self.depth

    def materials_cut(self, fruit: CuttableFruit) -> frozenset[FruitMaterial]:
        """
        The skin only; see :meth:`KnifeStroke.materials_cut`.
        """
        return fruit.materials_cut(frozenset({FruitMaterial.SKIN}))

    def pieces_to_cut(self, fruit: CuttableFruit) -> list[FruitPiece]:
        return list(fruit.pieces)

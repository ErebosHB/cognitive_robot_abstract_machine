"""
The solid avocado held lying above the cutting board by two poles stuck into its
underside, so that it stays where it is while it is cut, peeled or scooped out.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

import manifold3d
import mujoco
import numpy as np

from cutting import FruitMaterial, FruitPiece
from exceptions import FruitNotCut
from knife_strokes import StrokeAroundPit
from solid_avocado_demonstration import SolidAvocadoDemonstration

# %% the held avocado


@dataclass
class HeldAvocadoDemonstration(SolidAvocadoDemonstration, ABC):
    """
    The avocado lies on its side, its long axis along the world's x axis and its stem
    pointing that way, on two upright poles whose tips are stuck into its underside.
    Its frame's x axis points down.
    """

    axis_height: float = 0.09
    """
    How high above the board the avocado's long axis lies, in metres.
    """

    pole_positions: tuple[float, float] = (0.017, 0.085)
    """
    Where along the avocado's long axis the poles stand, measured from its base, in
    metres; both clear of the pit.
    """

    pole_radius: float = 0.003
    """
    The radius of the poles in metres.
    """

    pole_tip_depth: float = 0.015
    """
    How far below the avocado's long axis the tips of the poles end, in metres.
    """

    fruit_length: float = 0.111
    """
    The length of the avocado from base to stem, in metres.
    """

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.09)
    """
    The point the viewer's camera looks at, in metres: the held avocado.
    """

    def place_avocado(self, avocado: mujoco.MjsBody) -> None:
        """
        Put the avocado's middle above the middle of the board.
        """
        avocado.pos = [
            -self.fruit_length / 2,
            0,
            self.board_size[2] + self.axis_height,
        ]
        avocado.quat = [np.sqrt(0.5), 0, np.sqrt(0.5), 0]

    def add_equipment(self, spec: mujoco.MjSpec) -> None:
        """
        Add the poles, standing on the board.
        """
        for start, end in self.pole_ends():
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*start, *end],
                size=[self.pole_radius, 0, 0],
                rgba=[0.7, 0.72, 0.75, 1],
                friction=[0.6, 0.01, 0.001],
            )

    def holders(self) -> list[manifold3d.Manifold]:
        """
        The poles; see :meth:`SolidAvocadoDemonstration.holders`.
        """
        poles = []
        for start, end in self.pole_ends():
            shaft = manifold3d.Manifold.cylinder(
                float(end[2] - start[2]), self.pole_radius, self.pole_radius, 24
            ).translate(start.tolist())
            tip = manifold3d.Manifold.sphere(self.pole_radius, 24).translate(
                end.tolist()
            )
            poles.append(shaft + tip)
        return poles

    def pole_ends(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        :return: The foot and the tip of each pole's axis, in the world, in metres.
        """
        board_top = self.board_size[2]
        tip_height = board_top + self.axis_height - self.pole_tip_depth
        return [
            (
                np.array([position - self.fruit_length / 2, 0.0, board_top]),
                np.array([position - self.fruit_length / 2, 0.0, tip_height]),
            )
            for position in self.pole_positions
        ]


# %% halving it


@dataclass
class HalvedPieces:
    """
    What halving the held avocado and taking its pit out leaves.
    """

    upper_half: FruitPiece
    """
    The upper half, laid aside on the board cut face up.
    """

    lower_half: FruitPiece
    """
    The lower half, still held by the poles.
    """


@dataclass
class HalvingAvocadoDemonstration(HeldAvocadoDemonstration, ABC):
    """
    The knife first halves the held avocado level round its pit; the upper half and
    then the pit are laid aside.
    """

    halving: StrokeAroundPit = field(
        default_factory=lambda: StrokeAroundPit(
            np.array([0.003, 0.002, 0.0485]),
            np.array([-1.0, 0.0, 0.0]),
            approach=np.array([0.0, -1.0, 0.0]),
        )
    )
    """
    The stroke that halves the avocado, in a level plane through the pit's middle.
    """

    upper_half_set_aside: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.09, 0.0])
    )
    """
    How far the upper half is laid aside, in the world, in metres.
    """

    pit_place: np.ndarray = field(default_factory=lambda: np.array([0.165, 0.0, 0.0]))
    """
    Where on the board the pit is laid, in the world, in metres; only its horizontal
    part counts, as the pit may lie in either half.
    """

    def halve(self) -> HalvedPieces:
        """
        Halve the avocado round the pit, lay the upper half aside cut face up and take
        the pit out.

        :return: The halves.
        :raises FruitNotCut: If the stroke leaves the avocado whole.
        """
        fruit = self.fruit
        cut_off = self.cut_along(self.halving)
        if len(fruit.pieces) != 2:
            raise FruitNotCut(2, len(fruit.pieces))
        upper = next(entry for entry in cut_off if entry.on_normal_side)
        self.let_pieces_rest()
        self.lay_aside(upper.piece, self.upper_half_set_aside, upper.face_normal)
        self.let_pieces_rest()
        pit = fruit.separate(fruit.piece_with(FruitMaterial.PIT), FruitMaterial.PIT)
        self.show_model()
        self.lay_aside(pit, self.pit_place - self.centre_of(pit), None)
        self.let_pieces_rest()
        # Taking the pit out made a new piece of the half it lay in.
        halves = [piece for piece in fruit.pieces if piece is not pit]
        upper_half = next(piece for piece in halves if not fruit.is_held(piece))
        lower_half = next(piece for piece in halves if fruit.is_held(piece))
        return HalvedPieces(upper_half, lower_half)

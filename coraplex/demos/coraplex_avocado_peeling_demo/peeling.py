"""
Peeling, cutting, pitting and slicing a fruit whose skin is made of tiles held by
equality constraints and whose flesh is made of welded slices, as the produce models of
``private_meshes`` describe them in ``mujoco_peel.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np
from exceptions import FruitPartNotFound
from typing_extensions import Any

# %% the peeling description


@dataclass
class SkinTileDescription:
    """
    One skin tile and the constraint that holds it to its flesh.
    """

    body: str
    """
    The name of the tile's body.
    """

    geom: str
    """
    The name of the tile's geom.
    """

    equality: str
    """
    The name of the equality constraint holding the tile to its flesh.
    """

    flesh: str
    """
    The name of the flesh body the tile is held to.
    """

    strip: str
    """
    The name of the strip, running from stem to base, that the tile belongs to.
    """

    index: int
    """
    The position of the tile in its strip, counted from the stem.
    """

    outward_normal: np.ndarray
    """
    The direction pointing out of the fruit at the tile, in the frame of its flesh body.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SkinTileDescription:
        """
        :param data: One entry of the ``tiles`` list of a peeling description.
        :return: The tile it describes.
        """
        return cls(
            body=data["body"],
            geom=data["geom"],
            equality=data["equality"],
            flesh=data["flesh"],
            strip=data["strip"],
            index=data["index"],
            outward_normal=np.array(data["normal"]),
        )


class FruitSide(StrEnum):
    """
    The two halves of a fruit, on either side of the plane it is cut along.
    """

    LEFT = "left"
    RIGHT = "right"


@dataclass
class SliceLink:
    """
    An equality constraint joining something across a cut between two flesh slices of
    one half: the weld between the slices, or the joint between two pieces of a skin
    strip.
    """

    equality: str
    """
    The name of the equality constraint.
    """

    side: FruitSide
    """
    The half whose slices it joins.
    """

    slice_index: int
    """
    The slice, counted from the stem, after which the cut runs.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SliceLink:
        """
        :param data: One entry of the ``slices`` or ``skinJoins`` list of a peeling
            description.
        :return: The link it describes.
        """
        return cls(
            equality=data["equality"],
            side=FruitSide(data["side"]),
            slice_index=data["slice"],
        )


@dataclass
class PeelingDescription:
    """
    The parts and constraints of a peelable fruit model.
    """

    tiles: list[SkinTileDescription]
    """
    The skin tiles, strip by strip from stem to base.
    """

    flesh_slices: dict[FruitSide, list[str]]
    """
    The names of the flesh slices of each half, from stem to base.
    """

    cut_equalities: list[str]
    """
    The names of the welds holding the two flesh halves together.
    """

    slice_links: list[SliceLink]
    """
    The constraints joining neighbouring slices of a half, flesh and skin.
    """

    pit_equality: str
    """
    The name of the weld holding the pit in its cavity.
    """

    pit_body: str
    """
    The name of the pit's body.
    """

    pit_flesh: str
    """
    The name of the flesh slice the pit is held to.
    """

    pit_side: FruitSide
    """
    The half that holds the pit.
    """

    released_skin_affinity: int
    """
    The contact affinity bits of a skin tile once it no longer sits on the flesh.
    """

    released_pit_affinity: int
    """
    The contact affinity bits of the pit once it has left its cavity.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PeelingDescription:
        """
        :param data: The parsed content of a ``mujoco_peel.json``.
        :return: The description it holds.
        """
        constraints = data["constraints"]
        return cls(
            tiles=[SkinTileDescription.from_json(tile) for tile in data["tiles"]],
            flesh_slices={
                FruitSide(side): names for side, names in data["flesh"].items()
            },
            cut_equalities=constraints["cut"]["equalities"],
            slice_links=[
                SliceLink.from_json(link)
                for link in constraints["slices"] + data["skinJoins"]
            ],
            pit_equality=constraints["pit"]["equality"],
            pit_body=constraints["pit"]["body"],
            pit_flesh=constraints["pit"]["flesh"],
            pit_side=FruitSide(constraints["pit"]["side"]),
            released_skin_affinity=data["releaseCollision"]["skin"],
            released_pit_affinity=data["releaseCollision"]["pit"],
        )

    @classmethod
    def from_file(cls, path: Path) -> PeelingDescription:
        """
        :param path: A ``mujoco_peel.json`` file.
        :return: The description it holds.
        """
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


# %% the fruit in a simulation


@dataclass
class SkinTile:
    """
    A skin tile resolved to the ids of a compiled MuJoCo model.
    """

    description: SkinTileDescription
    """
    The tile as the peeling description names it.
    """

    equality_id: int
    """
    The id of the constraint holding the tile to its flesh.
    """

    geom_id: int
    """
    The id of the tile's geom.
    """

    flesh_body_id: int
    """
    The id of the flesh body the tile is held to.
    """


@dataclass
class SkinSeparations:
    """
    How far each skin tile of a fruit has moved away from its seat on the flesh, in the
    order of :attr:`PeelableFruit.tiles`; zero for tiles no longer attached.
    """

    outward: np.ndarray
    """
    The distances in metres the tiles are lifted off the flesh; negative when pressed
    in.
    """

    along_surface: np.ndarray
    """
    The distances in metres the tiles are dragged along the flesh.
    """


@dataclass
class PeelableFruit:
    """
    A fruit in a running MuJoCo simulation whose skin tears off where it is pulled.

    Call :meth:`release_pulled_tiles` after every simulation step. A tile is released once
    it stays lifted :attr:`release_distance` off its seat or dragged :attr:`release_shear`
    along it for :attr:`release_steps` steps; pressing a tile into the flesh never
    releases it.

    .. note:: The criterion is a distance because MuJoCo scales constraint forces with the
        mass of the constrained bodies, so a light tile never reaches a meaningful force.
    """

    model: mujoco.MjModel
    """
    The compiled model holding the fruit.
    """

    data: mujoco.MjData
    """
    The simulation state of :attr:`model`.
    """

    description: PeelingDescription
    """
    The parts and constraints of the fruit.
    """

    name_prefix: str = ""
    """
    The prefix the fruit's names were given when it was attached to the scene.
    """

    release_distance: float = 0.0015
    """
    How far in metres a tile may be lifted off the flesh before it tears off.
    """

    release_shear: float = 0.003
    """
    How far in metres a tile may be dragged along the flesh before it tears off.
    """

    release_steps: int = 5
    """
    How many simulation steps in a row a tile must stay past a limit before it tears
    off, so a momentary jolt passed through the constraints tears nothing.
    """

    tiles: list[SkinTile] = field(init=False)
    """
    The skin tiles, resolved to ids of :attr:`model`.
    """

    equality_ids: np.ndarray = field(init=False)
    """
    The ids of the constraints holding the :attr:`tiles` to their flesh, in their order.
    """

    flesh_body_ids: np.ndarray = field(init=False)
    """
    The ids of the flesh bodies the :attr:`tiles` are held to, in their order.
    """

    outward_normals: np.ndarray = field(init=False)
    """
    The outward directions at the :attr:`tiles` in their flesh frames, one row each.
    """

    steps_pulled: np.ndarray = field(init=False)
    """
    For how many steps in a row each of the :attr:`tiles` has been past a limit.
    """

    def __post_init__(self) -> None:
        self.tiles = [
            SkinTile(
                description=tile,
                equality_id=self.id_of(mujoco.mjtObj.mjOBJ_EQUALITY, tile.equality),
                geom_id=self.id_of(mujoco.mjtObj.mjOBJ_GEOM, tile.geom),
                flesh_body_id=self.id_of(mujoco.mjtObj.mjOBJ_BODY, tile.flesh),
            )
            for tile in self.description.tiles
        ]
        self.equality_ids = np.array([tile.equality_id for tile in self.tiles])
        self.flesh_body_ids = np.array([tile.flesh_body_id for tile in self.tiles])
        self.outward_normals = np.array(
            [tile.description.outward_normal for tile in self.tiles]
        )
        self.steps_pulled = np.zeros(len(self.tiles), dtype=int)

    def id_of(self, object_type: mujoco.mjtObj, name: str) -> int:
        """
        :param object_type: The kind of MuJoCo object.
        :param name: The object's name without the fruit's prefix.
        :return: The object's id in :attr:`model`.
        """
        full_name = self.name_prefix + name
        object_id = mujoco.mj_name2id(self.model, object_type, full_name)
        if object_id < 0:
            raise FruitPartNotFound(full_name)
        return object_id

    def is_attached(self, tile: SkinTile) -> bool:
        """
        :param tile: A tile of this fruit.
        :return: Whether the tile still sits on its flesh.
        """
        return bool(self.data.eq_active[tile.equality_id])

    def attached_tiles(self) -> list[SkinTile]:
        """
        :return: The tiles still sitting on the flesh.
        """
        return [tile for tile in self.tiles if self.is_attached(tile)]

    def separations(self) -> SkinSeparations:
        """
        :return: How far every tile has moved from its seat in the current state.
        """
        equality_rows = np.flatnonzero(
            self.data.efc_type[: self.data.nefc]
            == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        )
        constrained_ids, first_indices = np.unique(
            self.data.efc_id[equality_rows], return_index=True
        )
        first_row_of_equality = np.full(self.model.neq, -1)
        first_row_of_equality[constrained_ids] = equality_rows[first_indices]
        first_rows = first_row_of_equality[self.equality_ids]
        constrained = first_rows >= 0
        # The first three rows hold the tile's anchor minus the flesh's anchor, in world axes.
        gaps = np.zeros((len(self.tiles), 3))
        gaps[constrained] = self.data.efc_pos[
            first_rows[constrained, None] + np.arange(3)
        ]
        flesh_rotations = self.data.xmat[self.flesh_body_ids].reshape(-1, 3, 3)
        normals = np.einsum("nij,nj->ni", flesh_rotations, self.outward_normals)
        outward = np.einsum("ni,ni->n", gaps, normals)
        along_surface = np.linalg.norm(gaps - outward[:, None] * normals, axis=1)
        return SkinSeparations(outward=outward, along_surface=along_surface)

    def release_pulled_tiles(self) -> list[SkinTile]:
        """
        Release every attached tile that has been pulled or dragged further than its
        limits for :attr:`release_steps` steps in a row.

        :return: The tiles released by this call.
        """
        separations = self.separations()
        pulled = (separations.outward > self.release_distance) | (
            separations.along_surface > self.release_shear
        )
        self.steps_pulled = np.where(pulled, self.steps_pulled + 1, 0)
        pulled_tiles = [
            self.tiles[index]
            for index in np.flatnonzero(self.steps_pulled >= self.release_steps)
        ]
        for tile in pulled_tiles:
            self.release_tile(tile)
        return pulled_tiles

    def release_tile(self, tile: SkinTile) -> None:
        """
        Detach a tile from the flesh; from then on it collides with the flesh.

        :param tile: A tile of this fruit.
        """
        self.data.eq_active[tile.equality_id] = 0
        self.model.geom_conaffinity[tile.geom_id] = (
            self.description.released_skin_affinity
        )

    def cut(self) -> None:
        """
        Let the two flesh halves come apart, as after cutting around the pit.
        """
        for equality in self.description.cut_equalities:
            self.data.eq_active[self.id_of(mujoco.mjtObj.mjOBJ_EQUALITY, equality)] = 0

    def slice(self, side: FruitSide, slice_index: int | None = None) -> None:
        """
        Cut through the flesh and the skin of one half between two of its slices.

        :param side: The half to slice.
        :param slice_index: The slice, counted from the stem, after which to cut; every
            cut of the half if None.
        """
        for link in self.description.slice_links:
            if link.side == side and slice_index in (None, link.slice_index):
                self.data.eq_active[
                    self.id_of(mujoco.mjtObj.mjOBJ_EQUALITY, link.equality)
                ] = 0

    def remove_pit(self) -> None:
        """
        Free the pit from its seat in the flesh; it collides with the flesh throughout.
        """
        self.data.eq_active[
            self.id_of(mujoco.mjtObj.mjOBJ_EQUALITY, self.description.pit_equality)
        ] = 0
        pit_body_id = self.id_of(mujoco.mjtObj.mjOBJ_BODY, self.description.pit_body)
        for geom_id in np.flatnonzero(self.model.geom_bodyid == pit_body_id):
            self.model.geom_conaffinity[geom_id] = (
                self.description.released_pit_affinity
            )

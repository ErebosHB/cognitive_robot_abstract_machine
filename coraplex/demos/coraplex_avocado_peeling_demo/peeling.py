"""
Peeling, cutting and pitting a fruit whose parts are free bodies held by equality
constraints, as the produce models of ``private_meshes`` describe them in
``mujoco_peel.json`` (format 2).

Every skin tile is welded to the flesh chunk under it, neighbouring flesh chunks to
each other and the pit into its seat. MuJoCo has no breakable constraints, so
:class:`PeelableFruit` switches them:

- A tile comes off when the weld holding it stays loaded past the skin's hold on the
  flesh: pulled outward at an edge, or dragged along the flesh.
- Neighbouring tiles are linked by hinges that switch on once one of the two has come
  off, so a peeled tile pulls its neighbours along. A link tears when it is loaded past
  its strength; links along a column, stem to base, are stronger than links across, so
  a peel runs from stem to base in a strip.
- A knife plane releases every flesh weld and skin link between parts on opposite
  sides of it, and each separate piece then collides with the others.

The criteria are forces: in equilibrium they are what a robot applies, whatever the
masses of the parts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np
from exceptions import FruitPartNotFound, UnsupportedFruitModel
from typing_extensions import Any

# %% the peeling description


class FruitSide(StrEnum):
    """
    The two halves of a fruit, on either side of the plane x = 0 of its own frame.
    """

    LEFT = "left"
    RIGHT = "right"


class LinkKind(StrEnum):
    """
    Which way a link between two skin tiles runs.
    """

    ALONG = "along"
    """
    Between two tiles of one column, stem to base.
    """

    ACROSS = "across"
    """
    Between two tiles of neighbouring columns.
    """


@dataclass
class SkinTileDescription:
    """
    One skin tile and the weld that holds it to its flesh chunk.
    """

    body: str
    """
    The name of the tile's body.
    """

    geom: str
    """
    The name of the tile's own collision geom, used once it has come off.
    """

    shell: str
    """
    The name of the copy of the tile's collision shape on its flesh chunk, used while
    the tile is on.
    """

    equality: str
    """
    The name of the weld holding the tile to its flesh chunk.
    """

    flesh: str
    """
    The name of the flesh chunk the tile is welded to.
    """

    row: int
    """
    The tile's row, counted from the stem.
    """

    column: int
    """
    The tile's column around the fruit.
    """

    position: np.ndarray
    """
    The tile's centre of mass in the fruit's frame, in metres.
    """

    probes: np.ndarray
    """
    The midpoints of the tile's stem, base and two side edges in the fruit's frame, one
    row each.
    """

    normal: np.ndarray
    """
    The direction pointing out of the fruit at the tile, in the fruit's frame.
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
            shell=data["shell"],
            equality=data["equality"],
            flesh=data["flesh"],
            row=data["row"],
            column=data["column"],
            position=np.array(data["position"]),
            probes=np.array(data["probes"]),
            normal=np.array(data["normal"]),
        )


@dataclass
class FleshChunkDescription:
    """
    One chunk of the flesh: a wedge of one slab across the fruit.
    """

    body: str
    """
    The name of the chunk's body.
    """

    slab: int
    """
    The slab across the fruit the chunk belongs to, counted from the stem.
    """

    sector: int
    """
    The wedge around the fruit's axis the chunk belongs to.
    """

    side: FruitSide
    """
    The half of the fruit the chunk lies in.
    """

    position: np.ndarray
    """
    The chunk's centre of mass in the fruit's frame, in metres.
    """

    geoms: list[str]
    """
    The names of the chunk's collision geoms.
    """

    mass: float
    """
    The chunk's mass in kilograms.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FleshChunkDescription:
        """
        :param data: One entry of the ``flesh`` list of a peeling description.
        :return: The chunk it describes.
        """
        return cls(
            body=data["body"],
            slab=data["slab"],
            sector=data["sector"],
            side=FruitSide(data["side"]),
            position=np.array(data["position"]),
            geoms=data["geoms"],
            mass=data["massKg"],
        )


@dataclass
class FleshWeldDescription:
    """
    A weld between two neighbouring flesh chunks.
    """

    equality: str
    """
    The name of the weld.
    """

    bodies: tuple[str, str]
    """
    The names of the two chunks it holds together.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FleshWeldDescription:
        """
        :param data: One entry of the ``fleshWelds`` list of a peeling description.
        :return: The weld it describes.
        """
        first, second = data["bodies"]
        return cls(equality=data["equality"], bodies=(first, second))


@dataclass
class SkinLinkDescription:
    """
    A hinge along the edge two neighbouring skin tiles share: a point connection at each
    end of the edge.
    """

    equalities: tuple[str, str]
    """
    The names of the two point connections.
    """

    tiles: tuple[str, str]
    """
    The names of the two tiles' bodies.
    """

    kind: LinkKind
    """
    Which way the link runs.
    """

    toughness: float
    """
    How much stronger than the plain skin the link is.
    """

    ends: np.ndarray
    """
    The two ends of the shared edge in the fruit's frame, one row each.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SkinLinkDescription:
        """
        :param data: One entry of the ``links`` list of a peeling description.
        :return: The link it describes.
        """
        first_equality, second_equality = data["equalities"]
        first_tile, second_tile = data["tiles"]
        return cls(
            equalities=(first_equality, second_equality),
            tiles=(first_tile, second_tile),
            kind=LinkKind(data["kind"]),
            toughness=data["toughness"],
            ends=np.array(data["ends"]),
        )


@dataclass
class PitDescription:
    """
    The pit and the weld holding it in its seat.
    """

    equality: str
    """
    The name of the weld holding the pit in its seat.
    """

    body: str
    """
    The name of the pit's body.
    """

    flesh: str
    """
    The name of the flesh chunk the pit is welded to.
    """

    side: FruitSide
    """
    The half of the fruit holding most of the pit.
    """

    position: np.ndarray
    """
    The pit's centre of mass in the fruit's frame, in metres.
    """

    geoms: list[str]
    """
    The names of the pit's collision geoms.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PitDescription:
        """
        :param data: The ``pit`` entry of a peeling description.
        :return: The pit it describes.
        """
        return cls(
            equality=data["equality"],
            body=data["body"],
            flesh=data["flesh"],
            side=FruitSide(data["side"]),
            position=np.array(data["position"]),
            geoms=data["geoms"],
        )


@dataclass
class ContactBits:
    """
    The contact type bits the fruit's parts use.
    """

    world: int
    """
    The bit of the world and of robots, MuJoCo's default.
    """

    skin: int
    """
    The bit of the skin.
    """

    pit: int
    """
    The bit of the pit.
    """

    flesh: list[int]
    """
    The bits handed out to the separate pieces of flesh.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ContactBits:
        """
        :param data: The ``bits`` entry of a peeling description.
        :return: The bits it lists.
        """
        return cls(
            world=data["world"], skin=data["skin"], pit=data["pit"], flesh=data["flesh"]
        )


@dataclass
class PeelingDescription:
    """
    The parts and constraints of a peelable fruit model.
    """

    tiles: list[SkinTileDescription]
    """
    The skin tiles.
    """

    flesh: list[FleshChunkDescription]
    """
    The flesh chunks.
    """

    flesh_welds: list[FleshWeldDescription]
    """
    The welds between neighbouring flesh chunks.
    """

    links: list[SkinLinkDescription]
    """
    The links between neighbouring skin tiles.
    """

    pit: PitDescription
    """
    The pit.
    """

    bits: ContactBits
    """
    The contact bits of the fruit's parts.
    """

    supported_format: int = 2
    """
    The version of ``mujoco_peel.json`` this description reads.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any], source: str) -> PeelingDescription:
        """
        :param data: The parsed content of a ``mujoco_peel.json``.
        :param source: Where it was read from, for the error message.
        :return: The description it holds.
        :raises UnsupportedFruitModel: If it is not of :attr:`supported_format`.
        """
        if data.get("format") != cls.supported_format:
            raise UnsupportedFruitModel(source, data.get("format"))
        return cls(
            tiles=[SkinTileDescription.from_json(tile) for tile in data["tiles"]],
            flesh=[FleshChunkDescription.from_json(chunk) for chunk in data["flesh"]],
            flesh_welds=[
                FleshWeldDescription.from_json(weld) for weld in data["fleshWelds"]
            ],
            links=[SkinLinkDescription.from_json(link) for link in data["links"]],
            pit=PitDescription.from_json(data["pit"]),
            bits=ContactBits.from_json(data["bits"]),
        )

    @classmethod
    def from_file(cls, path: Path) -> PeelingDescription:
        """
        :param path: A ``mujoco_peel.json`` file.
        :return: The description it holds.
        """
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")), str(path))


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

    index: int
    """
    The tile's position in :attr:`PeelableFruit.tiles`.
    """

    body_id: int
    """
    The id of the tile's body.
    """

    geom_id: int
    """
    The id of the tile's own collision geom.
    """

    shell_id: int
    """
    The id of the copy of the tile's collision shape on its flesh chunk.
    """

    flesh_body_id: int
    """
    The id of the flesh chunk the tile is welded to.
    """


@dataclass
class PeelableFruit:
    """
    A fruit in a running MuJoCo simulation that peels where it is pulled and can be cut.

    Call :meth:`release_pulled_tiles` after every simulation step. The fruit changes the
    model as it comes apart: contact bits, and armature on peeled tiles. After
    ``mujoco.mj_resetData``, call :meth:`reset`.
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

    peel_stress: float = 5000.0
    """
    The load in pascals, times a tile's area, that pulling a tile outward at an edge
    takes to peel it: about 0.2 N in the middle of an avocado.
    """

    shear_stress: float = 1e6
    """
    The load in pascals, times a tile's area, that dragging a tile along the flesh takes
    to tear it off.
    """

    tear_strength: float = 10.0
    """
    The load in newtons per metre of edge, times its toughness, that tears a link.
    """

    release_steps: int = 5
    """
    How many simulation steps in a row a tile or a link must stay past its limit before
    it gives, so a momentary jolt passed through the constraints breaks nothing.
    """

    hanging_armature: tuple[float, float] = (2e-3, 2e-8)
    """
    The armature in kilograms and kilogram square metres a peeled tile still hanging on
    links gets on its free joint: inertia without weight, without which a 0.05 g tile is
    thrown far within one step whenever a neighbouring link tears.
    """

    tiles: list[SkinTile] = field(init=False)
    """
    The skin tiles, resolved to ids of :attr:`model`.
    """

    tile_index: dict[str, int] = field(init=False)
    """
    The position in :attr:`tiles` of each tile, by its body's name.
    """

    flesh_body_ids: np.ndarray = field(init=False)
    """
    The body ids of the flesh chunks, in the order of the description.
    """

    flesh_index: dict[str, int] = field(init=False)
    """
    The position of each flesh chunk in the description, by its body's name.
    """

    pit_body_id: int = field(init=False)
    """
    The id of the pit's body.
    """

    released: np.ndarray = field(init=False)
    """
    For each tile, whether it has come off.
    """

    steps_pulled: np.ndarray = field(init=False)
    """
    For how many steps in a row each tile has been loaded past its limit.
    """

    link_on: np.ndarray = field(init=False)
    """
    For each link, whether it is switched on: one of its tiles is off, and it is neither
    torn nor cut.
    """

    link_was_on: np.ndarray = field(init=False)
    """
    For each link, whether it has ever been switched on.
    """

    link_torn: np.ndarray = field(init=False)
    """
    For each link, whether it has torn.
    """

    link_cut: np.ndarray = field(init=False)
    """
    For each link, whether a knife has cut it.
    """

    steps_stretched: np.ndarray = field(init=False)
    """
    For how many steps in a row each link has been loaded past its limit.
    """

    weld_cut: np.ndarray = field(init=False)
    """
    For each flesh weld, whether a knife has cut it.
    """

    arrays: dict[str, np.ndarray] = field(init=False)
    """
    Fixed per-part arrays derived from the description and the model, by name.
    """

    def __post_init__(self) -> None:
        description = self.description
        body, geom, equality = (
            mujoco.mjtObj.mjOBJ_BODY,
            mujoco.mjtObj.mjOBJ_GEOM,
            mujoco.mjtObj.mjOBJ_EQUALITY,
        )
        self.flesh_index = {chunk.body: i for i, chunk in enumerate(description.flesh)}
        self.flesh_body_ids = self.ids_of(body, [c.body for c in description.flesh])
        self.tile_index = {tile.body: i for i, tile in enumerate(description.tiles)}
        self.tiles = [
            SkinTile(
                description=tile,
                index=i,
                body_id=self.id_of(body, tile.body),
                geom_id=self.id_of(geom, tile.geom),
                shell_id=self.id_of(geom, tile.shell),
                flesh_body_id=self.id_of(body, tile.flesh),
            )
            for i, tile in enumerate(description.tiles)
        ]
        self.pit_body_id = self.id_of(body, description.pit.body)
        tiles = description.tiles
        tile_bodies = np.array([tile.body_id for tile in self.tiles])
        probes = np.array([tile.probes for tile in tiles])
        positions = np.array([tile.position for tile in tiles])
        normals = np.array([tile.normal for tile in tiles])
        # From each tile's centre to the midpoints of its edges, along the skin.
        reach = probes - positions[:, None]
        reach -= np.einsum("npi,ni->np", reach, normals)[..., None] * normals[:, None]
        edge_lever = np.maximum(np.linalg.norm(reach, axis=-1), 1e-4)
        ends = np.array([link.ends for link in description.links])
        shell_owner = np.full(self.model.ngeom, -1)
        shell_owner[[tile.shell_id for tile in self.tiles]] = np.arange(len(tiles))
        flesh_geoms = [self.ids_of(geom, chunk.geoms) for chunk in description.flesh]
        self.arrays = {
            "tile_equality": self.ids_of(equality, [t.equality for t in tiles]),
            "tile_flesh": np.array([self.flesh_index[t.flesh] for t in tiles]),
            "tile_geom": np.array([tile.geom_id for tile in self.tiles]),
            "tile_shell": np.array([tile.shell_id for tile in self.tiles]),
            "tile_dof": self.model.body_dofadr[tile_bodies][:, None] + np.arange(6),
            "tile_area": np.linalg.norm(probes[:, 1] - probes[:, 0], axis=1)
            * np.linalg.norm(probes[:, 3] - probes[:, 2], axis=1),
            "tile_normal": normals,
            "edge_lever": edge_lever,
            "edge_out": np.cross(reach / edge_lever[..., None], normals[:, None]),
            "shell_owner": shell_owner,
            "link_equality": self.ids_of(
                equality, [n for link in description.links for n in link.equalities]
            ).reshape(-1, 2),
            "link_a": np.array(
                [self.tile_index[k.tiles[0]] for k in description.links]
            ),
            "link_b": np.array(
                [self.tile_index[k.tiles[1]] for k in description.links]
            ),
            "link_limit": self.tear_strength
            * np.maximum(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1), 1e-4)
            * np.array([link.toughness for link in description.links]),
            "weld_equality": self.ids_of(
                equality, [w.equality for w in description.flesh_welds]
            ),
            "weld_a": np.array(
                [self.flesh_index[w.bodies[0]] for w in description.flesh_welds]
            ),
            "weld_b": np.array(
                [self.flesh_index[w.bodies[1]] for w in description.flesh_welds]
            ),
            "flesh_geoms_flat": np.concatenate(flesh_geoms),
            "flesh_geom_chunk": np.concatenate(
                [np.full(len(g), i) for i, g in enumerate(flesh_geoms)]
            ),
            "pit_geoms": self.ids_of(geom, description.pit.geoms),
            "pit_equality": np.array([self.id_of(equality, description.pit.equality)]),
        }
        managed = np.concatenate(
            [
                self.arrays["flesh_geoms_flat"],
                self.arrays["tile_shell"],
                self.arrays["tile_geom"],
                self.arrays["pit_geoms"],
            ]
        )
        self.arrays["managed_geoms"] = managed
        self.arrays["managed_bodies"] = np.unique(self.model.geom_bodyid[managed])
        self.reset()

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

    def ids_of(self, object_type: mujoco.mjtObj, names: list[str]) -> np.ndarray:
        """
        :param object_type: The kind of MuJoCo object.
        :param names: The objects' names without the fruit's prefix.
        :return: The objects' ids in :attr:`model`.
        """
        return np.array([self.id_of(object_type, name) for name in names], dtype=int)

    def reset(self) -> None:
        """
        Put the fruit back together: every tile on, no link torn, nothing cut. Call it
        after ``mujoco.mj_resetData``, which puts the bodies and constraints back.
        """
        tile_count, link_count = len(self.tiles), len(self.description.links)
        self.released = np.zeros(tile_count, dtype=bool)
        self.steps_pulled = np.zeros(tile_count, dtype=int)
        self.link_on = np.zeros(link_count, dtype=bool)
        self.link_was_on = np.zeros(link_count, dtype=bool)
        self.link_torn = np.zeros(link_count, dtype=bool)
        self.link_cut = np.zeros(link_count, dtype=bool)
        self.steps_stretched = np.zeros(link_count, dtype=int)
        self.weld_cut = np.zeros(len(self.description.flesh_welds), dtype=bool)
        self.data.eq_active[self.arrays["link_equality"].ravel()] = 0
        self.switch_links()
        self.update_collisions()

    # %% state

    def is_attached(self, tile: SkinTile) -> bool:
        """
        :param tile: A tile of this fruit.
        :return: Whether the tile is still on the flesh.
        """
        return not self.released[tile.index]

    def peeled_tiles(self) -> list[SkinTile]:
        """
        :return: The tiles that have come off.
        """
        return [tile for tile in self.tiles if self.released[tile.index]]

    def peeled_with(self, tile: SkinTile) -> list[SkinTile]:
        """
        :param tile: A tile of this fruit.
        :return: The peeled tiles that came off together with *tile*: joined to it through
            peeled tiles by links, including links that have torn since.
        """
        return self.joined_to(tile, self.link_was_on)

    def carried_with(self, tile: SkinTile) -> list[SkinTile]:
        """
        :param tile: A tile of this fruit.
        :return: The peeled tiles still hanging on *tile*: joined to it through peeled
            tiles by links that hold now.
        """
        return self.joined_to(tile, self.link_on)

    def joined_to(self, tile: SkinTile, links: np.ndarray) -> list[SkinTile]:
        """
        :param tile: A tile of this fruit.
        :param links: For each link, whether to follow it.
        :return: The peeled tiles joined to *tile* through peeled tiles by those links,
            *tile* itself included; none if it is still on.
        """
        if not self.released[tile.index]:
            return []
        link_a, link_b = self.arrays["link_a"], self.arrays["link_b"]
        both = links & self.released[link_a] & self.released[link_b]
        neighbours: dict[int, list[int]] = {}
        for a, b in zip(link_a[both], link_b[both]):
            neighbours.setdefault(int(a), []).append(int(b))
            neighbours.setdefault(int(b), []).append(int(a))
        reached, frontier = {tile.index}, [tile.index]
        while frontier:
            for other in neighbours.get(frontier.pop(), []):
                if other not in reached:
                    reached.add(other)
                    frontier.append(other)
        return [self.tiles[index] for index in sorted(reached)]

    def piece_labels(self) -> np.ndarray:
        """
        :return: For each flesh chunk, a label shared by the chunks of the same piece of
            flesh: connected over the flesh welds not cut.
        """
        parent = np.arange(len(self.description.flesh))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        welds = ~self.weld_cut
        for a, b in zip(self.arrays["weld_a"][welds], self.arrays["weld_b"][welds]):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
        return np.array([find(index) for index in range(len(parent))])

    def pieces(self) -> list[list[str]]:
        """
        :return: The names of the flesh chunks of each separate piece of flesh, largest
            piece first.
        """
        labels = self.piece_labels()
        names = [chunk.body for chunk in self.description.flesh]
        groups = [
            [names[i] for i in np.flatnonzero(labels == label)]
            for label in np.unique(labels)
        ]
        return sorted(groups, key=len, reverse=True)

    # %% loads

    def constraint_forces(self, equalities: np.ndarray, rows: int) -> np.ndarray:
        """
        :param equalities: Ids of equality constraints, of any shape.
        :param rows: How many of each constraint's first rows to read.
        :return: Those rows of each constraint's force, zero where it is off; for a weld
            or a connection, the force on its first body in world axes.
        """
        data = self.data
        equality_rows = np.flatnonzero(
            data.efc_type[: data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        )
        ids, first = np.unique(data.efc_id[equality_rows], return_index=True)
        first_row = np.full(self.model.neq, -1)
        first_row[ids] = equality_rows[first]
        row = first_row[equalities]
        on = row >= 0
        forces = np.zeros((*np.shape(equalities), rows))
        forces[on] = data.efc_force[row[on][:, None] + np.arange(rows)]
        return forces

    def tile_loads(self) -> tuple[np.ndarray, np.ndarray]:
        """
        :return: For each tile, the load in newtons pulling it outward at its most loaded
            edge, and the load dragging it along the flesh; zero for peeled tiles.
        """
        wrench = self.constraint_forces(self.arrays["tile_equality"], 6)
        # MuJoCo's rotational weld rows carry twice the torque.
        force, torque = wrench[:, :3], 0.5 * wrench[:, 3:]
        flesh = self.flesh_body_ids[self.arrays["tile_flesh"]]
        rotation = self.data.xmat[flesh].reshape(-1, 3, 3)
        normal = np.einsum("nij,nj->ni", rotation, self.arrays["tile_normal"])
        # The weld holds the tile in place: a pull outward shows as an inward force, and
        # a moment lifting an edge as a torque turning it back.
        pull = -np.einsum("ni,ni->n", force, normal)
        lever = np.einsum("nij,npj->npi", rotation, self.arrays["edge_out"])
        tip = -np.einsum("ni,npi->np", torque, lever) / self.arrays["edge_lever"]
        edge = (pull[:, None] + tip).max(axis=1)
        drag = force + pull[:, None] * normal + self.shell_friction()
        shear = np.linalg.norm(drag, axis=1)
        edge[self.released] = 0.0
        shear[self.released] = 0.0
        return edge, shear

    def shell_friction(self) -> np.ndarray:
        """
        :return: For each tile, the friction in newtons of the contacts on its shell, in
            world axes.
        """
        data = self.data
        friction = np.zeros((len(self.tiles), 3))
        if not data.ncon:
            return friction
        owner = self.arrays["shell_owner"]
        first, second = data.contact.geom1[: data.ncon], data.contact.geom2[: data.ncon]
        local = np.zeros(6)
        for contact in np.flatnonzero((owner[first] >= 0) | (owner[second] >= 0)):
            mujoco.mj_contactForce(self.model, data, contact, local)
            frame = data.contact.frame[contact].reshape(3, 3)
            tangential = local[1] * frame[1] + local[2] * frame[2]
            for geom, sign in ((first[contact], -1.0), (second[contact], 1.0)):
                if owner[geom] >= 0:
                    friction[owner[geom]] += sign * tangential
        return friction

    def link_tension(self) -> np.ndarray:
        """
        :return: For each link, the load in newtons trying to tear it: twice the larger of
            the forces at the two ends of its edge; zero where it is off.
        """
        force = np.linalg.norm(
            self.constraint_forces(self.arrays["link_equality"], 3), axis=-1
        )
        tension = 2 * force.max(axis=1)
        tension[~self.link_on] = 0.0
        return tension

    # %% coming apart

    def release_pulled_tiles(self) -> list[SkinTile]:
        """
        Peel every tile loaded past its hold on the flesh and tear every link loaded past
        its strength, each for :attr:`release_steps` steps in a row.

        :return: The tiles that came off.
        """
        edge, shear = self.tile_loads()
        area = self.arrays["tile_area"]
        pulled = ~self.released & (
            (edge > self.peel_stress * area) | (shear > self.shear_stress * area)
        )
        self.steps_pulled = np.where(pulled, self.steps_pulled + 1, 0)
        released = np.flatnonzero(self.steps_pulled >= self.release_steps)
        stretched = self.link_on & (self.link_tension() > self.arrays["link_limit"])
        self.steps_stretched = np.where(stretched, self.steps_stretched + 1, 0)
        torn = self.steps_stretched >= self.release_steps
        if torn.any():
            self.link_torn |= torn
            self.steps_stretched[torn] = 0
        if len(released):
            self.release(released)
        elif torn.any():
            self.switch_links()
        return [self.tiles[index] for index in released]

    def release(self, indices: np.ndarray) -> None:
        """
        Take tiles off the flesh.

        :param indices: The tiles' positions in :attr:`tiles`.
        """
        self.released[indices] = True
        self.steps_pulled[indices] = 0
        self.data.eq_active[self.arrays["tile_equality"][indices]] = 0
        self.switch_links()
        self.update_collisions()

    def switch_links(self) -> None:
        """
        Switch on the links next to peeled tiles that are neither torn nor cut, and give
        peeled tiles hanging on links their armature.
        """
        link_a, link_b = self.arrays["link_a"], self.arrays["link_b"]
        wanted = ~(self.link_torn | self.link_cut) & (
            self.released[link_a] | self.released[link_b]
        )
        changed = wanted != self.link_on
        self.data.eq_active[self.arrays["link_equality"][changed].ravel()] = np.repeat(
            wanted[changed], 2
        )
        self.link_on = wanted
        self.link_was_on |= wanted
        hanging = np.zeros(len(self.tiles), dtype=bool)
        hanging[link_a[wanted]] = True
        hanging[link_b[wanted]] = True
        hanging &= self.released
        dofs = self.arrays["tile_dof"]
        linear, angular = self.hanging_armature
        self.model.dof_armature[dofs[:, :3]] = np.where(hanging, linear, 0.0)[:, None]
        self.model.dof_armature[dofs[:, 3:]] = np.where(hanging, angular, 0.0)[:, None]

    def cut(self, point: np.ndarray, normal: np.ndarray) -> int:
        """
        Cut every piece of flesh with a knife plane: release every flesh weld and skin
        link between chunks on opposite sides of it.

        :param point: A point of the plane, in world coordinates.
        :param normal: The plane's normal, in world axes.
        :return: How many flesh welds were cut.
        """
        side = (self.data.xipos[self.flesh_body_ids] - point) @ normal > 0
        weld_a, weld_b = self.arrays["weld_a"], self.arrays["weld_b"]
        welds = ~self.weld_cut & (side[weld_a] != side[weld_b])
        self.weld_cut |= welds
        self.data.eq_active[self.arrays["weld_equality"][welds]] = 0
        tile_flesh = self.arrays["tile_flesh"]
        chunk_a = tile_flesh[self.arrays["link_a"]]
        chunk_b = tile_flesh[self.arrays["link_b"]]
        self.link_cut |= side[chunk_a] != side[chunk_b]
        self.switch_links()
        self.update_collisions()
        return int(welds.sum())

    def cut_in_fruit_frame(
        self, point: np.ndarray, normal: np.ndarray, reference: str | None = None
    ) -> int:
        """
        Cut with a knife plane given in the fruit's own frame (metres, z along the long
        axis from the base), as a flesh chunk carries that frame.

        :param point: A point of the plane, in the fruit's frame.
        :param normal: The plane's normal, in the fruit's frame.
        :param reference: The name of the flesh chunk whose pose places the plane, one of
            the piece to be cut; the first chunk if None.
        :return: How many flesh welds were cut.
        """
        index = 0 if reference is None else self.flesh_index[reference]
        reference_chunk = self.description.flesh[index]
        body_id = self.flesh_body_ids[index]
        rotation = self.data.xmat[body_id].reshape(3, 3)
        world_point = self.data.xpos[body_id] + rotation @ (
            np.asarray(point, float) - reference_chunk.position
        )
        return self.cut(world_point, rotation @ np.asarray(normal, float))

    def halve(self, angle: float = 0.0, reference: str | None = None) -> int:
        """
        Cut the fruit lengthwise through its axis.

        :param angle: How far in radians the knife is turned about the axis from the
            plane x = 0 of the fruit; the cut lands on the nearest wedge boundary.
        :param reference: The name of a flesh chunk of the piece to be cut, whose pose
            places the knife; the first chunk if None.
        :return: How many flesh welds were cut.
        """
        return self.cut_in_fruit_frame(
            np.zeros(3), np.array([np.cos(angle), np.sin(angle), 0.0]), reference
        )

    def slice(self, height: float, reference: str | None = None) -> int:
        """
        Cut the fruit across its long axis.

        :param height: Where, in metres above the base; the cut lands on the nearest slab
            boundary.
        :param reference: The name of a flesh chunk of the piece to be cut, whose pose
            places the knife; the first chunk if None.
        :return: How many flesh welds were cut.
        """
        return self.cut_in_fruit_frame(
            np.array([0.0, 0.0, height]), np.array([0.0, 0.0, 1.0]), reference
        )

    def remove_pit(self) -> None:
        """
        Free the pit from its seat in the flesh; it collides with the flesh throughout.
        """
        self.data.eq_active[self.arrays["pit_equality"]] = 0

    # %% collisions

    def update_collisions(self) -> None:
        """
        Give every piece of flesh its own contact bit, so pieces collide with each other
        but not with themselves. The skin collides with the world and with other pieces:
        through the shells on the flesh while on, through the tiles themselves once
        peeled. MuJoCo filters contacts by the bits of whole bodies first, so the body
        bits are updated too.
        """
        model, bits = self.model, self.description.bits
        labels = self.piece_labels()
        order = {label: i for i, label in enumerate(dict.fromkeys(labels.tolist()))}
        piece_bits = np.array(bits.flesh, dtype=np.int64)
        chunk_bit = piece_bits[[order[label] % len(piece_bits) for label in labels]]
        every_piece = int(piece_bits.sum())
        geoms = self.arrays["flesh_geoms_flat"]
        geom_bit = chunk_bit[self.arrays["flesh_geom_chunk"]]
        model.geom_contype[geoms] = geom_bit
        model.geom_conaffinity[geoms] = (
            bits.world | bits.pit | (every_piece & ~geom_bit)
        )
        own = chunk_bit[self.arrays["tile_flesh"]]
        on = ~self.released
        shells, tile_geoms = self.arrays["tile_shell"], self.arrays["tile_geom"]
        model.geom_contype[shells] = np.where(on, bits.skin, 0)
        model.geom_conaffinity[shells] = np.where(
            on, bits.world | (every_piece & ~own), 0
        )
        model.geom_contype[tile_geoms] = np.where(on, 0, bits.skin)
        model.geom_conaffinity[tile_geoms] = np.where(on, 0, bits.world | every_piece)
        managed, bodies = self.arrays["managed_geoms"], self.arrays["managed_bodies"]
        model.body_contype[bodies] = 0
        model.body_conaffinity[bodies] = 0
        np.bitwise_or.at(
            model.body_contype, model.geom_bodyid[managed], model.geom_contype[managed]
        )
        np.bitwise_or.at(
            model.body_conaffinity,
            model.geom_bodyid[managed],
            model.geom_conaffinity[managed],
        )

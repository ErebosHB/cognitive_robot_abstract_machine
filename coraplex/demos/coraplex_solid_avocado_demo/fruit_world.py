"""
The pieces of a cut fruit as bodies of a :class:`semantic_digital_twin.world.World`, kept
in step with the MuJoCo scene they are simulated in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Avocado,
    AvocadoFlesh,
    AvocadoPiece,
    AvocadoPit,
    AvocadoSkin,
)
from semantic_digital_twin.world import Body, World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

from cutting import CuttableFruit, FruitMaterial, FruitPart, FruitPiece

# %% the world of the pieces


@dataclass
class FruitWorld:
    """
    A world holding a body for each piece of an avocado, free under the world's root,
    weighing what the piece weighs and colliding as it does in MuJoCo, and below it a
    weightless body looking like each of its parts. The whole avocado is annotated as an
    :class:`Avocado`, a piece cut from it as an :class:`AvocadoPiece`, each with its parts.
    """

    world: World = field(
        init=False, default_factory=lambda: World(mesh_decomposer=None)
    )
    """
    The world of the pieces. Its collision detector is paused, as MuJoCo checks the
    pieces' collisions; to check collisions in it, resume the detector's
    ``world_model_updater`` and ``world_state_updater`` and call ``on_model_change``.
    """

    root: Body = field(init=False)
    """
    The root of :attr:`world`, the frame of the MuJoCo scene.
    """

    connections: dict[FruitPiece, Connection6DoF] = field(
        init=False, default_factory=dict
    )
    """
    The connection of each piece's body to :attr:`root`.
    """

    def __post_init__(self) -> None:
        """
        Pause the world's collision detector and give the world its root.
        """
        # Left running, the detector would build the collision shapes of every piece
        # anew at each cut and move them at each pose update.
        detector = self.world.collision_manager.collision_detector
        detector.world_model_updater.pause()
        detector.world_state_updater.pause()
        self.root = Body(name=PrefixedName("world"))
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(self.root)

    def body_of(self, piece: FruitPiece) -> Body:
        """
        :param piece: A piece of the fruit as of the last :meth:`update`.
        :return: Its body.
        """
        return self.connections[piece].child

    def update(self, *, fruit: CuttableFruit) -> None:
        """
        Give the world a body for each piece the fruit is in now, and none for a piece
        it is no longer in, and put each body where its piece is.

        :param fruit: The fruit.
        """
        gone = [piece for piece in self.connections if piece not in fruit.pieces]
        new = [piece for piece in fruit.pieces if piece not in self.connections]
        if gone or new:
            with self.world.modify_world():
                for piece in gone:
                    self.remove_piece(piece=piece)
                for piece in new:
                    self.connections[piece] = self.add_piece(piece=piece, fruit=fruit)
        # The connections have no fixed offsets, so their degrees of freedom, position
        # and quaternion, are the pose of the piece's body in the root's frame.
        state = self.world.state
        for piece, connection in self.connections.items():
            pose = fruit.pose_of(piece)
            w, x, y, z = pose.quaternion
            for dof, value in zip(
                connection.passive_dofs, [*pose.position, x, y, z, w]
            ):
                state[dof.id].position = value
        self.world.notify_state_change()

    def remove_piece(self, *, piece: FruitPiece) -> None:
        """
        Remove the bodies of a piece and its parts, and their annotations, from the
        world.

        :param piece: A piece of the fruit as of the last :meth:`update`.
        """
        branch = self.world.get_kinematic_structure_entities_of_branch(
            self.connections.pop(piece).child
        )
        for annotation in list(self.world.semantic_annotations):
            if annotation.root in branch:
                self.world.remove_semantic_annotation(annotation)
        self.world.remove_branch_from_world(branch[0])

    def add_piece(self, *, piece: FruitPiece, fruit: CuttableFruit) -> Connection6DoF:
        """
        Add the bodies of a piece and its parts to the world, the piece's at the root,
        and annotate the piece as the whole avocado or as a piece cut from it.

        :param piece: A piece of the fruit.
        :param fruit: The fruit, its scene compiled with the piece's body in it.
        :return: The connection of the piece's body to :attr:`root`.
        """
        # A plain string, as the whole fruit's name may be an enum member of the demos,
        # which a process reading the world without their modules cannot rebuild.
        body = Body(name=PrefixedName(str(piece.body_name)))
        distribution = piece.mass_distribution()
        body.inertial = Inertial(
            mass=distribution.mass,
            center_of_mass=Point3.from_iterable(distribution.centre),
            inertia=InertiaTensor(data=distribution.inertia),
        )
        model = fruit.model
        body_id = fruit.body_id(piece)
        first_geom = model.body_geomadr[body_id]
        # MuJoCo collides every mesh through its convex hull.
        body.collision = ShapeCollection(
            [
                Mesh.from_trimesh(
                    fruit.mesh_in_body_frame(geom_id).convex_hull,
                    origin=HomogeneousTransformationMatrix(reference_frame=body),
                )
                for geom_id in range(
                    first_geom, first_geom + model.body_geomnum[body_id]
                )
                if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
            ],
            reference_frame=body,
        )
        self.world.add_kinematic_structure_entity(body)
        connection = Connection6DoF.create_with_dofs(
            world=self.world, parent=self.root, child=body
        )
        self.world.add_connection(connection)
        parts = {
            part.material: self.add_part(piece=piece, part=part, fruit=fruit, body=body)
            for part in piece.parts
        }
        annotation_type = (
            Avocado if piece.body_name == fruit.body_name else AvocadoPiece
        )
        self.world.add_semantic_annotation_recursively(
            annotation_type(
                name=body.name,
                root=body,
                skin=parts.get(FruitMaterial.SKIN),
                flesh=parts.get(FruitMaterial.FLESH),
                pit=parts.get(FruitMaterial.PIT),
            )
        )
        return connection

    def add_part(
        self, *, piece: FruitPiece, part: FruitPart, fruit: CuttableFruit, body: Body
    ) -> AvocadoSkin | AvocadoFlesh | AvocadoPit:
        """
        Add the body of a piece's part to the world, fixed to the piece's body.

        :param piece: The piece.
        :param part: One of its parts.
        :param fruit: The fruit, whose materials colour the part.
        :param body: The piece's body.
        :return: The annotation of the part's body as the skin, flesh or pit it is.
        """
        part_body = Body(name=PrefixedName(f"{piece.body_name}_{part.material}"))
        shape = Mesh.from_trimesh(
            FruitPart.triangle_mesh_of(part.solid),
            origin=HomogeneousTransformationMatrix(reference_frame=part_body),
        )
        material = fruit.spec.find_default(part.geom_class).geom.material
        shape.color = Color.from_rgba(*fruit.spec.material(material).rgba)
        part_body.visual = ShapeCollection([shape], reference_frame=part_body)
        part_body.inertial = Inertial.negligible()
        self.world.add_kinematic_structure_entity(part_body)
        self.world.add_connection(FixedConnection(parent=body, child=part_body))
        match part.material:
            case FruitMaterial.SKIN:
                annotation_type = AvocadoSkin
            case FruitMaterial.FLESH:
                annotation_type = AvocadoFlesh
            case FruitMaterial.PIT:
                annotation_type = AvocadoPit
        return annotation_type(name=part_body.name, root=part_body)

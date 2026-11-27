"""
The pieces of a cut fruit as bodies of a :class:`semantic_digital_twin.world.World`,
kept in step with the MuJoCo scene they are simulated in: where they are, how ripe and
firm they are, what sticks to what and where the flesh is bruised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import IsAdherent
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    WorstBruisePressure,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Avocado,
    AvocadoPiece,
    Flesh,
    Peel,
    Pit,
)
from semantic_digital_twin.world import Body, World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

from cutting import CuttableFruit, FruitPart, FruitPiece
from fruit_material import FruitMaterial

# %% the world of the pieces


@dataclass
class FruitWorld:
    """
    A world holding a body for each piece of an avocado, free under the world's root,
    weighing what the piece weighs and colliding as it does in MuJoCo, and below it a
    weightless body looking like each of its parts.

    The whole avocado is annotated as an
    :class:`Avocado`, a piece cut from it as an :class:`AvocadoPiece`, each with its parts.
    """

    world: World = field(
        init=False, default_factory=lambda: World(mesh_decomposer=None)
    )
    """
    The world of the pieces.

    Its collision detector is paused, as MuJoCo checks the pieces' collisions; to check
    collisions in it, resume the detector's ``world_model_updater`` and
    ``world_state_updater`` and call ``on_model_change``.
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

    annotations: dict[FruitPiece, Avocado | AvocadoPiece] = field(
        init=False, default_factory=dict
    )
    """
    The annotation of each piece.
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
        self.update_adhesions(fruit=fruit)
        self.update_bruises(fruit=fruit)
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
        removed = self.annotations.pop(piece)
        for annotation in self.annotations.values():
            if isinstance(annotation, IsAdherent) and removed in annotation.adheres_to:
                annotation.come_off(other=removed)
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
        ContactParameters.create_for_surface(
            sliding_friction=float(
                fruit.spec.find_default(fruit.collision_class).geom.friction[0]
            )
        ).apply_to([body])
        annotation_type = (
            Avocado if piece.body_name == fruit.body_name else AvocadoPiece
        )
        annotation = annotation_type(
            name=body.name,
            root=body,
            peel=parts.get(FruitMaterial.SKIN),
            flesh=parts.get(FruitMaterial.FLESH),
            pit=parts.get(FruitMaterial.PIT),
        )
        if fruit.texture is not None:
            fruit.texture.describe(annotation=annotation)
        self.world.add_semantic_annotation_recursively(annotation)
        self.annotations[piece] = annotation
        return connection

    def update_adhesions(self, *, fruit: CuttableFruit) -> None:
        """
        Let the annotation of each piece cut from the fruit stick to those of exactly
        the pieces the piece sticks to now.

        :param fruit: The fruit, its pieces those of the last :meth:`update`.
        """
        adherent = {
            piece: annotation
            for piece, annotation in self.annotations.items()
            if isinstance(annotation, IsAdherent)
        }
        named = {piece.body_name: piece for piece in adherent}
        sticking = {
            (named[adhesion.held], named[adhesion.holder])
            for adhesion in fruit.adhesions
        }
        changes = [
            (annotation, other, (piece, holder) in sticking)
            for piece, annotation in adherent.items()
            for holder, other in adherent.items()
            if ((piece, holder) in sticking) != (other in annotation.adheres_to)
        ]
        if not changes:
            return
        with self.world.modify_world():
            for annotation, other, sticks in changes:
                if sticks:
                    annotation.stick_to(other=other)
                else:
                    annotation.come_off(other=other)

    def update_bruises(self, *, fruit: CuttableFruit) -> None:
        """
        Let the annotation of each piece's flesh know the worst bruise that reaches into
        it, where that is worse than the one it knows.

        :param fruit: The fruit, its pieces those of the last :meth:`update`.
        """
        worse = []
        for piece, annotation in self.annotations.items():
            if annotation.flesh is None:
                continue
            known = annotation.flesh.worst_bruise_pressure
            flesh = piece.part_of(FruitMaterial.FLESH)
            pressures = [
                bruise.pressure
                for bruise in fruit.bruises
                if (known is None or bruise.pressure > known.value.upper)
                and bruise.affects(part=flesh)
            ]
            if pressures:
                worse.append((annotation.flesh, max(pressures)))
        if not worse:
            return
        with self.world.modify_world():
            for flesh, pressure in worse:
                flesh.set_quality(
                    quality=WorstBruisePressure.from_measurement(value=pressure)
                )

    def add_part(
        self, *, piece: FruitPiece, part: FruitPart, fruit: CuttableFruit, body: Body
    ) -> Peel | Flesh | Pit:
        """
        Add the body of a piece's part to the world, fixed to the piece's body.

        :param piece: The piece.
        :param part: One of its parts.
        :param fruit: The fruit, whose materials colour the part.
        :param body: The piece's body.
        :return: The annotation of the part's body as the peel, flesh or pit it is.
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
        annotation_type = {
            FruitMaterial.SKIN: Peel,
            FruitMaterial.FLESH: Flesh,
            FruitMaterial.PIT: Pit,
        }[part.material]
        annotation = annotation_type(
            name=part_body.name,
            root=part_body,
            mass_density=MassDensity.from_measurement(value=part.density),
        )
        if fruit.texture is not None:
            fruit.texture.describe(annotation=annotation)
        return annotation

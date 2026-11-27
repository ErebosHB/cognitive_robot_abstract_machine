"""
Tests of telling KnowRob what the semantic annotations of a world are.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, fields
from types import ModuleType

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from random_events.interval import singleton
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    BruisingPressure,
    CuttingResistance,
    Firmness,
    MaterialElasticity,
    PeelStrength,
    ReleaseTorque,
    Ripeness,
    RipenessStage,
    WorstBruisePressure,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Avocado,
    AvocadoPiece,
    Flesh,
    Pit,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Iterator

# %% a world told to KnowRob


@pytest.fixture(scope="module")
def bridge_module() -> ModuleType:
    """
    The bridge module; skips where KnowRob or ROS is not installed.
    """
    if importlib.util.find_spec("knowrob") is None:
        pytest.skip("KnowRob is not installed")
    pytest.importorskip("rclpy")
    return importlib.import_module("semantic_digital_twin.adapters.knowrob.bridge")


@pytest.fixture(scope="module")
def session(bridge_module: ModuleType) -> Any:
    """
    One KnowRob session for the tests of this module; every bridge forgets what it told
    it when it stops.
    """
    return bridge_module.KnowRobSession.start()


@dataclass(kw_only=True)
class AvocadoWorld:
    """
    A world holding an avocado with its pit, and a bridge telling KnowRob about it.
    """

    world: World
    """
    The world.
    """

    avocado: Avocado
    """
    The annotation of the avocado.
    """

    bridge: Any
    """
    The bridge from the world to KnowRob.
    """


@pytest.fixture
def avocado_world(bridge_module: ModuleType, session: Any) -> Iterator[AvocadoWorld]:
    """
    An avocado with its pit, told to KnowRob; the bridge is stopped afterwards.
    """
    world = World()
    root = Body(name=PrefixedName("map"))
    avocado_body = Body(name=PrefixedName("avocado"))
    pit_body = Body(name=PrefixedName("avocado_pit"))
    with world.modify_world():
        for body in (root, avocado_body, pit_body):
            world.add_kinematic_structure_entity(body)
        world.add_connection(FixedConnection(parent=root, child=avocado_body))
        world.add_connection(FixedConnection(parent=avocado_body, child=pit_body))
        avocado = Avocado(
            name=avocado_body.name,
            root=avocado_body,
            pit=Pit(name=pit_body.name, root=pit_body),
        )
        world.add_semantic_annotation_recursively(avocado)
    bridge = bridge_module.KnowRobBridge(_world=world, session=session)
    yield AvocadoWorld(world=world, avocado=avocado, bridge=bridge)
    bridge.stop()


def individual(bridge_module: ModuleType, annotation: Any) -> str:
    """
    :param annotation: A semantic annotation.
    :return: The IRI KnowRob knows it by.
    """
    return bridge_module.Namespace.INDIVIDUALS.iri(name=annotation.name.name)


# %% what KnowRob is told


def test_annotation_is_of_its_class_and_each_class_above(
    bridge_module, session, avocado_world
):
    """
    KnowRob knows the avocado as an avocado and as the food it is, and that the class of
    avocados is right below that of fruit.
    """
    avocado = individual(bridge_module, avocado_world.avocado)
    for annotation_class in bridge_module.KnowRobBridge.classes_of(
        annotation_class=Avocado
    ):
        query = f"rdf:type(?avocado, sdt:'{annotation_class.__name__}')"
        assert session.ask(query=query) == [{"?avocado": avocado}]
    assert session.ask(query="rdfs:subClassOf(sdt:'Avocado', sdt:'Fruit')") == [{}]


def test_annotation_has_its_parts(bridge_module, session, avocado_world):
    """
    KnowRob knows that the avocado has its pit as a part.
    """
    answers = session.ask(query="dul:hasPart(?whole, ?part)")
    assert answers == [
        {
            "?whole": individual(bridge_module, avocado_world.avocado),
            "?part": individual(bridge_module, avocado_world.avocado.pit),
        }
    ]


def test_annotation_has_the_frame_of_its_body(bridge_module, session, avocado_world):
    """
    KnowRob knows the tf frame each annotation's body is published under.
    """
    frame_names = avocado_world.bridge.frame_names
    answers = session.ask(query="knowrob:frameName(?entity, ?frame)")
    assert {(answer["?entity"], answer["?frame"]) for answer in answers} == {
        (
            individual(bridge_module, annotation),
            frame_names.preferred_name(annotation.root),
        )
        for annotation in avocado_world.world.semantic_annotations
    }


def test_annotation_removed_from_the_world_is_forgotten(session, avocado_world):
    """
    Once the pit's annotation leaves the world, KnowRob knows of no pit any more.
    """
    world = avocado_world.world
    with world.modify_world():
        world.remove_semantic_annotation(avocado_world.avocado.pit)
    assert session.ask(query="rdf:type(?pit, sdt:'Pit')") == []


def test_stopped_bridge_leaves_nothing_behind(session, avocado_world):
    """
    A bridge that stops makes KnowRob forget everything it told it about the world.
    """
    avocado_world.bridge.stop()
    assert session.ask(query="rdf:type(?avocado, sdt:'Avocado')") == []
    assert session.ask(query="dul:hasPart(?whole, ?part)") == []
    assert session.ask(query="rdfs:subClassOf(sdt:'Avocado', sdt:'Fruit')") == []


# %% the physics KnowRob is told


@dataclass(kw_only=True)
class RipeAvocadoWorld:
    """
    A world holding a ripe avocado with flesh and pit whose qualities are known, the two
    halves of another avocado, and a bridge telling KnowRob about them.
    """

    world: World
    """
    The world.
    """

    avocado: Avocado
    """
    The annotation of the ripe avocado.
    """

    halves: list[AvocadoPiece]
    """
    The annotations of the two halves.
    """

    friction: ContactParameters
    """
    The contact parameters of the avocado's surface.
    """

    bridge: Any
    """
    The bridge from the world to KnowRob.
    """


@pytest.fixture
def ripe_avocado_world(
    bridge_module: ModuleType, session: Any
) -> Iterator[RipeAvocadoWorld]:
    """
    A ripe avocado and two halves, told to KnowRob; the bridge is stopped afterwards.
    """
    world = World()
    root = Body(name=PrefixedName("map"))
    avocado_body = Body(name=PrefixedName("ripe_avocado"))
    avocado_body.inertial = Inertial(mass=0.25)
    avocado_body.collision = ShapeCollection(
        [Box(origin=HomogeneousTransformationMatrix(reference_frame=avocado_body))],
        reference_frame=avocado_body,
    )
    friction = ContactParameters.create_for_surface(sliding_friction=0.6)
    friction.apply_to([avocado_body])
    flesh_body = Body(name=PrefixedName("ripe_avocado_flesh"))
    pit_body = Body(name=PrefixedName("ripe_avocado_pit"))
    for part_body in (flesh_body, pit_body):
        part_body.inertial = Inertial.negligible()
    half_bodies = [Body(name=PrefixedName(f"half_{number}")) for number in range(2)]
    with world.modify_world():
        for body in (root, avocado_body, flesh_body, pit_body, *half_bodies):
            world.add_kinematic_structure_entity(body)
        for body in (avocado_body, *half_bodies):
            world.add_connection(FixedConnection(parent=root, child=body))
        for part_body in (flesh_body, pit_body):
            world.add_connection(FixedConnection(parent=avocado_body, child=part_body))
        avocado = Avocado(
            name=avocado_body.name,
            root=avocado_body,
            ripeness=Ripeness(value=singleton(0.5).simple_sets[0]),
            firmness=Firmness.from_measurement(value=12.5),
            flesh=Flesh(
                name=flesh_body.name,
                root=flesh_body,
                cutting_resistance=CuttingResistance.from_measurement(value=800.0),
                bruising_pressure=BruisingPressure.from_measurement(value=2.5e5),
                material_elasticity=MaterialElasticity(
                    youngs_modulus=singleton(1.0e6).simple_sets[0],
                    poissons_ratio=singleton(0.35).simple_sets[0],
                ),
            ),
            pit=Pit(
                name=pit_body.name,
                root=pit_body,
                release_torque=ReleaseTorque.from_measurement(value=0.05),
            ),
        )
        halves = [AvocadoPiece(name=body.name, root=body) for body in half_bodies]
        world.add_semantic_annotation_recursively(avocado)
        for half in halves:
            world.add_semantic_annotation(half)
    bridge = bridge_module.KnowRobBridge(_world=world, session=session)
    yield RipeAvocadoWorld(
        world=world, avocado=avocado, halves=halves, friction=friction, bridge=bridge
    )
    bridge.stop()


def written(bridge_module: ModuleType, iri: str) -> str:
    """
    :param iri: The IRI of a class of the annotations or of their regions.
    :return: The class as written in queries.
    """
    return f"{bridge_module.Namespace.CLASSES.prefix}:'{iri.rpartition('#')[2]}'"


def quality_values(session: Any, *, attribute: str, value_property: str) -> dict:
    """
    :param attribute: The attribute class of a quality, as written in queries.
    :param value_property: The property giving the value of its regions, as written
        in queries.
    :return: The value of that quality of each individual that has one, by the
        individual's IRI.
    """
    answers = session.ask(
        query=f"dul:hasRegion(?entity, ?region), rdf:type(?region, {attribute}), "
        f"{value_property}(?region, ?value)"
    )
    return {answer["?entity"]: float(answer["?value"]) for answer in answers}


def test_mass_is_told_as_a_mass_attribute(bridge_module, session, ripe_avocado_world):
    """
    KnowRob knows the mass of the avocado as SOMA's mass attribute, and no mass of its
    parts, whose mass is the avocado's, or of the halves, which weigh nothing known.
    """
    avocado = ripe_avocado_world.avocado
    masses = quality_values(
        session, attribute="soma:'MassAttribute'", value_property="soma:hasMassValue"
    )
    assert masses == {individual(bridge_module, avocado): pytest.approx(avocado.mass)}


def test_friction_is_told_as_a_friction_attribute(
    bridge_module, session, ripe_avocado_world
):
    """
    KnowRob knows the sliding friction the avocado's body declares as SOMA's friction
    attribute.
    """
    frictions = quality_values(
        session,
        attribute="soma:'FrictionAttribute'",
        value_property="soma:hasFrictionValue",
    )
    assert frictions == {
        individual(bridge_module, ripe_avocado_world.avocado): pytest.approx(
            ripe_avocado_world.friction.friction.sliding
        )
    }


def test_ripeness_is_told_with_its_stage(bridge_module, session, ripe_avocado_world):
    """
    KnowRob knows how far the avocado has ripened, and that its ripeness lies in the
    stage that holds it, a stage of ripeness.
    """
    avocado = ripe_avocado_world.avocado
    stage = written(
        bridge_module,
        bridge_module.Namespace.CLASSES.camel_case_iri(
            words=avocado.ripeness.stage.name
        ),
    )
    ripeness = f"sdt:'{Ripeness.__name__}'"
    answers = session.ask(
        query=f"dul:hasRegion(?avocado, ?region), rdf:type(?region, {stage})"
    )
    assert [answer["?avocado"] for answer in answers] == [
        individual(bridge_module, avocado)
    ]
    for below, above in (
        (stage, f"sdt:'{RipenessStage.__name__}'"),
        (f"sdt:'{RipenessStage.__name__}'", ripeness),
        (ripeness, "dul:'PhysicalAttribute'"),
    ):
        assert session.ask(query=f"rdfs:subClassOf({below}, {above})") == [{}]
    values = quality_values(
        session, attribute=ripeness, value_property="dul:hasRegionDataValue"
    )
    assert values == {individual(bridge_module, avocado): avocado.ripeness.value.lower}


def test_ripeness_known_only_by_its_stage_is_told_without_a_value(
    bridge_module, session, ripe_avocado_world
):
    """
    Once the avocado is known only to be overripe, KnowRob knows its stage and no value
    of its ripeness.
    """
    world, avocado = ripe_avocado_world.world, ripe_avocado_world.avocado
    with world.modify_world():
        avocado.set_quality(quality=Ripeness.from_stage(stage=RipenessStage.OVERRIPE))
    stage = written(
        bridge_module,
        bridge_module.Namespace.CLASSES.camel_case_iri(
            words=RipenessStage.OVERRIPE.name
        ),
    )
    answers = session.ask(
        query=f"dul:hasRegion(?avocado, ?region), rdf:type(?region, {stage})"
    )
    assert [answer["?avocado"] for answer in answers] == [
        individual(bridge_module, avocado)
    ]
    values = quality_values(
        session,
        attribute=f"sdt:'{Ripeness.__name__}'",
        value_property="dul:hasRegionDataValue",
    )
    assert values == {}


def test_qualities_are_told_as_regions_holding_their_values(
    bridge_module, session, ripe_avocado_world
):
    """
    KnowRob knows each value of the qualities of the avocado, its flesh and its pit, and
    no quality that is not known; values are held in single precision.
    """
    single_precision = 1e-6
    avocado = ripe_avocado_world.avocado
    for annotation in (avocado, avocado.flesh, avocado.pit):
        for quality in annotation.qualities:
            for region in bridge_module.KnowRobBridge.quality_regions(quality=quality):
                values = quality_values(
                    session,
                    attribute=written(bridge_module, region.attribute_class),
                    value_property="dul:hasRegionDataValue",
                )
                assert values == {
                    individual(bridge_module, annotation): pytest.approx(
                        region.value.lower, rel=single_precision
                    )
                }
    assert (
        quality_values(
            session,
            attribute=f"sdt:'{PeelStrength.__name__}'",
            value_property="dul:hasRegionDataValue",
        )
        == {}
    )


def test_quality_of_two_values_is_told_as_a_region_for_each(
    bridge_module, ripe_avocado_world
):
    """
    The elasticity of the flesh is told as one region for each of its constants, named
    after the constant.
    """
    elasticity = ripe_avocado_world.avocado.flesh.material_elasticity
    regions = bridge_module.KnowRobBridge.quality_regions(quality=elasticity)
    assert [region.attribute_class for region in regions] == [
        bridge_module.Namespace.CLASSES.camel_case_iri(words=constant.name)
        for constant in fields(MaterialElasticity)
    ]
    assert [region.value for region in regions] == [
        elasticity.youngs_modulus,
        elasticity.poissons_ratio,
    ]


def test_quality_classes_are_physical_attributes(session, ripe_avocado_world):
    """
    KnowRob knows the classes of the qualities SOMA lacks as classes of physical
    attributes.
    """
    query = (
        f"rdfs:subClassOf(sdt:'{CuttingResistance.__name__}', dul:'PhysicalAttribute')"
    )
    assert session.ask(query=query) == [{}]


def test_only_cuttable_parts_have_cuttability(
    bridge_module, session, ripe_avocado_world
):
    """
    KnowRob knows that a knife can cut the flesh, and not that it can cut the pit.
    """
    answers = session.ask(
        query="soma:hasDisposition(?part, ?disposition), "
        "rdf:type(?disposition, soma:'Cuttability')"
    )
    assert [answer["?part"] for answer in answers] == [
        individual(bridge_module, ripe_avocado_world.avocado.flesh)
    ]


def test_adhesion_is_told_and_forgotten_once_it_breaks(
    bridge_module, session, ripe_avocado_world
):
    """
    KnowRob knows one half sticks to the other while it does, and no longer once it has
    come off.
    """
    world = ripe_avocado_world.world
    half, other_half = ripe_avocado_world.halves
    with world.modify_world():
        half.stick_to(other=other_half)
    assert session.ask(query="soma:isLinkedTo(?piece, ?other)") == [
        {
            "?piece": individual(bridge_module, half),
            "?other": individual(bridge_module, other_half),
        }
    ]
    with world.modify_world():
        half.come_off(other=other_half)
    assert session.ask(query="soma:isLinkedTo(?piece, ?other)") == []


def test_changed_quality_replaces_the_one_told(
    bridge_module, session, ripe_avocado_world
):
    """
    Once the flesh is bruised, then bruised harder, KnowRob knows only the worst bruise.
    """
    world, flesh = ripe_avocado_world.world, ripe_avocado_world.avocado.flesh
    for pressure in (3e5, 4e5):
        with world.modify_world():
            flesh.set_quality(
                quality=WorstBruisePressure.from_measurement(value=pressure)
            )
    values = quality_values(
        session,
        attribute=f"sdt:'{WorstBruisePressure.__name__}'",
        value_property="dul:hasRegionDataValue",
    )
    assert values == {
        individual(bridge_module, flesh): flesh.worst_bruise_pressure.value.lower
    }

"""
Tests of telling KnowRob what the semantic annotations of a world are.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Avocado,
    Pit,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
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
    KnowRob knows the avocado as an avocado and as the food it is, and that the class
    of avocados is right below that of fruit.
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

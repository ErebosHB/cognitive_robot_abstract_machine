"""
Tells KnowRob what the semantic annotations of a world are and keeps it in step with
the world: each annotation as an individual of its class and the classes above it, with
the parts it has, the tf frame of its body, its mass, friction and qualities, whether a
knife can cut it and what it sticks to.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from uuid import UUID

from krrood.patterns.mixin import Mixin
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.adapters.ros.tf_publisher import TfFrameNames
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from random_events.interval import SimpleInterval, singleton
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasRootKinematicStructureEntity,
    IsAdherent,
    IsCuttable,
)
from semantic_digital_twin.semantic_annotations.part_whole.relationship import HasParts
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    Quality,
    Ripeness,
    RipenessStage,
)
from semantic_digital_twin.semantic_annotations.qualities.relationship import (
    HasQualities,
)
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Any

# %% the vocabulary


class Vocabulary(StrEnum):
    """
    The IRIs of the terms of other ontologies the knowledge is stated in.
    """

    TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    CLASS = "http://www.w3.org/2002/07/owl#Class"
    HAS_PART = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasPart"
    FRAME_NAME = "http://knowrob.org/kb/knowrob.owl#frameName"
    HAS_REGION = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasRegion"
    HAS_REGION_DATA_VALUE = (
        "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasRegionDataValue"
    )
    PHYSICAL_ATTRIBUTE = (
        "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#PhysicalAttribute"
    )
    MASS_ATTRIBUTE = "http://www.ease-crc.org/ont/SOMA.owl#MassAttribute"
    HAS_MASS_VALUE = "http://www.ease-crc.org/ont/SOMA.owl#hasMassValue"
    FRICTION_ATTRIBUTE = "http://www.ease-crc.org/ont/SOMA.owl#FrictionAttribute"
    HAS_FRICTION_VALUE = "http://www.ease-crc.org/ont/SOMA.owl#hasFrictionValue"
    HAS_DISPOSITION = "http://www.ease-crc.org/ont/SOMA.owl#hasDisposition"
    CUTTABILITY = "http://www.ease-crc.org/ont/SOMA.owl#Cuttability"
    IS_LINKED_TO = "http://www.ease-crc.org/ont/SOMA.owl#isLinkedTo"

    @property
    def local_name(self) -> str:
        """
        :return: The name of the term within its ontology.
        """
        return self.rpartition("#")[2]


class StatementValueKind(StrEnum):
    """
    What the value of a statement is.
    """

    IRI = "iri"
    """
    The IRI of something.
    """

    TEXT = "text"
    """
    Text.
    """

    DOUBLE = "double"
    """
    A floating-point number, written as text.
    """


class Namespace(StrEnum):
    """
    The namespaces of the classes of the annotations and of the annotations
    themselves, as KnowRob knows them.
    """

    CLASSES = "https://github.com/cram2/cognitive_robot_abstract_machine/semantic_digital_twin"
    INDIVIDUALS = "https://github.com/cram2/cognitive_robot_abstract_machine/semantic_digital_twin/world"

    @property
    def prefix(self) -> str:
        """
        :return: The prefix the namespace is written with in queries.
        """
        match self:
            case Namespace.CLASSES:
                return "sdt"
            case Namespace.INDIVIDUALS:
                return "world"

    def iri(self, *, name: str) -> str:
        """
        :param name: The name of a class or an individual in this namespace.
        :return: Its IRI.
        """
        return f"{self}#{name}"

    def camel_case_iri(self, *, words: str) -> str:
        """
        :param words: Words joined by underscores, in any case, such as
            ``youngs_modulus``.
        :return: The IRI of the name the words make, each capitalised and run together.
        """
        return self.iri(name="".join(word.capitalize() for word in words.split("_")))

    @classmethod
    def full_iri(cls, *, written: str) -> str:
        """
        :param written: A value as KnowRob writes it, shortened with the prefix of a
            namespace if it is an IRI in one.
        :return: The value, an IRI in one of these namespaces written in full.
        """
        for namespace in cls:
            prefix = f"{namespace.prefix}:"
            if written.startswith(prefix):
                return namespace.iri(name=written.removeprefix(prefix).strip("'"))
        return written.strip("'")


class SettingsFile(PathEnumeration):
    """
    The files KnowRob is set up from.
    """

    KNOWROB_SETTINGS = Path(__file__).resolve().parent / "knowrob_settings.json"
    """
    In-memory storage with the Prolog reasoner, and the ontologies of KnowRob and DUL.
    """


@dataclass(frozen=True, kw_only=True)
class Statement:
    """
    A statement to KnowRob: a subject, the predicate said of it and its value.
    """

    subject: str
    """
    The IRI of what the statement is about.
    """

    predicate: str
    """
    The IRI of what is said of the subject.
    """

    value: str
    """
    The IRI the predicate relates the subject to, or the text or number it gives it.
    """

    value_kind: StatementValueKind = StatementValueKind.IRI
    """
    What :attr:`value` is.
    """


@dataclass(frozen=True, kw_only=True)
class AttributeRegion:
    """
    A region of a class of physical attributes that an individual has, holding the value
    of one of its attributes, such as its mass.
    """

    attribute_class: str
    """
    The IRI of the class of the region.
    """

    value: SimpleInterval
    """
    The value the region holds, in the unit of its class.
    """

    value_property: str = Vocabulary.HAS_REGION_DATA_VALUE
    """
    The IRI of the property relating the region to its value.
    """

    def region_iri(self, *, owner: str) -> str:
        """
        :param owner: The IRI of the individual that has the region.
        :return: The IRI of the region, named after the individual and the class.
        """
        return f"{owner}_{self.attribute_class.rpartition('#')[2]}"

    def to_statements(self, *, owner: str) -> list[Statement]:
        """
        :param owner: The IRI of the individual that has the region.
        :return: That the individual has the region and the region is of its class, and
            the value it holds where that is a single point; KnowRob has no ranges.
        """
        region = self.region_iri(owner=owner)
        statements = [
            Statement(subject=owner, predicate=Vocabulary.HAS_REGION, value=region),
            Statement(
                subject=region, predicate=Vocabulary.TYPE, value=self.attribute_class
            ),
        ]
        if self.value.is_singleton():
            statements.append(
                Statement(
                    subject=region,
                    predicate=self.value_property,
                    value=repr(float(self.value.lower)),
                    value_kind=StatementValueKind.DOUBLE,
                )
            )
        return statements


# %% the session


@dataclass(kw_only=True)
class KnowRobSession:
    """
    A KnowRob knowledge base, embedded in this process, that is told statements and
    asked queries in KnowRob's query language.
    """

    library: ModuleType
    """
    The ``knowrob`` module.
    """

    knowledge_base: Any
    """
    The knowledge base, a ``knowrob.KnowledgeBase``.
    """

    @classmethod
    def start(
        cls, *, settings_file: Path = SettingsFile.KNOWROB_SETTINGS
    ) -> KnowRobSession:
        """
        :param settings_file: The KnowRob settings; the prefixes of :class:`Namespace`
            are added to them.
        :return: A session with a new knowledge base.
        """
        # KnowRob embeds SWI-Prolog, whose compiled libraries link against it; that
        # works only if the symbols of the module are shared while it is loaded.
        flags = sys.getdlopenflags()
        sys.setdlopenflags(flags | os.RTLD_GLOBAL)
        library = importlib.import_module("knowrob")
        sys.setdlopenflags(flags)
        settings = json.loads(settings_file.read_text())
        settings["semantic-web"] = {
            "prefixes": [
                {"alias": namespace.prefix, "uri": str(namespace)}
                for namespace in Namespace
            ]
        }
        return cls(
            library=library,
            knowledge_base=library.KnowledgeBase(json.dumps(settings)),
        )

    def tell(self, *, statements: list[Statement]) -> None:
        """
        :param statements: Statements KnowRob is to know from now on.
        """
        for statement in statements:
            self.knowledge_base.insertOne(self.triple_of(statement=statement))

    def forget(self, *, statements: list[Statement]) -> None:
        """
        :param statements: Statements KnowRob was told and is to know no more.
        """
        for statement in statements:
            self.knowledge_base.removeOne(self.triple_of(statement=statement))

    def triple_of(self, *, statement: Statement) -> Any:
        """
        :param statement: A statement.
        :return: The statement as a KnowRob triple.
        """
        triple = self.library.TripleCopy()
        triple.setSubject(statement.subject)
        triple.setPredicate(statement.predicate)
        match statement.value_kind:
            case StatementValueKind.IRI:
                triple.setObjectIRI(statement.value)
            case StatementValueKind.TEXT:
                triple.setXSDValue(statement.value, self.library.XSDType.STRING)
            case StatementValueKind.DOUBLE:
                triple.setXSDValue(statement.value, self.library.XSDType.DOUBLE)
        return triple

    def ask(self, *, query: str) -> list[dict[str, str]]:
        """
        :param query: A query in KnowRob's query language, such as
            ``rdf:type(?piece, sdt:'AvocadoPiece')``.
        :return: The bindings of each answer KnowRob knows to be true, keyed by the
            variables as written in the query, IRIs of :class:`Namespace` in full;
            empty if it knows none.
        """
        library = self.library
        modality = {
            "epistemicOperator": library.EpistemicOperator.KNOWLEDGE,
            "aboutAgentIRI": "",
            "confidence": 0.0,
            "temporalOperator": library.TemporalOperator.CURRENTLY,
            "minPastTimestamp": -1.0,
            "maxPastTimestamp": -1.0,
        }
        formula = library.InterfaceUtils.applyModality(
            modality, library.QueryParser.parse(query)
        )
        queue = self.knowledge_base.submitQuery(
            formula, library.QueryContext(library.QueryFlag.QUERY_FLAG_ALL_SOLUTIONS)
        ).createQueue()
        answers = []
        token = queue.pop_front()
        while token.tokenType() == library.TokenType.ANSWER_TOKEN:
            if token.isPositive():
                # Each binding is a reference, the variable and its value.
                answers.append(
                    {
                        str(binding[1]): Namespace.full_iri(written=str(binding[2]))
                        for binding in token.substitution()
                    }
                )
            token = queue.pop_front()
        return answers


# %% the bridge


@dataclass(eq=False, kw_only=True)
class KnowRobBridge(ModelChangeCallback):
    """
    Keeps a KnowRob session knowing what the semantic annotations of a world are, for
    as long as it is attached to the world: what an annotation added to the world is,
    from then on, and nothing about one removed from it.
    """

    session: KnowRobSession
    """
    The session that is told about the world.
    """

    told: dict[UUID, list[Statement]] = field(init=False, default_factory=dict)
    """
    The statements told about each annotation in the world, by the annotation's id.
    """

    told_about_classes: dict[str, list[Statement]] = field(
        init=False, default_factory=dict
    )
    """
    The statements told about each class the annotations and their regions are of, by
    the IRI of the class.
    """

    frame_names: TfFrameNames = field(init=False, default_factory=TfFrameNames)
    """
    Names the tf frame of each annotation's body as the world's tf publisher does.
    """

    def __post_init__(self) -> None:
        """
        Tell the session about the stages of ripeness, and about the annotations the
        world has already.
        """
        super().__post_init__()
        ripeness = Namespace.CLASSES.iri(name=Ripeness.__name__)
        stage_class = Namespace.CLASSES.iri(name=RipenessStage.__name__)
        self.tell_about_class(iri=ripeness, above=[Vocabulary.PHYSICAL_ATTRIBUTE])
        self.tell_about_class(iri=stage_class, above=[ripeness])
        for stage in RipenessStage:
            self.tell_about_class(
                iri=Namespace.CLASSES.camel_case_iri(words=stage.name),
                above=[stage_class],
            )
        self.on_model_change()

    def on_model_change(self, **kwargs: Any) -> None:
        annotations = {
            annotation.id: annotation for annotation in self._world.semantic_annotations
        }
        parts = {
            part.id
            for annotation in annotations.values()
            if isinstance(annotation, HasParts)
            for part in annotation.parts
        }
        for gone in [
            annotation_id
            for annotation_id in self.told
            if annotation_id not in annotations
        ]:
            self.session.forget(statements=self.told.pop(gone))
        for annotation_id, annotation in annotations.items():
            self.tell_about_classes_used(annotation=annotation)
            statements = self.statements_about(
                annotation=annotation, is_part=annotation_id in parts
            )
            told = self.told.get(annotation_id, [])
            self.session.forget(
                statements=[
                    statement for statement in told if statement not in statements
                ]
            )
            self.session.tell(
                statements=[
                    statement for statement in statements if statement not in told
                ]
            )
            self.told[annotation_id] = statements

    def stop(self) -> None:
        """
        Make the session forget everything told about the world, and detach from it.
        """
        for statements in [*self.told.values(), *self.told_about_classes.values()]:
            self.session.forget(statements=statements)
        self.told.clear()
        self.told_about_classes.clear()
        super().stop()

    def tell_about_class(self, *, iri: str, above: list[str]) -> None:
        """
        Tell the session, once, that something is a class, and the classes right above
        it.

        :param iri: The IRI of the class.
        :param above: The IRIs of the classes right above it.
        """
        if iri in self.told_about_classes:
            return
        statements = [
            Statement(subject=iri, predicate=Vocabulary.TYPE, value=Vocabulary.CLASS)
        ]
        statements.extend(
            Statement(subject=iri, predicate=Vocabulary.SUBCLASS_OF, value=parent)
            for parent in above
        )
        self.told_about_classes[iri] = statements
        self.session.tell(statements=statements)

    def tell_about_classes_used(self, *, annotation: SemanticAnnotation) -> None:
        """
        Tell the session about the class of an annotation and each class of annotations
        above it, and that the class of each region of its qualities is one of physical
        attributes.

        :param annotation: A semantic annotation of the world.
        """
        for annotation_class in self.classes_of(annotation_class=type(annotation)):
            self.tell_about_class(
                iri=Namespace.CLASSES.iri(name=annotation_class.__name__),
                above=[
                    Namespace.CLASSES.iri(name=parent.__name__)
                    for parent in self.classes_of(annotation_class=annotation_class)[
                        1:2
                    ]
                ],
            )
        if not isinstance(annotation, HasQualities):
            return
        for quality in annotation.qualities:
            for region in self.quality_regions(quality=quality):
                self.tell_about_class(
                    iri=region.attribute_class, above=[Vocabulary.PHYSICAL_ATTRIBUTE]
                )

    def statements_about(
        self, *, annotation: SemanticAnnotation, is_part: bool
    ) -> list[Statement]:
        """
        :param annotation: A semantic annotation of the world.
        :param is_part: Whether the annotation is a part of another one, whose mass
            holds its own.
        :return: That it is of its class and of each class above it, that it has each
            of its parts, the name of its body's tf frame, its mass, friction and
            qualities, whether a knife can cut it and what it sticks to.
        """
        iri = Namespace.INDIVIDUALS.iri(name=annotation.name.name)
        statements = [
            Statement(
                subject=iri,
                predicate=Vocabulary.TYPE,
                value=Namespace.CLASSES.iri(name=annotation_class.__name__),
            )
            for annotation_class in self.classes_of(annotation_class=type(annotation))
        ]
        if isinstance(annotation, HasParts):
            statements.extend(
                Statement(
                    subject=iri,
                    predicate=Vocabulary.HAS_PART,
                    value=Namespace.INDIVIDUALS.iri(name=part.name.name),
                )
                for part in annotation.parts
            )
        if isinstance(annotation, HasRootKinematicStructureEntity):
            statements.append(
                Statement(
                    subject=iri,
                    predicate=Vocabulary.FRAME_NAME,
                    value=self.frame_names.preferred_name(annotation.root),
                    value_kind=StatementValueKind.TEXT,
                )
            )
        statements.extend(
            statement
            for region in self.body_regions(annotation=annotation, is_part=is_part)
            for statement in region.to_statements(owner=iri)
        )
        if isinstance(annotation, HasQualities):
            statements.extend(
                statement
                for quality in annotation.qualities
                for statement in self.statements_about_quality(iri=iri, quality=quality)
            )
        if isinstance(annotation, IsCuttable):
            disposition = f"{iri}_{Vocabulary.CUTTABILITY.local_name}"
            statements.extend(
                (
                    Statement(
                        subject=iri,
                        predicate=Vocabulary.HAS_DISPOSITION,
                        value=disposition,
                    ),
                    Statement(
                        subject=disposition,
                        predicate=Vocabulary.TYPE,
                        value=Vocabulary.CUTTABILITY,
                    ),
                )
            )
        if isinstance(annotation, IsAdherent):
            statements.extend(
                Statement(
                    subject=iri,
                    predicate=Vocabulary.IS_LINKED_TO,
                    value=Namespace.INDIVIDUALS.iri(name=other.name.name),
                )
                for other in annotation.adheres_to
            )
        return statements

    @staticmethod
    def body_regions(
        *, annotation: SemanticAnnotation, is_part: bool
    ) -> list[AttributeRegion]:
        """
        :param annotation: A semantic annotation of the world.
        :param is_part: Whether the annotation is a part of another one, whose mass
            holds its own.
        :return: Its mass where it weighs something and is no part, and the sliding
            friction of its body's surface where the body declares contact parameters.
        """
        regions = []
        if not is_part and annotation.mass > 0.0:
            regions.append(
                AttributeRegion(
                    attribute_class=Vocabulary.MASS_ATTRIBUTE,
                    value_property=Vocabulary.HAS_MASS_VALUE,
                    value=singleton(annotation.mass).simple_sets[0],
                )
            )
        if (
            not isinstance(annotation, HasRootBody)
            or not annotation.root.has_collision()
        ):
            return regions
        declared = [
            parameters
            for shape in annotation.root.collision
            for parameters in [shape.get_simulator_property_of_type(ContactParameters)]
            if parameters is not None
        ]
        if declared:
            friction = max(parameters.friction.sliding for parameters in declared)
            regions.append(
                AttributeRegion(
                    attribute_class=Vocabulary.FRICTION_ATTRIBUTE,
                    value_property=Vocabulary.HAS_FRICTION_VALUE,
                    value=singleton(friction).simple_sets[0],
                )
            )
        return regions

    @staticmethod
    def quality_regions(*, quality: Quality) -> list[AttributeRegion]:
        """
        :param quality: A quality of an annotation.
        :return: A region for each value of the quality, of a class named after the
            quality where it has one value, else after the field of each value.
        """
        values = {
            quality_field.name: getattr(quality, quality_field.name)
            for quality_field in fields(quality)
        }
        intervals = {
            name: value
            for name, value in values.items()
            if isinstance(value, SimpleInterval)
        }
        if len(intervals) == 1:
            return [
                AttributeRegion(
                    attribute_class=Namespace.CLASSES.iri(name=type(quality).__name__),
                    value=value,
                )
                for value in intervals.values()
            ]
        return [
            AttributeRegion(
                attribute_class=Namespace.CLASSES.camel_case_iri(words=name),
                value=value,
            )
            for name, value in intervals.items()
        ]

    def statements_about_quality(
        self, *, iri: str, quality: Quality
    ) -> list[Statement]:
        """
        :param iri: The IRI of an individual.
        :param quality: One of its qualities.
        :return: That the individual has the regions of the quality, and that the region
            of a ripeness is of the stage that holds it.
        """
        regions = self.quality_regions(quality=quality)
        statements = [
            statement
            for region in regions
            for statement in region.to_statements(owner=iri)
        ]
        if isinstance(quality, Ripeness):
            statements.extend(
                Statement(
                    subject=region.region_iri(owner=iri),
                    predicate=Vocabulary.TYPE,
                    value=Namespace.CLASSES.camel_case_iri(words=stage.name),
                )
                for region in regions
                for stage in RipenessStage
                if stage.holds(value=quality.value)
            )
        return statements

    @staticmethod
    def classes_of(*, annotation_class: type) -> list[type]:
        """
        :param annotation_class: A class of semantic annotations.
        :return: The class and the classes of annotations above it, the class first,
            leaving out mixins.
        """
        return [
            candidate
            for candidate in annotation_class.__mro__
            if issubclass(candidate, SemanticAnnotation)
            and candidate is not SemanticAnnotation
            and Mixin not in candidate.__bases__
        ]

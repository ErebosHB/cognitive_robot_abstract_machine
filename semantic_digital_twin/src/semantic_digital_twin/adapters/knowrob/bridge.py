"""
Tells KnowRob what the semantic annotations of a world are and keeps it in step with
the world: each annotation as an individual of its class and the classes above it, with
the parts it has and the tf frame of its body.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from uuid import UUID

from krrood.patterns.mixin import Mixin
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.adapters.ros.tf_publisher import TfFrameNames
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from semantic_digital_twin.part_finding.relationships import PartRelationshipInspector
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
    PartWholeRelationship,
)
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
    The IRI the predicate relates the subject to, or the text it gives it.
    """

    value_is_text: bool = False
    """
    Whether :attr:`value` is text rather than an IRI.
    """


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
        if statement.value_is_text:
            triple.setXSDValue(statement.value, self.library.XSDType.STRING)
        else:
            triple.setObjectIRI(statement.value)
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

    told_about_classes: dict[type, list[Statement]] = field(
        init=False, default_factory=dict
    )
    """
    The statements told about each class of the annotations.
    """

    frame_names: TfFrameNames = field(init=False, default_factory=TfFrameNames)
    """
    Names the tf frame of each annotation's body as the world's tf publisher does.
    """

    def __post_init__(self) -> None:
        """
        Tell the session about the annotations the world has already.
        """
        super().__post_init__()
        self.on_model_change()

    def on_model_change(self, **kwargs: Any) -> None:
        annotations = {
            annotation.id: annotation for annotation in self._world.semantic_annotations
        }
        for gone in [
            annotation_id
            for annotation_id in self.told
            if annotation_id not in annotations
        ]:
            self.session.forget(statements=self.told.pop(gone))
        for annotation_id, annotation in annotations.items():
            if annotation_id in self.told:
                continue
            for annotation_class in self.classes_of(annotation_class=type(annotation)):
                self.tell_about_class(annotation_class=annotation_class)
            self.told[annotation_id] = self.statements_about(annotation=annotation)
            self.session.tell(statements=self.told[annotation_id])

    def stop(self) -> None:
        """
        Make the session forget everything told about the world, and detach from it.
        """
        for statements in [*self.told.values(), *self.told_about_classes.values()]:
            self.session.forget(statements=statements)
        self.told.clear()
        self.told_about_classes.clear()
        super().stop()

    def tell_about_class(self, *, annotation_class: type) -> None:
        """
        Tell the session, once, that a class of annotations is a class, and the class
        right above it.

        :param annotation_class: A class of semantic annotations.
        """
        if annotation_class in self.told_about_classes:
            return
        iri = Namespace.CLASSES.iri(name=annotation_class.__name__)
        statements = [
            Statement(subject=iri, predicate=Vocabulary.TYPE, value=Vocabulary.CLASS)
        ]
        above = self.classes_of(annotation_class=annotation_class)[1:2]
        statements.extend(
            Statement(
                subject=iri,
                predicate=Vocabulary.SUBCLASS_OF,
                value=Namespace.CLASSES.iri(name=parent.__name__),
            )
            for parent in above
        )
        self.told_about_classes[annotation_class] = statements
        self.session.tell(statements=statements)

    def statements_about(self, *, annotation: SemanticAnnotation) -> list[Statement]:
        """
        :param annotation: A semantic annotation of the world.
        :return: That it is of its class and of each class above it, that it has each
            of its parts, and the name of its body's tf frame.
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
        if isinstance(annotation, PartWholeRelationship):
            statements.extend(
                Statement(
                    subject=iri,
                    predicate=Vocabulary.HAS_PART,
                    value=Namespace.INDIVIDUALS.iri(name=part.name.name),
                )
                for relationship in PartRelationshipInspector.all_of_type(
                    type(annotation)
                )
                for part in relationship.parts_of(annotation)
            )
        if isinstance(annotation, HasRootKinematicStructureEntity):
            statements.append(
                Statement(
                    subject=iri,
                    predicate=Vocabulary.FRAME_NAME,
                    value=self.frame_names.preferred_name(annotation.root),
                    value_is_text=True,
                )
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

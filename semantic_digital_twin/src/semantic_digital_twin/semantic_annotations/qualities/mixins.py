"""
Mixins for each kind of quality an annotation can have, such as the ripeness of a fruit
or the mass density of its flesh.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.qualities.relationship_marker import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    BruisingPressure,
    CuttingResistance,
    Firmness,
    MassDensity,
    MaterialElasticity,
    PeelStrength,
    ReleaseTorque,
    Ripeness,
    ScoopingResistance,
    WorstBruisePressure,
)
from semantic_digital_twin.semantic_annotations.qualities.relationship import (
    HasQualities,
)

# %% Qualities of food


@dataclass(eq=False, kw_only=True)
class HasRipeness(HasQualities, Mixin):
    """
    A mixin class for semantic annotations that ripen.
    """

    ripeness: Ripeness | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How far the semantic annotation has ripened, or ``None`` when that is unknown.
    """


@dataclass(eq=False, kw_only=True)
class HasFirmness(HasQualities, Mixin):
    """
    A mixin class for semantic annotations whose firmness is measured as a whole, such as
    a fruit.
    """

    firmness: Firmness | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How firm the semantic annotation is, or ``None`` when that is unknown.
    """


# %% Material qualities


@dataclass(eq=False, kw_only=True)
class HasMassDensity(HasQualities, Mixin):
    """
    A mixin class for semantic annotations made of a material with a mass density.
    """

    mass_density: MassDensity | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the semantic annotation's material, or ``None`` when that is unknown.
    """


@dataclass(eq=False, kw_only=True)
class HasMaterialElasticity(HasMassDensity, Mixin):
    """
    A mixin class for semantic annotations made of an elastic material, which, being
    matter, also has a mass density.
    """

    material_elasticity: MaterialElasticity | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elasticity of the semantic annotation's material, or ``None`` when that is
    unknown.
    """


# %% Resistance to tools


@dataclass(eq=False, kw_only=True)
class HasCuttingResistance(HasQualities, Mixin):
    """
    A mixin class for semantic annotations of material a blade has to cut its way
    through.
    """

    cutting_resistance: CuttingResistance | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How hard the material resists a blade, or ``None`` when that is unknown.
    """


@dataclass(eq=False, kw_only=True)
class HasScoopingResistance(HasQualities, Mixin):
    """
    A mixin class for semantic annotations of material a spoon can scoop out.
    """

    scooping_resistance: ScoopingResistance | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How hard the material resists a spoon, or ``None`` when that is unknown.
    """


@dataclass(eq=False, kw_only=True)
class HasPeelStrength(HasQualities, Mixin):
    """
    A mixin class for semantic annotations of a peel that can be pulled off what it
    covers.
    """

    peel_strength: PeelStrength | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How hard the peel holds on, or ``None`` when that is unknown.
    """


@dataclass(eq=False, kw_only=True)
class HasReleaseTorque(HasQualities, Mixin):
    """
    A mixin class for semantic annotations of a pit that is twisted free of the flesh
    it sticks in.
    """

    release_torque: ReleaseTorque | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How hard the pit holds on, or ``None`` when that is unknown.
    """


# %% Bruising


@dataclass(eq=False, kw_only=True)
class HasBruisingPressure(HasQualities, Mixin):
    """
    A mixin class for semantic annotations of tissue a hard enough press bruises.
    """

    bruising_pressure: BruisingPressure | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How hard a press has to be to bruise the tissue, or ``None`` when that is
    unknown.
    """


@dataclass(eq=False, kw_only=True)
class HasWorstBruisePressure(HasQualities, Mixin):
    """
    A mixin class for semantic annotations of tissue that records the worst press that
    bruised it.
    """

    worst_bruise_pressure: WorstBruisePressure | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The worst press that has bruised the tissue, or ``None`` while it is not bruised.
    """

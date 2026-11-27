"""
How an avocado's texture depends on its ripeness.

One number, the firmness a standard press measures on the whole fruit, sets its ripeness
and every property; each property runs log-linearly from its value in a hard fruit to
its value in an overripe one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
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
    ScoopingResistance,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Avocado,
    AvocadoPiece,
    Flesh,
    Peel,
    Pit,
)

# %% one property


class DrawnQuantity(IntEnum):
    """
    What a fruit's seed draws, each from a stream of its own spawned from the seed.
    """

    FIRMNESS = 0
    """
    The firmness of a fruit of random ripeness.
    """

    VARIATION = 1
    """
    How the fruit's properties differ from those of other fruits as firm.
    """

    def generator(self, *, seed: int) -> np.random.Generator:
        """
        :param seed: The seed of a fruit.
        :return: The random generator this quantity is drawn with.
        """
        return np.random.default_rng(
            np.random.SeedSequence(seed).spawn(len(DrawnQuantity))[self]
        )


@dataclass(kw_only=True)
class TextureProperty:
    """
    A property of an avocado's texture over its ripening.
    """

    when_hard: float
    """
    The property's value in a fruit as firm as :attr:`AvocadoTexture.hard_firmness`.
    """

    when_overripe: float
    """
    The property's value in a fruit as firm as :attr:`AvocadoTexture.overripe_firmness`.
    """

    relative_spread: float = 0.2
    """
    How much the property varies between fruits of the same firmness: the standard
    deviation of its logarithm.
    """

    def value_at(self, *, softening: float) -> float:
        """
        :param softening: How far the fruit has softened, from 0 when hard to 1 when
            overripe.
        :return: The property's value then, log-linearly between the two ends.
        """
        return math.exp(
            (1 - softening) * math.log(self.when_hard)
            + softening * math.log(self.when_overripe)
        )


# %% the texture over ripening


@dataclass(kw_only=True)
class AvocadoTexture:
    """
    How every property of an avocado's texture depends on its firmness.

    Firmness is the force in newtons a flat probe of 5 cm diameter reaches when pressed
    10 mm into the whole fruit at its equator.
    """

    hard_firmness: float = 200.0
    """
    The firmness of a fruit just picked, in newtons, after Ngubane et al.

    2024.
    """

    overripe_firmness: float = 7.0
    """
    The firmness of a fruit past eating, in newtons, after Ngubane et al.

    2024.
    """

    press_depth: float = 0.01
    """
    How far in metres the standard press goes into the fruit.
    """

    stage_firmness: dict[RipenessStage, float] = field(
        default_factory=lambda: {
            RipenessStage.RIPE: 40.0,
            RipenessStage.OVERRIPE: 10.0,
        }
    )
    """
    The firmness in newtons at which a fruit enters each stage of ripeness after the
    first.

    Overripe below 10 N follows Jeong and Huber 2004; ripe, firm enough to slice, below
    40 N is an estimate.
    """

    elastic_modulus: TextureProperty = field(
        default_factory=lambda: TextureProperty(when_hard=2.29e6, when_overripe=0.16e6)
    )
    """
    The flesh's elastic modulus in pascals, from picking to eating ripe after Fariñas et
    al.

    2021.
    """

    bruising_pressure: TextureProperty = field(
        default_factory=lambda: TextureProperty(when_hard=1.02e6, when_overripe=1.03e5)
    )
    """
    The contact pressure in pascals above which a press bruises the flesh: the flesh's
    yield stress, 0 and 20 days after harvest after Baryeh 2000.
    """

    flesh_cutting_resistance: TextureProperty = field(
        default_factory=lambda: TextureProperty(
            when_hard=1000.0, when_overripe=90.0, relative_spread=0.3
        )
    )
    """
    The force in newtons per metre of cutting edge a blade needs to go through flesh; an
    estimate, falling like the rupture stress Baryeh 2000 measured.
    """

    skin_cutting_resistance: TextureProperty = field(
        default_factory=lambda: TextureProperty(
            when_hard=1500.0, when_overripe=1000.0, relative_spread=0.3
        )
    )
    """
    The force in newtons per metre of cutting edge a blade needs to go through skin,
    whose lignified cells barely soften; an estimate.
    """

    scooping_resistance: TextureProperty = field(
        default_factory=lambda: TextureProperty(
            when_hard=4e5, when_overripe=1e4, relative_spread=0.3
        )
    )
    """
    The pressure in pascals on its leading face a spoon's bowl needs to push through
    flesh; an estimate.
    """

    peel_strength: TextureProperty = field(
        default_factory=lambda: TextureProperty(
            when_hard=400.0, when_overripe=20.0, relative_spread=0.3
        )
    )
    """
    The force in newtons per metre of strip width that pulls skin off the flesh; an
    estimate.
    """

    release_torque: TextureProperty = field(
        default_factory=lambda: TextureProperty(
            when_hard=1.0, when_overripe=0.02, relative_spread=0.3
        )
    )
    """
    The torque in newton metres that twists the pit free of the flesh round it; an
    estimate.
    """

    poisson_ratio: float = 0.35
    """
    The flesh's Poisson ratio, within the 0.27 to 0.41 measured on avocados.
    """

    slicing_relief: float = 0.4
    """
    The share of the cutting resistance a blade saves at most by drawing along its edge
    while it pushes, after DiSECt's 40 % less force for slicing.
    """

    def softening(self, *, firmness: float) -> float:
        """
        :param firmness: The firmness of a fruit, in newtons.
        :return: How far it has softened, from 0 at :attr:`hard_firmness` to 1 at
            :attr:`overripe_firmness`, log-linearly and clipped to that range.
        """
        share = math.log(self.hard_firmness / firmness) / math.log(
            self.hard_firmness / self.overripe_firmness
        )
        return min(1.0, max(0.0, share))

    def ripeness(self, *, firmness: float) -> Ripeness:
        """
        :param firmness: The firmness of a fruit, in newtons.
        :return: How far it has ripened, from the start of ripening at
            :attr:`hard_firmness` to its end at :attr:`overripe_firmness`, entering each
            stage at its :attr:`stage_firmness`; log-linear in between, clipped.
        """
        points = sorted(
            [
                (self.hard_firmness, Ripeness.scale.lower),
                *(
                    (stage_firmness, stage.value.lower)
                    for stage, stage_firmness in self.stage_firmness.items()
                ),
                (self.overripe_firmness, Ripeness.scale.upper),
            ],
            reverse=True,
        )
        position = np.interp(
            -math.log(firmness),
            [-math.log(point_firmness) for point_firmness, _ in points],
            [point_position for _, point_position in points],
        )
        return Ripeness(value=singleton(float(position)).simple_sets[0])


# %% the texture of one fruit


@dataclass(kw_only=True)
class FruitTexture:
    """
    The texture of one avocado: its firmness, and the value of each property of
    :class:`AvocadoTexture` for it, varied as one fruit varies from another.
    """

    texture: AvocadoTexture
    """
    How the properties depend on firmness.
    """

    firmness: float
    """
    The fruit's firmness in newtons.
    """

    seed: int
    """
    The seed the fruit's variation is drawn from; the same seed gives the same fruit.
    """

    ripeness: Ripeness = field(init=False)
    """
    How far the fruit has ripened.
    """

    elastic_modulus: float = field(init=False)
    """
    See :attr:`AvocadoTexture.elastic_modulus`.
    """

    bruising_pressure: float = field(init=False)
    """
    See :attr:`AvocadoTexture.bruising_pressure`.
    """

    flesh_cutting_resistance: float = field(init=False)
    """
    See :attr:`AvocadoTexture.flesh_cutting_resistance`.
    """

    skin_cutting_resistance: float = field(init=False)
    """
    See :attr:`AvocadoTexture.skin_cutting_resistance`.
    """

    scooping_resistance: float = field(init=False)
    """
    See :attr:`AvocadoTexture.scooping_resistance`.
    """

    peel_strength: float = field(init=False)
    """
    See :attr:`AvocadoTexture.peel_strength`.
    """

    release_torque: float = field(init=False)
    """
    See :attr:`AvocadoTexture.release_torque`.
    """

    def __post_init__(self) -> None:
        """
        Draw the value of each property for this fruit.
        """
        texture = self.texture
        softening = texture.softening(firmness=self.firmness)
        generator = DrawnQuantity.VARIATION.generator(seed=self.seed)

        def drawn(texture_property: TextureProperty) -> float:
            """
            :param texture_property: A property of the texture.
            :return: Its value for this fruit.
            """
            return texture_property.value_at(softening=softening) * math.exp(
                generator.normal(scale=texture_property.relative_spread)
            )

        self.ripeness = texture.ripeness(firmness=self.firmness)
        self.elastic_modulus = drawn(texture.elastic_modulus)
        self.bruising_pressure = drawn(texture.bruising_pressure)
        self.flesh_cutting_resistance = drawn(texture.flesh_cutting_resistance)
        self.skin_cutting_resistance = drawn(texture.skin_cutting_resistance)
        self.scooping_resistance = drawn(texture.scooping_resistance)
        self.peel_strength = drawn(texture.peel_strength)
        self.release_torque = drawn(texture.release_torque)

    @classmethod
    def random(cls, *, texture: AvocadoTexture, seed: int) -> FruitTexture:
        """
        :param texture: How the properties depend on firmness.
        :param seed: The seed of the fruit.
        :return: A fruit of a firmness drawn log-uniformly between hard and overripe.
        """
        generator = DrawnQuantity.FIRMNESS.generator(seed=seed)
        firmness = math.exp(
            generator.uniform(
                math.log(texture.overripe_firmness), math.log(texture.hard_firmness)
            )
        )
        return cls(texture=texture, firmness=firmness, seed=seed)

    def contact_pressure(self, *, force: float, radius: float) -> float:
        """
        :param force: The force in newtons a contact presses the fruit with.
        :param radius: The radius of curvature in metres of the fruit where it is
            pressed.
        :return: The mean pressure in pascals over the area the fruit flattens to
            against a flat surface, after Hertz.
        """
        effective_modulus = self.elastic_modulus / (1.0 - self.texture.poisson_ratio**2)
        contact_radius = (3.0 * force * radius / (4.0 * effective_modulus)) ** (1 / 3)
        return force / (math.pi * contact_radius**2)

    def contact_time_constant(self, *, mass: float, impedance: float) -> float:
        """
        :param mass: The mass of the whole fruit, in kilograms.
        :param impedance: The share of a press a contact resists at once.
        :return: How fast in seconds a contact pushes the fruit back out, so that a
            steady press sinks into it as far per newton as the standard press does.
        """
        compliance = self.texture.press_depth / self.firmness
        return math.sqrt(compliance * mass * impedance / (1.0 - impedance))

    def describe(
        self, *, annotation: Avocado | AvocadoPiece | Peel | Flesh | Pit
    ) -> None:
        """
        State this fruit's ripeness and firmness, or how the material of one of its
        parts gives way, in the annotation of a piece of it or of a part, before it
        joins a world.

        :param annotation: The annotation.
        """
        match annotation:
            case Avocado() | AvocadoPiece():
                annotation.ripeness = Ripeness(value=self.ripeness.value)
                annotation.firmness = Firmness.from_measurement(value=self.firmness)
            case Peel():
                annotation.cutting_resistance = CuttingResistance.from_measurement(
                    value=self.skin_cutting_resistance
                )
                annotation.peel_strength = PeelStrength.from_measurement(
                    value=self.peel_strength
                )
            case Flesh():
                annotation.cutting_resistance = CuttingResistance.from_measurement(
                    value=self.flesh_cutting_resistance
                )
                annotation.scooping_resistance = ScoopingResistance.from_measurement(
                    value=self.scooping_resistance
                )
                annotation.bruising_pressure = BruisingPressure.from_measurement(
                    value=self.bruising_pressure
                )
                annotation.material_elasticity = MaterialElasticity(
                    youngs_modulus=singleton(self.elastic_modulus).simple_sets[0],
                    poissons_ratio=singleton(self.texture.poisson_ratio).simple_sets[0],
                )
            case Pit():
                annotation.release_torque = ReleaseTorque.from_measurement(
                    value=self.release_torque
                )

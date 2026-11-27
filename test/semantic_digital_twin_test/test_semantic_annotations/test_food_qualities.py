"""
Tests for how food gives way: the firmness of a fruit, how the material of its parts
resists tools and bruises, whether a knife can cut it, and pieces that stick together.
"""

from __future__ import annotations

# %% Imports

import pytest
from random_events.interval import Bound, SimpleInterval

from semantic_digital_twin.exceptions import NoFieldAcceptsEntity
from semantic_digital_twin.semantic_annotations.mixins import IsCuttable
from semantic_digital_twin.semantic_annotations.qualities.exceptions import (
    QualityOutsideScale,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    BruisingPressure,
    CuttingResistance,
    Firmness,
    MassDensity,
    PeelStrength,
    PhysicalQuantity,
    Quality,
    ReleaseTorque,
    Ripeness,
    RipenessStage,
    ScoopingResistance,
    WorstBruisePressure,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Avocado,
    AvocadoPiece,
    Flesh,
    FruitPart,
    Peel,
    Pit,
)
from semantic_digital_twin.world import World

# %% Fixtures


@pytest.fixture
def world() -> World:
    """
    An empty world with a root body to spawn fruits and their parts into.
    """
    return World.create_with_root_body("root")


def point(value: float) -> SimpleInterval:
    """
    :param value: A measured value.
    :return: The interval holding only ``value``.
    """
    return SimpleInterval.from_data(
        lower=value, upper=value, left=Bound.CLOSED, right=Bound.CLOSED
    )


def by_class(qualities: list[Quality]) -> list[Quality]:
    """
    :param qualities: Qualities of an annotation.
    :return: The qualities, ordered by the name of their class.
    """
    return sorted(qualities, key=lambda quality: type(quality).__name__)


# %% Physical quantities

physical_quantity_types = [
    MassDensity,
    Firmness,
    CuttingResistance,
    ScoopingResistance,
    PeelStrength,
    ReleaseTorque,
    BruisingPressure,
    WorstBruisePressure,
]


@pytest.mark.parametrize("quantity_type", physical_quantity_types)
def test_quantity_at_the_lower_end_of_its_scale_is_rejected(
    quantity_type: type[PhysicalQuantity],
) -> None:
    """
    A quantity of zero, the open lower end of every physical quantity's scale, is
    refused.
    """
    with pytest.raises(QualityOutsideScale):
        quantity_type.from_measurement(value=quantity_type.scale.lower)


def test_measured_quantity_holds_only_the_measured_value() -> None:
    """
    A quantity measured at a value lies at that single point of its scale.
    """
    assert Firmness.from_measurement(value=12.5).value == point(12.5)


# %% Stages of ripeness


def test_stage_holds_a_value_only_if_the_whole_value_lies_in_it() -> None:
    """
    A stage holds its own interval and the point it begins at, but neither the point the
    next stage begins at nor a value reaching into the next stage.
    """
    ripe, overripe = RipenessStage.RIPE, RipenessStage.OVERRIPE
    reaching_into_overripe = SimpleInterval.from_data(
        lower=ripe.value.lower,
        upper=overripe.value.upper,
        left=Bound.CLOSED,
        right=Bound.CLOSED,
    )
    assert ripe.holds(value=ripe.value)
    assert ripe.holds(value=point(ripe.value.lower))
    assert not ripe.holds(value=point(overripe.value.lower))
    assert not ripe.holds(value=reaching_into_overripe)


# %% Qualities of fruits and their parts


@pytest.mark.parametrize("fruit_type", [Avocado, AvocadoPiece])
def test_avocado_and_its_pieces_take_a_ripeness_and_a_firmness(
    world: World, fruit_type: type[Avocado | AvocadoPiece]
) -> None:
    """
    An avocado, and a piece cut from one, takes how far it has ripened and how firm it
    is as its qualities.
    """
    qualities = [
        Ripeness.from_stage(stage=RipenessStage.RIPE),
        Firmness.from_measurement(value=12.5),
    ]
    with world.modify_world():
        fruit = fruit_type.create_with_new_body_in_world(name="fruit", world=world)
        for quality in qualities:
            fruit.set_quality(quality=quality)

    assert by_class(fruit.qualities) == by_class(qualities)


@pytest.mark.parametrize(
    "part_type, qualities",
    [
        (
            Peel,
            [
                CuttingResistance.from_measurement(value=1500.0),
                PeelStrength.from_measurement(value=400.0),
            ],
        ),
        (
            Flesh,
            [
                CuttingResistance.from_measurement(value=90.0),
                ScoopingResistance.from_measurement(value=1.0e4),
                BruisingPressure.from_measurement(value=1.0e5),
                WorstBruisePressure.from_measurement(value=2.0e5),
            ],
        ),
        (Pit, [ReleaseTorque.from_measurement(value=0.02)]),
    ],
)
def test_fruit_part_takes_the_qualities_of_how_it_gives_way(
    world: World, part_type: type[FruitPart], qualities: list[Quality]
) -> None:
    """
    Each part of a fruit takes how its material resists the tools that work it, and the
    flesh also how it bruises, as its qualities.
    """
    with world.modify_world():
        part = part_type.create_with_new_body_in_world(name="part", world=world)
        for quality in qualities:
            part.set_quality(quality=quality)

    assert by_class(part.qualities) == by_class(qualities)


def test_pit_takes_no_cutting_resistance(world: World) -> None:
    """
    A pit refuses a cutting resistance, as no knife cuts it.
    """
    with world.modify_world():
        pit = Pit.create_with_new_body_in_world(name="pit", world=world)
        with pytest.raises(NoFieldAcceptsEntity):
            pit.set_quality(quality=CuttingResistance.from_measurement(value=1.0e4))


# %% Cutting and sticking


@pytest.mark.parametrize(
    "part_type, cuttable", [(Peel, True), (Flesh, True), (Pit, False)]
)
def test_knife_cuts_peel_and_flesh_but_not_the_pit(
    part_type: type[FruitPart], cuttable: bool
) -> None:
    """
    A knife can cut the peel and the flesh of a fruit, but not its pit.
    """
    assert issubclass(part_type, IsCuttable) == cuttable


def test_piece_sticks_to_another_until_it_comes_off(world: World) -> None:
    """
    A piece of avocado that comes to stick to another sticks to it, and no longer once
    it has come off.
    """
    with world.modify_world():
        half, other_half = (
            AvocadoPiece.create_with_new_body_in_world(name=name, world=world)
            for name in ("half", "other_half")
        )
        half.stick_to(other=other_half)
    assert half.adheres_to == [other_half]

    with world.modify_world():
        half.come_off(other=other_half)
    assert half.adheres_to == []

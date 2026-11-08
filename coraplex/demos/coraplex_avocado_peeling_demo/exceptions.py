from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% the fruit model


@dataclass
class FruitModelNotFound(DataclassException):
    """
    Raised when the folder of a peelable fruit model lacks its MuJoCo files.
    """

    model_directory: Path
    """
    The folder that was searched.
    """

    def error_message(self) -> str:
        return f"No peelable MuJoCo fruit model in {self.model_directory}."

    def suggest_correction(self) -> str:
        return (
            "Pass the avocado folder of a private_meshes checkout "
            "(objects/produce/avocado), which holds model.xml and mujoco_peel.json."
        )


@dataclass
class FruitPartNotFound(DataclassException):
    """
    Raised when a part named in the peeling description is missing from the MuJoCo
    model.
    """

    part_name: str
    """
    The full MuJoCo name of the missing part, prefix included.
    """

    def error_message(self) -> str:
        return f"The MuJoCo model has no part named {self.part_name}."

    def suggest_correction(self) -> str:
        return "Use the prefix the fruit was attached with, and the description generated with its model."


# %% the simulation


@dataclass
class UnstableSimulation(DataclassException):
    """
    Raised when MuJoCo reports diverging accelerations.
    """

    simulation_time: float
    """
    The simulated time in seconds at which the instability was detected.
    """

    def error_message(self) -> str:
        return f"The simulation became unstable at {self.simulation_time:.3f} s."

    def suggest_correction(self) -> str:
        return "Lower the time step or check that grip constraints start at the gripped pose."


# %% the demonstration outcome


@dataclass
class SkinTornAtRest(DataclassException):
    """
    Raised when skin tiles come off while nothing pulls on the fruit.
    """

    torn_tiles: list[str]
    """
    The names of the tiles that came off.
    """

    def error_message(self) -> str:
        return f"{len(self.torn_tiles)} skin tiles came off a resting fruit: {self.torn_tiles}."

    def suggest_correction(self) -> str:
        return "Increase the release distance or stiffen the skin constraints of the model."


@dataclass
class StripNotPeeled(DataclassException):
    """
    Raised when pulling a strip leaves too much of it on the fruit.
    """

    strip: str
    """
    The name of the pulled strip.
    """

    peeled_tile_count: int
    """
    How many of the strip's tiles came off.
    """

    strip_tile_count: int
    """
    How many tiles the strip has.
    """

    def error_message(self) -> str:
        return (
            f"Only {self.peeled_tile_count} of the {self.strip_tile_count} tiles of strip "
            f"{self.strip} came off."
        )

    def suggest_correction(self) -> str:
        return "Pull further along the fruit, or lower the release distances."


@dataclass
class NeighbouringStripsTorn(DataclassException):
    """
    Raised when skin comes off that was not pulled, while peeling a strip or cutting the
    fruit.
    """

    torn_tiles: list[str]
    """
    The names of the tiles of other strips that came off.
    """

    def error_message(self) -> str:
        return f"{len(self.torn_tiles)} skin tiles came off that were not pulled: {self.torn_tiles}."

    def suggest_correction(self) -> str:
        return "Increase the release distances."


@dataclass
class PitStuck(DataclassException):
    """
    Raised when the freed pit does not leave its cavity.
    """

    displacement: float
    """
    How far in metres the pit moved relative to its flesh half.
    """

    def error_message(self) -> str:
        return (
            f"The pit moved only {self.displacement * 1000:.0f} mm out of its cavity."
        )

    def suggest_correction(self) -> str:
        return "Lift the covering flesh half away before pushing the pit out."


@dataclass
class SliceStuck(DataclassException):
    """
    Raised when a cut-off flesh slice does not come away on its own.
    """

    slice_displacement: float
    """
    How far in metres the slice that was carried away moved.
    """

    neighbour_displacement: float
    """
    How far in metres its neighbouring slice moved with it.
    """

    def error_message(self) -> str:
        return (
            f"The cut-off slice moved {self.slice_displacement * 1000:.0f} mm and took "
            f"its neighbour {self.neighbour_displacement * 1000:.0f} mm along."
        )

    def suggest_correction(self) -> str:
        return "Draw the slice off along the fruit's axis before lifting it."


@dataclass
class FruitNotPlaced(DataclassException):
    """
    Raised when the fruit set down on the cutting board does not come to rest there.
    """

    distance: float
    """
    How far in metres from the board's centre the fruit came to rest.
    """

    def error_message(self) -> str:
        return (
            f"The fruit came to rest {self.distance * 1000:.0f} mm from the cutting "
            f"board's centre."
        )

    def suggest_correction(self) -> str:
        return "Lower the fruit closer to the board before letting go of it."


@dataclass
class ArmsCollided(DataclassException):
    """
    Raised when the robot's two arms touch each other.
    """

    left_body: str
    """
    The name of the body of the left arm that was touched.
    """

    right_body: str
    """
    The name of the body of the right arm that touched it.
    """

    time: float
    """
    The simulation time in seconds of the contact.
    """

    def error_message(self) -> str:
        return (
            f"The arms touched at {self.time:.2f} s: {self.left_body} and "
            f"{self.right_body}."
        )

    def suggest_correction(self) -> str:
        return "Move one hand out of the other's way before the other one reaches in."


@dataclass
class PeelNotPlaced(DataclassException):
    """
    Raised when the peeled strip does not end up where it was laid down.
    """

    distance: float
    """
    How far in metres the farthest piece of the strip came to rest from where it was
    laid down.
    """

    def error_message(self) -> str:
        return (
            f"A piece of the peeled strip came to rest {self.distance * 1000:.0f} mm "
            f"from where it was laid down."
        )

    def suggest_correction(self) -> str:
        return "Lower the strip closer to the surface before letting go of it."


@dataclass
class UnsupportedFruitModel(DataclassException):
    """
    Raised when a fruit model's peeling description is of a format this demo cannot
    read.
    """

    source: str
    """
    The peeling description that was read.
    """

    found_format: int | None
    """
    The format it declares, if any.
    """

    def error_message(self) -> str:
        return (
            f"{self.source} is of format {self.found_format}, which this demo cannot "
            f"read."
        )

    def suggest_correction(self) -> str:
        return "Regenerate the fruit with the current generator of private_meshes."

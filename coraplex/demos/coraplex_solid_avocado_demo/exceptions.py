"""
The errors the avocado demos raise when the fruit or the simulation is not in the state
a demo needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% the fruit model


@dataclass
class FruitPartNotFound(DataclassException):
    """
    Raised when a part of the fruit is missing from the MuJoCo model.
    """

    part_name: str
    """
    The full MuJoCo name of the missing part, prefix included.
    """

    def error_message(self) -> str:
        return f"The MuJoCo model has no part named {self.part_name}."

    def suggest_correction(self) -> str:
        return "Name the fruit's body and its parts as the fruit's model does, with the prefix it was attached with."


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
        return "Lower the time step, or move tools and pieces in smaller steps."


# %% cutting


@dataclass
class FruitPartNotSolid(DataclassException):
    """
    Raised when the mesh of a part of a cuttable fruit does not enclose a volume.
    """

    vertex_count: int
    """
    How many vertices the mesh has.
    """

    face_count: int
    """
    How many faces the mesh has.
    """

    def error_message(self) -> str:
        return (
            f"A mesh of {self.vertex_count} vertices and {self.face_count} faces does "
            f"not enclose a volume, so it cannot be cut."
        )

    def suggest_correction(self) -> str:
        return "Give every part of the fruit a closed mesh without holes or self-intersections."


@dataclass
class FruitNotCut(DataclassException):
    """
    Raised when a stroke of the knife leaves the fruit in another number of pieces than
    expected.
    """

    expected_pieces: int
    """
    How many pieces the fruit should be in after the stroke.
    """

    actual_pieces: int
    """
    How many pieces it is in.
    """

    def error_message(self) -> str:
        return (
            f"The fruit is in {self.actual_pieces} pieces after the stroke, not in "
            f"{self.expected_pieces}."
        )

    def suggest_correction(self) -> str:
        return (
            "Lead the blade's cutting edge all the way through the fruit, with an edge "
            "longer than the fruit is wide where it is cut."
        )


@dataclass
class CutOffKnifePlane(DataclassException):
    """
    Raised when a cut surface does not lie in the plane of the blade that made it.
    """

    piece_name: str
    """
    The name of the body of the piece with that surface.
    """

    deviation: float
    """
    How far the surface lies from the plane, in metres.
    """

    def error_message(self) -> str:
        return (
            f"The cut surface of {self.piece_name} lies {self.deviation * 1000:.3f} mm "
            f"from the plane of the blade."
        )

    def suggest_correction(self) -> str:
        return "Cut the fruit along the plane the blade has when it has gone through."


@dataclass
class PiecesScattered(DataclassException):
    """
    Raised when a piece of the fruit is thrown away instead of coming to rest.
    """

    piece_name: str
    """
    The name of the body of the piece that moved farthest.
    """

    distance: float
    """
    How far it moved, in metres.
    """

    def error_message(self) -> str:
        return f"{self.piece_name} moved {self.distance * 1000:.0f} mm while coming to rest."

    def suggest_correction(self) -> str:
        return (
            "Check that the convex collision shapes of touching pieces do not overlap; "
            "raise the fruit's refinements if they do."
        )


@dataclass
class PiecesNotHinged(DataclassException):
    """
    Raised when two pieces of a fruit taken to be joined by a hinge are not.
    """

    first_piece: str
    """
    The name of the body of one piece.
    """

    second_piece: str
    """
    The name of the body of the other.
    """

    def error_message(self) -> str:
        return (
            f"No hinge joins {self.first_piece} to {self.second_piece}; the first "
            f"piece holds the hinge to the one after it."
        )

    def suggest_correction(self) -> str:
        return "Pass neighbouring segments made by one hinge cut, in their order."


@dataclass
class PitDoesNotFit(DataclassException):
    """
    Raised when no random pit tried fits into the avocado's flesh.
    """

    seed: int
    """
    The seed of the random choice.
    """

    attempts: int
    """
    How many random pits were tried.
    """

    def error_message(self) -> str:
        return (
            f"None of {self.attempts} random pits from seed {self.seed} fits into the "
            f"flesh with the margin it needs."
        )

    def suggest_correction(self) -> str:
        return "Narrow the pit's size range and shifts, or lower the flesh margin."

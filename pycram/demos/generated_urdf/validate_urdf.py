#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_urdf.py
================

Validation script for URDF files generated with the architecture/room URDF
generation prompt.

The script checks a URDF file (or a Markdown/text file that contains exactly
one ```xml ... ``` code block) against the requirements defined in the prompt
and outputs a structured report.

Usage
------
    python3 validate_urdf.py room.urdf
    python3 validate_urdf.py response.md            # extracts the xml block
    python3 validate_urdf.py room.urdf --strict     # warnings = errors
    python3 validate_urdf.py room.urdf --json       # machine-readable output
    python3 validate_urdf.py room.urdf --no-collision-check
    python3 validate_urdf.py room.urdf --check-output-format   # zero-prose rule

Exit Code: 0 = passed, 1 = errors found (in --strict mode also on warnings).

Notes
--------
* Pure standard library, no external dependencies.
* Structural checks (XML, kinematic tree, Visual/Collision/Material,
  Joints) are hard requirements -> Severity ERROR.
* Checks based on naming heuristics or standard proportions (heights, door counts,
  "every cabinet needs doors", corner cabinet single-entity, collisions) are marked
  as WARNING/INFO, because they cannot be unambiguously decided without the original
  floor plan or guaranteed naming conventions.
* Multiple <visual>/<collision> tags per link are supported (L-shaped corner cabinets).
  Root links (the floor) do not need a <collision>.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Severity / Result Model
# ---------------------------------------------------------------------------

ERROR = "ERROR"
WARN = "WARN"
INFO = "INFO"
PASS = "PASS"

_SEVERITY_ORDER = {ERROR: 0, WARN: 1, INFO: 2, PASS: 3}
_SEVERITY_ICON = {ERROR: "x", WARN: "!", INFO: "i", PASS: "+"}


@dataclass
class Result:
    check_id: str
    category: str
    severity: str
    message: str
    target: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "target": self.target,
        }


# ---------------------------------------------------------------------------
# Data Model (parsed from URDF)
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    kind: str  # "box" | "cylinder" | "sphere" | "mesh" | "unknown"
    size: Optional[List[float]] = None  # box: [x, y, z]
    radius: Optional[float] = None
    length: Optional[float] = None
    filename: Optional[str] = None  # mesh

    def signature(self) -> tuple:
        """Comparable signature (rounded) for equality checks."""

        def r(v):
            return None if v is None else round(v, 6)

        sz = None if self.size is None else tuple(r(v) for v in self.size)
        return (self.kind, sz, r(self.radius), r(self.length), self.filename)

    def half_extents(self) -> Optional[List[float]]:
        """Half extents as box approximation (for AABB)."""
        if self.kind == "box" and self.size:
            return [self.size[0] / 2.0, self.size[1] / 2.0, self.size[2] / 2.0]
        if (
            self.kind == "cylinder"
            and self.radius is not None
            and self.length is not None
        ):
            return [self.radius, self.radius, self.length / 2.0]
        if self.kind == "sphere" and self.radius is not None:
            return [self.radius, self.radius, self.radius]
        return None


@dataclass
class Origin:
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class Visual:
    origin: Origin
    geometry: Optional[Geometry]
    material_name: Optional[str]
    material_rgba: Optional[List[float]]


@dataclass
class Collision:
    origin: Origin
    geometry: Optional[Geometry]


@dataclass
class Link:
    name: str
    visuals: List[Visual] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)
    has_inertial: bool = False


@dataclass
class Limit:
    lower: Optional[float] = None
    upper: Optional[float] = None
    effort: Optional[float] = None
    velocity: Optional[float] = None


@dataclass
class Joint:
    name: str
    jtype: str
    parent: Optional[str]
    child: Optional[str]
    origin: Origin
    axis: Optional[List[float]] = None
    limit: Optional[Limit] = None


@dataclass
class UrdfModel:
    robot_name: Optional[str]
    links: Dict[str, Link]
    link_order: List[str]
    joints: List[Joint]
    global_materials: Dict[str, List[float]]  # name -> rgba


# ---------------------------------------------------------------------------
# XML Helper Functions (namespace-tolerant)
# ---------------------------------------------------------------------------


def _lname(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _children(el: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in list(el) if _lname(c) == name]


def _child(el: ET.Element, name: str) -> Optional[ET.Element]:
    found = _children(el, name)
    return found[0] if found else None


def _parse_floats(
    text: Optional[str], expected: Optional[int] = None
) -> Optional[List[float]]:
    if text is None:
        return None
    parts = text.replace(",", " ").split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if expected is not None and len(vals) != expected:
        return None
    return vals


# ---------------------------------------------------------------------------
# Parsing: XML -> UrdfModel
# ---------------------------------------------------------------------------


def parse_origin(el: Optional[ET.Element]) -> Origin:
    if el is None:
        return Origin()
    xyz = _parse_floats(el.get("xyz"), 3) or [0.0, 0.0, 0.0]
    rpy = _parse_floats(el.get("rpy"), 3) or [0.0, 0.0, 0.0]
    return Origin(xyz=xyz, rpy=rpy)


def parse_geometry(el: Optional[ET.Element]) -> Optional[Geometry]:
    if el is None:
        return None
    box = _child(el, "box")
    if box is not None:
        return Geometry(kind="box", size=_parse_floats(box.get("size"), 3))
    cyl = _child(el, "cylinder")
    if cyl is not None:
        r = _parse_floats(cyl.get("radius"), 1)
        l = _parse_floats(cyl.get("length"), 1)
        return Geometry(
            kind="cylinder", radius=(r[0] if r else None), length=(l[0] if l else None)
        )
    sph = _child(el, "sphere")
    if sph is not None:
        r = _parse_floats(sph.get("radius"), 1)
        return Geometry(kind="sphere", radius=(r[0] if r else None))
    mesh = _child(el, "mesh")
    if mesh is not None:
        return Geometry(kind="mesh", filename=mesh.get("filename"))
    return Geometry(kind="unknown")


def parse_material(
    el: Optional[ET.Element],
) -> Tuple[Optional[str], Optional[List[float]]]:
    if el is None:
        return None, None
    name = el.get("name")
    color = _child(el, "color")
    rgba = _parse_floats(color.get("rgba"), 4) if color is not None else None
    return name, rgba


def parse_link(el: ET.Element) -> Link:
    link = Link(name=el.get("name", ""))
    link.has_inertial = _child(el, "inertial") is not None
    for v in _children(el, "visual"):
        mname, mrgba = parse_material(_child(v, "material"))
        link.visuals.append(
            Visual(
                origin=parse_origin(_child(v, "origin")),
                geometry=parse_geometry(_child(v, "geometry")),
                material_name=mname,
                material_rgba=mrgba,
            )
        )
    for c in _children(el, "collision"):
        link.collisions.append(
            Collision(
                origin=parse_origin(_child(c, "origin")),
                geometry=parse_geometry(_child(c, "geometry")),
            )
        )
    return link


def parse_joint(el: ET.Element) -> Joint:
    parent_el = _child(el, "parent")
    child_el = _child(el, "child")
    axis_el = _child(el, "axis")
    limit_el = _child(el, "limit")
    axis = _parse_floats(axis_el.get("xyz"), 3) if axis_el is not None else None
    limit = None
    if limit_el is not None:

        def fget(attr):
            v = _parse_floats(limit_el.get(attr), 1)
            return v[0] if v else None

        limit = Limit(
            lower=fget("lower"),
            upper=fget("upper"),
            effort=fget("effort"),
            velocity=fget("velocity"),
        )
    return Joint(
        name=el.get("name", ""),
        jtype=el.get("type", ""),
        parent=parent_el.get("link") if parent_el is not None else None,
        child=child_el.get("link") if child_el is not None else None,
        origin=parse_origin(_child(el, "origin")),
        axis=axis,
        limit=limit,
    )


def build_model(root: ET.Element) -> UrdfModel:
    links: Dict[str, Link] = {}
    link_order: List[str] = []
    joints: List[Joint] = []
    global_materials: Dict[str, List[float]] = {}

    for el in _children(root, "material"):
        name, rgba = parse_material(el)
        if name and rgba is not None:
            global_materials[name] = rgba

    for el in _children(root, "link"):
        link = parse_link(el)
        links[link.name] = link
        link_order.append(link.name)

    for el in _children(root, "joint"):
        joints.append(parse_joint(el))

    return UrdfModel(
        robot_name=root.get("name"),
        links=links,
        link_order=link_order,
        joints=joints,
        global_materials=global_materials,
    )


# ---------------------------------------------------------------------------
# Geometry / Transforms (pure Python matrix math)
# ---------------------------------------------------------------------------


def _rpy_to_matrix(r: float, p: float, y: float) -> List[List[float]]:
    """URDF-rpy: extrinsic roll(x), pitch(y), yaw(z) -> R = Rz*Ry*Rx."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # Rz * Ry * Rx
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_vec(m: List[List[float]], v: List[float]) -> List[float]:
    return [m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3)]


def _mat_mat(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


Transform = Tuple[List[List[float]], List[float]]  # (R 3x3, t 3-vec)

_IDENTITY: Transform = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0])


def _origin_to_transform(o: Origin) -> Transform:
    return (_rpy_to_matrix(o.rpy[0], o.rpy[1], o.rpy[2]), list(o.xyz))


def _compose(a: Transform, b: Transform) -> Transform:
    """a then b: implies Transform from a-Frame to b-Frame child coordinates."""
    R = _mat_mat(a[0], b[0])
    t = [a[1][i] + _mat_vec(a[0], b[1])[i] for i in range(3)]
    return (R, t)


def _aabb_of_geometry(
    geom: Geometry, link_world: Transform, visual_origin: Origin
) -> Optional[Tuple[List[float], List[float]]]:
    """World AABB (min, max) of a geometry as box approximation."""
    he = geom.half_extents()
    if he is None:
        return None
    geom_world = _compose(link_world, _origin_to_transform(visual_origin))
    R, t = geom_world
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = [sx * he[0], sy * he[1], sz * he[2]]
                world = [t[i] + _mat_vec(R, local)[i] for i in range(3)]
                corners.append(world)
    mins = [min(c[i] for c in corners) for i in range(3)]
    maxs = [max(c[i] for c in corners) for i in range(3)]
    return mins, maxs


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

DOOR_KEYWORDS = ("door", "tuer", "tür", "tur", "flap", "klappe")
# Door subparts that are allowed to be fixed and DO NOT need
# their own rotation axis (handles, knobs, panels, glass inserts).
HANDLE_KEYWORDS = (
    "handle",
    "griff",
    "knob",
    "knauf",
    "pull",
    "grip",
    "panel",
    "insert",
)
WALL_KEYWORDS = ("wall", "wand")
COUNTER_KEYWORDS = ("counter", "countertop", "worktop", "arbeitsplatte")
BASE_CAB_KEYWORDS = ("cabinet", "cupboard", "unterschrank", "korpus")
WALL_CAB_KEYWORDS = (
    "wall_cab",
    "upper",
    "oberschrank",
    "haengeschrank",
    "hängeschrank",
    "wandschrank",
)
TABLE_KEYWORDS = ("table", "tisch", "desk", "schreibtisch")

# Links that MUST have movable doors according to the prompt (cabinets/cupboards).
# Intentionally specific so that e.g. "fridge" (appliance) DOES NOT match.
CABINET_KEYWORDS = (
    "cabinet",
    "cupboard",
    "unterschrank",
    "oberschrank",
    "hängeschrank",
    "haengeschrank",
    "wandschrank",
    "eckschrank",
    "hochschrank",
    "kleiderschrank",
    "wardrobe",
    "korpus",
)
# Corner cabinets: must be modeled as ONE link with multiple geometries (L-shape).
CORNER_KEYWORDS = (
    "corner",
    "eckschrank",
    "eck_",
    "_eck",
    "lshape",
    "l_shape",
    "l-shape",
)
# Appliances: do not receive a forced swing door requirement (exempt from CABINET check).
APPLIANCE_KEYWORDS = (
    "fridge",
    "refrigerator",
    "kuehlschrank",
    "kühlschrank",
    "freezer",
    "gefrierschrank",
    "oven",
    "ofen",
    "backofen",
    "herd",
    "stove",
    "cooktop",
    "kochfeld",
    "dishwasher",
    "spuelmaschine",
    "spülmaschine",
    "geschirrspueler",
    "geschirrspüler",
    "washer",
    "waschmaschine",
    "dryer",
    "trockner",
    "microwave",
    "mikrowelle",
    "hood",
    "dunstabzug",
)

PENETRATION_THRESHOLD = 0.04  # m, from this penetration in all 3 axes on -> collision
MAX_REASONABLE_DIM = 50.0  # m, larger individual dimensions -> presumably unit error


def _name_matches(name: str, keywords) -> bool:
    n = name.lower()
    return any(k in n for k in keywords)


class UrdfValidator:
    def __init__(self, model: UrdfModel, check_collisions: bool = True):
        self.m = model
        self.check_collisions_enabled = check_collisions
        self.results: List[Result] = []
        # Derived structures
        self.child_to_joint: Dict[str, Joint] = {}
        self.parent_to_joints: Dict[str, List[Joint]] = {}
        for j in model.joints:
            if j.child:
                self.child_to_joint[j.child] = j
            if j.parent:
                self.parent_to_joints.setdefault(j.parent, []).append(j)

    # -- Result Helpers ----------------------------------------------------

    def add(self, severity, check_id, category, message, target=None):
        self.results.append(Result(check_id, category, severity, message, target))

    def _passed(self, check_id, category, message):
        # Only add PASS if no problems have been recorded for this check_id.
        if not any(
            r.check_id == check_id and r.severity in (ERROR, WARN) for r in self.results
        ):
            self.add(PASS, check_id, category, message)

    # -- Material Resolution ------------------------------------------------

    def _link_color(self, link: Link) -> Optional[List[float]]:
        for v in link.visuals:
            if v.material_rgba is not None:
                return v.material_rgba
            if v.material_name and v.material_name in self.m.global_materials:
                return self.m.global_materials[v.material_name]
        return None

    def _has_material(self, link: Link) -> bool:
        for v in link.visuals:
            if v.material_name or v.material_rgba is not None:
                return True
        return False

    # -- World Transforms ----------------------------------------------

    def _world_transforms(self) -> Dict[str, Transform]:
        """BFS from base_link; unreachable links are omitted."""
        transforms: Dict[str, Transform] = {}
        root = (
            "base_link"
            if "base_link" in self.m.links
            else (self._find_roots()[0] if self._find_roots() else None)
        )
        if root is None:
            return transforms
        transforms[root] = _IDENTITY
        stack = [root]
        visited = {root}
        while stack:
            parent = stack.pop()
            for j in self.parent_to_joints.get(parent, []):
                if not j.child or j.child in visited:
                    continue
                transforms[j.child] = _compose(
                    transforms[parent], _origin_to_transform(j.origin)
                )
                visited.add(j.child)
                stack.append(j.child)
        return transforms

    def _find_roots(self) -> List[str]:
        has_parent = {j.child for j in self.m.joints if j.child}
        return [n for n in self.m.link_order if n not in has_parent]

    def _top_ancestor(self, link: str) -> str:
        """Topmost object directly under base_link (for grouping)."""
        cur = link
        seen = set()
        while cur not in seen:
            seen.add(cur)
            j = self.child_to_joint.get(cur)
            if j is None or j.parent is None:
                return cur
            if j.parent == "base_link":
                return cur
            cur = j.parent
        return cur

    # =======================================================================
    # Checks
    # =======================================================================

    def run(self) -> List[Result]:
        self.check_root()
        self.check_unique_names()
        self.check_tree()
        self.check_base_link()
        self.check_links()
        self.check_joints()
        self.check_doors()
        self.check_corner_cabinets()
        self.check_values_units()
        self.check_proportions()
        if self.check_collisions_enabled:
            self.check_overlaps()
        return self.results

    # --- A: Basic Structure --------------------------------------------------

    def check_root(self):
        cat = "A Basic Structure"
        if self.m.robot_name is None:
            self.add(WARN, "A3", cat, "<robot> element has no 'name' attribute.")
        if not self.m.links:
            self.add(ERROR, "A4", cat, "No <link> elements found.")
        if not self.m.joints:
            self.add(ERROR, "A4", cat, "No <joint> elements found.")
        self._passed("A4", cat, "At least one link and one joint present.")
        self._passed("A3", cat, "<robot> element present.")

    # --- B: Kinematic Tree --------------------------------------------

    def check_unique_names(self):
        cat = "B Kinematic Tree"
        seen = {}
        for n in self.m.link_order:
            seen[n] = seen.get(n, 0) + 1
        for n, c in seen.items():
            if c > 1:
                self.add(
                    ERROR, "B6", cat, f"Link name assigned multiple times ({c}x).", n
                )
        self._passed("B6", cat, "All link names are unique.")

        jseen = {}
        for j in self.m.joints:
            jseen[j.name] = jseen.get(j.name, 0) + 1
        for n, c in jseen.items():
            if c > 1:
                self.add(
                    ERROR, "B7", cat, f"Joint name assigned multiple times ({c}x).", n
                )
        self._passed("B7", cat, "All joint names are unique.")

    def check_tree(self):
        cat = "B Kinematic Tree"

        # B3: parent/child point to existing links
        for j in self.m.joints:
            if j.parent is None:
                self.add(ERROR, "B3", cat, "Joint missing <parent>.", j.name)
            elif j.parent not in self.m.links:
                self.add(
                    ERROR,
                    "B3",
                    cat,
                    f"Joint parent '{j.parent}' is not an existing link.",
                    j.name,
                )
            if j.child is None:
                self.add(ERROR, "B3", cat, "Joint missing <child>.", j.name)
            elif j.child not in self.m.links:
                self.add(
                    ERROR,
                    "B3",
                    cat,
                    f"Joint child '{j.child}' is not an existing link.",
                    j.name,
                )
        self._passed("B3", cat, "All joint parent/child point to existing links.")

        # B2/B1: exactly one Root, named base_link
        roots = self._find_roots()
        if len(roots) == 0:
            self.add(ERROR, "B2", cat, "No root link found (likely a cycle).")
        elif len(roots) > 1:
            self.add(
                ERROR,
                "B2",
                cat,
                "Multiple root links (tree disconnected): " + ", ".join(roots),
            )
        else:
            self._passed("B2", cat, "Exactly one root link.")
        if "base_link" not in self.m.links:
            self.add(ERROR, "B1", cat, "Root link 'base_link' missing.")
        elif roots and roots[0] != "base_link":
            self.add(
                ERROR, "B1", cat, f"Root link is '{roots[0]}', expected 'base_link'."
            )
        else:
            self._passed("B1", cat, "Root link is named 'base_link'.")

        # B4: no cycles, B5: all reachable
        child_to_parent = {j.child: j.parent for j in self.m.joints if j.child}
        for start in self.m.link_order:
            seen = set()
            cur = start
            while cur in child_to_parent:
                if cur in seen:
                    self.add(
                        ERROR, "B4", cat, "Cycle detected in kinematic tree.", start
                    )
                    break
                seen.add(cur)
                cur = child_to_parent[cur]
        self._passed("B4", cat, "No cycles in the tree.")

        transforms = self._world_transforms()
        unreachable = [n for n in self.m.link_order if n not in transforms]
        for n in unreachable:
            self.add(ERROR, "B5", cat, "Link is not reachable from base_link.", n)
        self._passed("B5", cat, "All links are reachable from base_link.")

    # --- C: base_link / Floor ---------------------------------------------

    def check_base_link(self):
        cat = "C base_link / Floor"
        bl = self.m.links.get("base_link")
        if bl is None:
            return
        if not bl.visuals or bl.visuals[0].geometry is None:
            self.add(WARN, "C1", cat, "base_link has no visual geometry.")
            return
        geom = bl.visuals[0].geometry
        if geom.kind != "box":
            self.add(
                WARN, "C1", cat, f"base_link should be a box, but is '{geom.kind}'."
            )
        else:
            self._passed("C1", cat, "base_link is a box.")
        # Thin + at z ~ 0.01
        if geom.size:
            thickness = geom.size[2]
            z = bl.visuals[0].origin.xyz[2]
            if thickness > 0.1:
                self.add(
                    WARN,
                    "C2",
                    cat,
                    f"Floor appears thick (z-height {thickness:.3f} m), expected thin (~0.02 m).",
                )
            if abs(z - 0.01) > 0.05:
                self.add(
                    WARN, "C2", cat, f"Floor center at z={z:.3f} m, expected ~0.01 m."
                )
            self._passed("C2", cat, "Floor is thin and positioned at z ~ 0.01.")

    # --- D: Links Visual / Collision / Material -------------------------------

    def check_links(self):
        cat = "D Visual / Collision / Material"
        # Root links (without parent joint) need no collision: floor is
        # usually just visible, collision is often handled by ground plane.
        root_links = {n for n in self.m.link_order if n not in self.child_to_joint}
        for name in self.m.link_order:
            link = self.m.links[name]

            # D1/D2: Visual & Collision present
            if not link.visuals:
                self.add(ERROR, "D1", cat, "Link missing <visual>.", name)
            if not link.collisions and name not in root_links:
                self.add(ERROR, "D2", cat, "Link missing <collision>.", name)

            # D6: Geometry type (mesh allowed; prompt recommends primitives)
            for v in link.visuals:
                if v.geometry is None:
                    self.add(ERROR, "D1", cat, "Visual missing <geometry>.", name)
                elif v.geometry.kind == "mesh":
                    self.add(
                        INFO,
                        "D6",
                        cat,
                        "Visual uses mesh (prompt recommends primitives, mesh is allowed).",
                        name,
                    )
                elif v.geometry.kind == "unknown":
                    self.add(ERROR, "D6", cat, "Unknown geometry type in visual.", name)
            for c in link.collisions:
                if c.geometry is None:
                    self.add(ERROR, "D2", cat, "Collision missing <geometry>.", name)
                elif c.geometry.kind == "mesh":
                    self.add(
                        INFO,
                        "D6",
                        cat,
                        "Collision uses mesh (prompt recommends primitives, mesh is allowed).",
                        name,
                    )

            # D3/D4: Visual and Collision geometry + origin identical.
            # Multi-set comparison for links with multiple geometries
            # (e.g. L-shaped corner cabinets).
            if link.visuals and link.collisions:
                v_sigs = sorted(
                    (v.geometry.signature() if v.geometry else None)
                    for v in link.visuals
                )
                c_sigs = sorted(
                    (c.geometry.signature() if c.geometry else None)
                    for c in link.collisions
                )
                if v_sigs != c_sigs:
                    self.add(
                        ERROR,
                        "D3",
                        cat,
                        "Visual and Collision geometry do not match "
                        f"({len(link.visuals)} visual / {len(link.collisions)} collision).",
                        name,
                    )
                else:
                    # Geometries match -> additionally compare placement (<origin>).
                    v_full = sorted(
                        _geom_origin_sig(v.geometry, v.origin) for v in link.visuals
                    )
                    c_full = sorted(
                        _geom_origin_sig(c.geometry, c.origin) for c in link.collisions
                    )
                    if v_full != c_full:
                        self.add(
                            WARN,
                            "D4",
                            cat,
                            "Visual and Collision geometry identical, but placement "
                            "(<origin>) differs.",
                            name,
                        )

            # D5: Material present
            if link.visuals and not self._has_material(link):
                self.add(ERROR, "D5", cat, "Link has no <material>.", name)

        self._passed("D1", cat, "Every link has a <visual>.")
        self._passed("D2", cat, "Every non-root link has a <collision>.")
        self._passed("D3", cat, "Visual and collision geometries match.")
        self._passed("D4", cat, "Visual and collision origins match.")
        self._passed("D5", cat, "Every link has a <material>.")
        self._passed("D6", cat, "Geometry types are valid (box/cylinder/sphere/mesh).")

    # --- E: Joints ---------------------------------------------------------

    def check_joints(self):
        cat = "E Joints"
        # fixed = static, revolute = doors, prismatic = drawers/sliding parts.
        allowed = {"fixed", "revolute", "prismatic"}
        # Movable joints that need <axis> and <limit>.
        movable = {"revolute", "prismatic"}
        for j in self.m.joints:
            if not j.jtype:
                self.add(ERROR, "E1", cat, "Joint missing 'type' attribute.", j.name)
                continue
            if j.jtype not in allowed:
                self.add(
                    WARN,
                    "E2",
                    cat,
                    f"Joint type '{j.jtype}' unexpected (allowed: fixed/revolute/prismatic).",
                    j.name,
                )

            if j.jtype in movable:
                # E4: axis (required/sensible for revolute AND prismatic)
                if j.axis is None:
                    default_hint = (
                        "URDF default 1 0 0 is usually wrong for doors"
                        if j.jtype == "revolute"
                        else "Movement direction should be explicitly set"
                    )
                    self.add(
                        WARN,
                        "E4",
                        cat,
                        f"{j.jtype} joint missing <axis> ({default_hint}).",
                        j.name,
                    )
                else:
                    norm = math.sqrt(sum(a * a for a in j.axis))
                    if norm < 1e-9:
                        self.add(
                            ERROR,
                            "E4",
                            cat,
                            f"{j.jtype} <axis> is a zero vector.",
                            j.name,
                        )
                    elif abs(norm - 1.0) > 1e-3:
                        self.add(
                            WARN,
                            "E4",
                            cat,
                            f"{j.jtype} <axis> not normalized (|axis|={norm:.3f}).",
                            j.name,
                        )
                    # Doors typically swing around z; just a hint for revolute.
                    if j.jtype == "revolute" and abs(abs(j.axis[2]) - 1.0) > 1e-3:
                        self.add(
                            INFO,
                            "E4",
                            cat,
                            "Revolute <axis> is not 'z' (0 0 1); unusual for doors.",
                            j.name,
                        )

                # E5: limit (required for revolute AND prismatic)
                if j.limit is None or j.limit.lower is None or j.limit.upper is None:
                    self.add(
                        ERROR,
                        "E5",
                        cat,
                        f"{j.jtype} joint needs <limit> with lower and upper.",
                        j.name,
                    )
                else:
                    lo, up = j.limit.lower, j.limit.upper
                    if lo > up:
                        self.add(
                            ERROR,
                            "E5",
                            cat,
                            f"Limit lower ({lo}) greater than upper ({up}).",
                            j.name,
                        )
                    elif j.jtype == "revolute":
                        # E6: Rotation angle plausibility (~0 to ~1.57 rad)
                        if abs(up) > 6.3:
                            self.add(
                                WARN,
                                "E6",
                                cat,
                                f"upper={up} very large; possibly degrees instead of radians.",
                                j.name,
                            )
                        elif abs(up) > 0 and not (1.3 <= abs(up) <= 1.9):
                            self.add(
                                INFO,
                                "E6",
                                cat,
                                f"Door swing upper={up} rad deviates from ~1.57 (90 degrees).",
                                j.name,
                            )
                    else:
                        # prismatic: Limit is a travel distance in meters -> different plausibility
                        if abs(up) > 3.0:
                            self.add(
                                WARN,
                                "E6",
                                cat,
                                f"prismatic travel upper={up} m very large; possibly unit error.",
                                j.name,
                            )

            # E7: origin (for placement)
            if _close_list(j.origin.xyz, [0, 0, 0]) and _close_list(
                j.origin.rpy, [0, 0, 0]
            ):
                self.add(
                    INFO,
                    "E7",
                    cat,
                    "Joint <origin> is 0/0/0 - verify if this placement is intended.",
                    j.name,
                )

        self._passed("E1", cat, "Every joint has a type.")
        self._passed("E2", cat, "Only fixed/revolute/prismatic joints used.")
        self._passed("E4", cat, "Movable joints have a plausible <axis>.")
        self._passed("E5", cat, "Movable joints have valid <limit> values.")
        self._passed("E6", cat, "Swing/slide paths plausible.")

    # --- F: Doors / Cabinets ------------------------------------------------

    def check_doors(self):
        cat = "F Doors / Cabinets"
        revolute_joints = [j for j in self.m.joints if j.jtype == "revolute"]
        root_links = {n for n in self.m.link_order if n not in self.child_to_joint}

        # F0: Every cabinet/cupboard MUST have movable doors (new, CRITICAL in
        # the prompt). Heuristic via naming patterns; appliances (fridge etc.) and
        # door/handle links themselves are excluded. Drawers (prismatic) also count
        # as a movable opening.
        for name in self.m.link_order:
            if name in root_links:
                continue
            if not _name_matches(name, CABINET_KEYWORDS):
                continue
            if _name_matches(name, DOOR_KEYWORDS) or _name_matches(
                name, HANDLE_KEYWORDS
            ):
                continue  # is a door/handle itself, not a corpus
            if _name_matches(name, APPLIANCE_KEYWORDS):
                continue  # Appliance, no forced swing door
            own_joint = self.child_to_joint.get(name)
            if own_joint is not None and own_joint.jtype in ("revolute", "prismatic"):
                continue  # itself a movable part, not a static corpus
            movable_children = [
                j
                for j in self.parent_to_joints.get(name, [])
                if j.jtype in ("revolute", "prismatic")
            ]
            if not movable_children:
                self.add(
                    WARN,
                    "F0",
                    cat,
                    "Cabinet missing movable doors/drawers - according to the prompt, EVERY "
                    "cabinet/cupboard MUST have movable doors.",
                    name,
                )
        self._passed("F0", cat, "Every identified cabinet has movable doors/drawers.")

        # F1: Door links should be connected via revolute.
        # Excluded: handles/panels and subparts that are firmly attached to an
        # (already rotatable) door - these may be fixed.
        for name in self.m.link_order:
            if not _name_matches(name, DOOR_KEYWORDS):
                continue
            if _name_matches(name, HANDLE_KEYWORDS):
                continue  # Handle/Panel: may be fixed
            j = self.child_to_joint.get(name)
            if j is None:
                self.add(WARN, "F1", cat, "Link named as door has no joint.", name)
                continue
            # Subpart of a door? (Parent is itself a door) -> may be fixed
            if j.parent and _name_matches(j.parent, DOOR_KEYWORDS):
                continue
            if j.jtype != "revolute":
                self.add(
                    ERROR,
                    "F1",
                    cat,
                    f"Door is connected via '{j.jtype}' joint, expected revolute.",
                    name,
                )
        self._passed("F1", cat, "Doors are attached via revolute joints.")

        # F2: Door material != Corpus material
        for j in revolute_joints:
            if not j.child or not j.parent:
                continue
            door = self.m.links.get(j.child)
            body = self.m.links.get(j.parent)
            if door is None or body is None:
                continue
            dcol = self._link_color(door)
            bcol = self._link_color(body)
            if dcol is None or bcol is None:
                self.add(
                    WARN,
                    "F2",
                    cat,
                    "Door or corpus material missing; color difference unverifiable.",
                    j.child,
                )
            elif _close_list(dcol, bcol, tol=1e-3):
                self.add(
                    ERROR,
                    "F2",
                    cat,
                    "Door has the same material color as its corpus "
                    "(according to prompt they MUST differ).",
                    j.child,
                )
        self._passed("F2", cat, "Doors have a different color than their corpus.")

        # F3: Door Count Rule (Corpus > 0.6 m -> 2 doors, else 1)
        # Heuristic: Corpus = parent of a revolute joint; Door count = #revolute children.
        # Floors and walls are not cabinets (room doors are attached to them, which
        # are always single doors regardless of wall width) -> exclude.
        cabinet_doors: Dict[str, int] = {}
        for j in revolute_joints:
            if not j.parent or j.parent == "base_link":
                continue
            if _name_matches(j.parent, WALL_KEYWORDS):
                continue
            cabinet_doors[j.parent] = cabinet_doors.get(j.parent, 0) + 1
        for body_name, ndoors in cabinet_doors.items():
            body = self.m.links.get(body_name)
            if body is None or not body.visuals or body.visuals[0].geometry is None:
                continue
            # Corner cabinets / L-shapes: simple width rule is inapplicable
            # (ambiguous "width", often only one corner door) -> skip.
            if _name_matches(body_name, CORNER_KEYWORDS) or len(body.visuals) > 1:
                continue
            geom = body.visuals[0].geometry
            if geom.kind != "box" or not geom.size:
                continue
            width = max(geom.size[0], geom.size[1])  # larger horizontal edge
            if width > 0.6 and ndoors < 2:
                self.add(
                    WARN,
                    "F3",
                    cat,
                    f"Corpus width ~{width:.2f} m (>0.6 m), but only {ndoors} door(s) - "
                    "according to prompt typically two doors (splitting from middle).",
                    body_name,
                )
            elif width <= 0.6 and ndoors > 1:
                self.add(
                    WARN,
                    "F3",
                    cat,
                    f"Corpus width ~{width:.2f} m (<=0.6 m), but {ndoors} doors - "
                    "according to prompt typically one single door.",
                    body_name,
                )
        self._passed(
            "F3", cat, "Door count matches corpus width (1 for <=0.6 m, 2 for >0.6 m)."
        )

    # --- K: Corner Cabinets (Single-Entity-Rule) ----------------------------

    def check_corner_cabinets(self):
        cat = "K Corner Cabinets"
        found_any = False
        for name in self.m.link_order:
            if not _name_matches(name, CORNER_KEYWORDS):
                continue
            # Corner doors/handles themselves are not the corpus.
            if _name_matches(name, DOOR_KEYWORDS) or _name_matches(
                name, HANDLE_KEYWORDS
            ):
                continue
            found_any = True
            link = self.m.links[name]
            nvis = len(link.visuals)
            ncol = len(link.collisions)
            # K1: Single-Entity-Rule - ONE link with multiple geometries for L-shape.
            if nvis < 2 or ncol < 2:
                self.add(
                    WARN,
                    "K1",
                    cat,
                    "Corner cabinet should form the L-shape as ONE link with multiple "
                    f"<visual>/<collision> elements (found: {nvis} visual, {ncol} collision). "
                    "If modeled as two separate cabinets: merge them into one link "
                    "according to prompt.",
                    name,
                )
        if not found_any:
            self.add(
                INFO,
                "K0",
                cat,
                "No links containing 'corner'/'eck' pattern found - "
                "single-entity check skipped.",
            )
        self._passed(
            "K1",
            cat,
            "Corner cabinets are modeled as a single link with L-shape (multiple geometries).",
        )

    # --- G: Values & Units ---------------------------------------------

    def check_values_units(self):
        cat = "G Values / Units"

        # G1/G3: all geometry dimensions positive & plausible
        def check_geom(geom: Optional[Geometry], where: str, name: str):
            if geom is None:
                return
            dims = []
            if geom.kind == "box" and geom.size:
                dims = list(geom.size)
            elif geom.kind == "cylinder":
                dims = [geom.radius, geom.length]
            elif geom.kind == "sphere":
                dims = [geom.radius]
            for d in dims:
                if d is None:
                    self.add(
                        ERROR, "G1", cat, f"Invalid/missing dimension ({where}).", name
                    )
                elif d <= 0:
                    self.add(
                        ERROR, "G3", cat, f"Non-positive dimension {d} ({where}).", name
                    )
                elif d > MAX_REASONABLE_DIM:
                    self.add(
                        WARN,
                        "G3",
                        cat,
                        f"Extremely large dimension {d} m ({where}); possible unit error.",
                        name,
                    )

        for name in self.m.link_order:
            link = self.m.links[name]
            for v in link.visuals:
                check_geom(v.geometry, "visual", name)
            for c in link.collisions:
                check_geom(c.geometry, "collision", name)
        self._passed("G1", cat, "All dimensions are valid numbers.")
        self._passed(
            "G3", cat, "All dimensions are positive and plausible (meter scale)."
        )

        # G2: Angles in radian range
        for j in self.m.joints:
            for ang in j.origin.rpy:
                if abs(ang) > 2 * math.pi + 0.01:
                    self.add(
                        WARN,
                        "G2",
                        cat,
                        f"rpy angle {ang} > 2pi; possibly degrees instead of radians.",
                        j.name,
                    )
        self._passed("G2", cat, "Angles are within radian range.")

    # --- H: Standard Proportions (Heuristics) -----------------------------

    def check_proportions(self):
        cat = "H Proportions (Heuristics)"
        transforms = self._world_transforms()

        def world_z_span(name) -> Optional[Tuple[float, float, float]]:
            link = self.m.links.get(name)
            if link is None or name not in transforms or not link.visuals:
                return None
            zmins, zmaxs = [], []
            for v in link.visuals:
                if v.geometry is None:
                    continue
                box = _aabb_of_geometry(v.geometry, transforms[name], v.origin)
                if box is not None:
                    zmins.append(box[0][2])
                    zmaxs.append(box[1][2])
            if not zmins:
                return None
            zmin, zmax = min(zmins), max(zmaxs)
            return zmin, zmax, (zmax - zmin)

        any_check = False
        for name in self.m.link_order:
            if name == "base_link":
                continue
            span = world_z_span(name)
            if span is None:
                continue
            zmin, zmax, height = span

            if _name_matches(name, WALL_KEYWORDS):
                any_check = True
                if not (2.2 <= zmax <= 2.9):
                    self.add(
                        INFO,
                        "H1",
                        cat,
                        f"Wall top edge at z={zmax:.2f} m (Standard 2.4-2.7 m).",
                        name,
                    )
            elif _name_matches(name, WALL_CAB_KEYWORDS):
                any_check = True
                if zmin < 1.2:
                    self.add(
                        INFO,
                        "H3",
                        cat,
                        f"Wall cabinet bottom edge at z={zmin:.2f} m (Standard ~1.35 m).",
                        name,
                    )
            elif _name_matches(name, COUNTER_KEYWORDS) or _name_matches(
                name, BASE_CAB_KEYWORDS
            ):
                any_check = True
                if not (0.78 <= zmax <= 1.0):
                    self.add(
                        INFO,
                        "H2",
                        cat,
                        f"Countertop/Base cabinet top edge at z={zmax:.2f} m "
                        "(Standard ~0.9 m).",
                        name,
                    )
            elif _name_matches(name, TABLE_KEYWORDS):
                any_check = True
                if not (0.68 <= zmax <= 0.82):
                    self.add(
                        INFO,
                        "H4",
                        cat,
                        f"Table top edge at z={zmax:.2f} m (Standard ~0.75 m).",
                        name,
                    )

        if not any_check:
            self.add(
                INFO,
                "H0",
                cat,
                "No links with standard naming patterns (wall/counter/cabinet/table) "
                "found - height heuristics skipped.",
            )
        self._passed("H1", cat, "Wall heights plausible.")
        self._passed("H2", cat, "Countertop heights plausible.")
        self._passed("H3", cat, "Wall cabinet heights plausible.")
        self._passed("H4", cat, "Table heights plausible.")

    # --- I: Collisions / Overlaps (Heuristics) ----------------------

    def check_overlaps(self):
        cat = "I Collisions (Heuristics)"
        transforms = self._world_transforms()

        # A LIST of World-AABBs per link (one for each visual geometry).
        # This properly handles L-shaped corner cabinets and empty inner corners
        # won't trigger false alarms (no Union-AABB).
        boxes: Dict[str, List[Tuple[List[float], List[float]]]] = {}
        for name in self.m.link_order:
            link = self.m.links[name]
            if name not in transforms or not link.visuals:
                continue
            blist = []
            for v in link.visuals:
                if v.geometry is None:
                    continue
                box = _aabb_of_geometry(v.geometry, transforms[name], v.origin)
                if box is not None:
                    blist.append(box)
            if blist:
                boxes[name] = blist

        names = [n for n in boxes if n != "base_link"]
        flagged = False
        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                a, b = names[i], names[k]
                # Same object group (e.g. corpus + top + sink) -> expected overlay
                if self._top_ancestor(a) == self._top_ancestor(b):
                    continue
                # Largest penetration over all box pairs of both links
                pen = 0.0
                for ba in boxes[a]:
                    for bb in boxes[b]:
                        p = _penetration(ba, bb)
                        if p is not None and p > pen:
                            pen = p
                if pen <= PENETRATION_THRESHOLD:
                    continue
                # Walls: flush fitting is intended -> just INFO
                if _name_matches(a, WALL_KEYWORDS) or _name_matches(b, WALL_KEYWORDS):
                    self.add(
                        INFO,
                        "I1",
                        cat,
                        f"'{a}' and '{b}' overlap ~{pen:.2f} m (Wall - flush fitting potentially intended).",
                        f"{a} <-> {b}",
                    )
                else:
                    flagged = True
                    self.add(
                        WARN,
                        "I1",
                        cat,
                        f"'{a}' and '{b}' intersect by ~{pen:.2f} m "
                        "(Objects should not stand inside each other).",
                        f"{a} <-> {b}",
                    )
        if not flagged:
            self._passed(
                "I1", cat, "No significant penetrations between different objects."
            )


# ---------------------------------------------------------------------------
# Small Numerical Helpers
# ---------------------------------------------------------------------------


def _close_list(a, b, tol: float = 1e-6) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _geom_origin_sig(geom: Optional[Geometry], origin: Origin) -> tuple:
    """Signature from geometry AND placement (for Visual/Collision comparison)."""
    gsig = geom.signature() if geom else None
    return (
        gsig,
        tuple(round(x, 6) for x in origin.xyz),
        tuple(round(x, 6) for x in origin.rpy),
    )


def _penetration(box_a, box_b) -> Optional[float]:
    """Minimum overlap depth over all 3 axes; None if disjoint."""
    amin, amax = box_a
    bmin, bmax = box_b
    overlaps = []
    for i in range(3):
        ov = min(amax[i], bmax[i]) - max(amin[i], bmin[i])
        if ov <= 0:
            return None
        overlaps.append(ov)
    return min(overlaps)


# ---------------------------------------------------------------------------
# Input Reading / Extract xml block
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:xml|XML)?\s*\n(.*?)```", re.DOTALL)


def load_input(path: str, check_output_format: bool) -> Tuple[str, List[Result]]:
    """Reads file, extracts xml codeblock if present, checks zero-prose rule."""
    results: List[Result] = []
    cat = "O Output Format"
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = _FENCE_RE.findall(raw)
    if blocks:
        if check_output_format:
            if len(blocks) != 1:
                results.append(
                    Result(
                        "O1",
                        cat,
                        ERROR,
                        f"Found {len(blocks)} code blocks, expected exactly one.",
                    )
                )
            else:
                results.append(
                    Result("O1", cat, PASS, "Exactly one code block present.")
                )
            outside = _FENCE_RE.sub("", raw).strip()
            if outside:
                preview = outside[:60].replace("\n", " ")
                results.append(
                    Result(
                        "O2",
                        cat,
                        ERROR,
                        f'Text found outside the code block: "{preview}..." '
                        "(Zero-prose rule violated).",
                    )
                )
            else:
                results.append(
                    Result("O2", cat, PASS, "No text outside the code block.")
                )
        return blocks[0], results

    # No fenced block -> treat entire file as URDF
    if check_output_format:
        results.append(
            Result(
                "O1",
                cat,
                WARN,
                "No ```xml block found; entire file will be evaluated as URDF.",
            )
        )
    return raw, results


# ---------------------------------------------------------------------------
# Report Output
# ---------------------------------------------------------------------------


def _supports_color(no_color: bool) -> bool:
    return (not no_color) and sys.stdout.isatty()


def print_report(path: str, results: List[Result], no_color: bool, verbose: bool):
    color = _supports_color(no_color)

    def c(code, text):
        return f"\033[{code}m{text}\033[0m" if color else text

    sev_color = {ERROR: "1;31", WARN: "1;33", INFO: "1;36", PASS: "1;32"}

    print("=" * 70)
    print("URDF Validation Report")
    print(f"File: {path}")
    print("=" * 70)

    # Group by category, order by severity
    categories: Dict[str, List[Result]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat in sorted(categories):
        rows = categories[cat]
        rows.sort(key=lambda r: _SEVERITY_ORDER[r.severity])
        shown = [
            r
            for r in rows
            if verbose
            or r.severity != PASS
            or not any(o.severity != PASS and o.check_id == r.check_id for o in rows)
        ]
        if not shown:
            shown = rows
        print(f"\n[{cat}]")
        for r in shown:
            if r.severity == PASS and not verbose:
                # Only show PASS if no problem exists for this check
                if any(o.check_id == r.check_id and o.severity != PASS for o in rows):
                    continue
            icon = c(sev_color[r.severity], _SEVERITY_ICON[r.severity])
            tgt = f"  [{r.target}]" if r.target else ""
            print(f"  {icon} {r.check_id:<4} {r.message}{tgt}")

    # Summary
    n_err = sum(1 for r in results if r.severity == ERROR)
    n_warn = sum(1 for r in results if r.severity == WARN)
    n_info = sum(1 for r in results if r.severity == INFO)
    print("\n" + "-" * 70)
    print(f"Summary: {n_err} Errors, {n_warn} Warnings, {n_info} Infos")
    if n_err == 0:
        print(c("1;32", "Result: PASSED"))
    else:
        print(c("1;31", "Result: FAILED"))
    print("-" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validates a URDF file against the requirements of the "
        "room URDF generation prompt."
    )
    parser.add_argument("file", help="URDF file or Markdown/Text with ```xml block")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors (Exit code 1)."
    )
    parser.add_argument(
        "--no-collision-check",
        action="store_true",
        help="Skip heuristic collision checks.",
    )
    parser.add_argument(
        "--check-output-format",
        action="store_true",
        help="Check zero-prose rule (exactly one xml block, no external text).",
    )
    parser.add_argument("--json", action="store_true", help="Output result as JSON.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual checks that have passed as well.",
    )
    args = parser.parse_args(argv)

    # Load file / extract xml
    try:
        urdf_text, pre_results = load_input(args.file, args.check_output_format)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"ERROR while reading: {e}", file=sys.stderr)
        return 2

    results: List[Result] = list(pre_results)

    # Parse XML
    try:
        root = ET.fromstring(urdf_text)
        results.append(Result("A2", "A Basic Structure", PASS, "File is valid XML."))
    except ET.ParseError as e:
        results.append(
            Result("A2", "A Basic Structure", ERROR, f"XML parsing error: {e}")
        )
        if args.json:
            print(
                json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)
            )
        else:
            print_report(args.file, results, args.no_color, args.verbose)
        return 1

    if _lname(root) != "robot":
        results.append(
            Result(
                "A3",
                "A Basic Structure",
                ERROR,
                f"Root element is <{_lname(root)}>, expected <robot>.",
            )
        )
        if args.json:
            print(
                json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)
            )
        else:
            print_report(args.file, results, args.no_color, args.verbose)
        return 1

    # Build and validate model
    model = build_model(root)
    validator = UrdfValidator(model, check_collisions=not args.no_collision_check)
    results.extend(validator.run())

    # Output
    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        print_report(args.file, results, args.no_color, args.verbose)

    n_err = sum(1 for r in results if r.severity == ERROR)
    n_warn = sum(1 for r in results if r.severity == WARN)
    if n_err > 0 or (args.strict and n_warn > 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

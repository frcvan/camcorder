#!/usr/bin/env python3
"""Extract fixed geometry from the 2026 AdvantageScope model.

The output is geometry, not image correspondences: detecting an arbitrary line
in footage does not identify which of the field's similar markings it is.

Two products, for two different kinds of evidence:

* `markings.json` — the long red and blue alliance tape, whose colour says which
  line it is, so a match needs no pose to be identified.
* `structure.json` — box outlines of the structure each AprilTag is mounted on,
  keyed by tag id. These are *not* identified from the picture at all: the tag
  that was decoded already says which structure is in view, so the only thing
  left to check is whether an image edge really lies where the model says that
  edge is. Out-of-plane parts of these boxes are the evidence a single planar
  tag cannot give: no pose scale change can move a tag's own corners and the
  top of the trench it stands in by the same amount.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "assets/field/2026-rebuilt-advantagescope"


def transform(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape((4, 4), order="F")
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(node.get("scale", [1, 1, 1]))
    matrix[:3, 3] = node.get("translation", [0, 0, 0])
    return matrix


def gltf_json(path: Path) -> dict:
    with path.open("rb") as file:
        magic, version, _ = struct.unpack("<4sII", file.read(12))
        size, kind = struct.unpack("<II", file.read(8))
        if magic != b"glTF" or version != 2 or kind != 0x4E4F534A:
            raise ValueError("expected a glTF 2 JSON chunk")
        return json.loads(file.read(size))


def field_markings(model: Path = ASSET / "model.glb") -> list[dict]:
    """Every gaffer-tape line on the field, in field coordinates."""
    gltf = gltf_json(model)
    layout = json.loads((ROOT / "assets/tags/2026-rebuilt-welded.json").read_text())
    length = layout["field-dimensions"]["length"]
    width = layout["field-dimensions"]["width"]
    to_field = np.array([[-1, 0, 0, length/2], [0, 0, 1, width/2], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    result: list[dict] = []

    def visit(index: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][index]
        matrix = parent @ transform(node)
        name = node.get("name", "")
        if name.startswith("Gaffer Tape,") and "mesh" in node:
            for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
                accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
                low, high = np.array(accessor["min"]), np.array(accessor["max"])
                axis = int(np.argmax(high - low))
                if high[axis] - low[axis] < 0.3:
                    continue
                middle = (low + high) * 0.5
                start, end = middle.copy(), middle.copy()
                start[axis], end[axis] = low[axis], high[axis]
                points = [(to_field @ matrix @ np.append(point, 1))[:3].tolist() for point in (start, end)]
                if all(math.isfinite(value) for point in points for value in point):
                    result.append({"name": name, "start": points[0], "end": points[1]})
        for child in node.get("children", []):
            visit(child, matrix)

    for node in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(node, np.eye(4))
    return result


# Structure names worth keeping: mount hardware, not game pieces or carpet.
STRUCTURE = (
    "AprilTag Base Poly",
    "Mount Assy",
    "Trench",
    "Hub",
    "Tower",
    "Alliance Wall",
)
# A node further than this from every tag is not that tag's structure.
STRUCTURE_RADIUS_M = 2.0
# Edges shorter than this are too small to find in broadcast footage.
MIN_EDGE_M = 0.25


def field_structure(model: Path = ASSET / "model.glb") -> list[dict]:
    """Bounding boxes of the structure around each tag, keyed by tag id.

    Every mesh node whose bounding box lies within `STRUCTURE_RADIUS_M` of a
    tag's centre is attached to that tag. The union of those boxes gives the
    mount and whatever it is bolted to; its twelve edges are what the matcher
    looks for in the picture.
    """
    gltf = gltf_json(model)
    layout = json.loads((ROOT / "assets/tags/2026-rebuilt-welded.json").read_text())
    length = layout["field-dimensions"]["length"]
    width = layout["field-dimensions"]["width"]
    to_field = np.array(
        [[-1, 0, 0, length / 2], [0, 0, 1, width / 2], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    tags = {
        int(tag["ID"]): np.array(
            [
                tag["pose"]["translation"]["x"],
                tag["pose"]["translation"]["y"],
                tag["pose"]["translation"]["z"],
            ]
        )
        for tag in layout["field-tags"]
    }
    boxes: list[tuple[str, np.ndarray, np.ndarray]] = []

    def visit(index: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][index]
        matrix = parent @ transform(node)
        name = node.get("name", "")
        if "mesh" in node and any(word in name for word in STRUCTURE):
            low = np.full(3, np.inf)
            high = np.full(3, -np.inf)
            for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
                accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
                small, large = np.array(accessor["min"]), np.array(accessor["max"])
                corners = np.array(
                    [
                        [small[0] if a else large[0], small[1] if b else large[1], small[2] if c else large[2]]
                        for a in (0, 1)
                        for b in (0, 1)
                        for c in (0, 1)
                    ]
                )
                moved = (matrix @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
                field = np.stack(
                    [length / 2 - moved[:, 0], width / 2 + moved[:, 2], moved[:, 1]], axis=1
                )
                low = np.minimum(low, field.min(axis=0))
                high = np.maximum(high, field.max(axis=0))
            boxes.append((name, low, high))
        for child in node.get("children", []):
            visit(child, matrix)

    for node in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(node, np.eye(4))

    per_tag: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, low, high in boxes:
        centre = (low + high) * 0.5
        nearest = min(tags.items(), key=lambda item: float(np.linalg.norm(item[1] - centre)))
        if float(np.linalg.norm(nearest[1] - centre)) > STRUCTURE_RADIUS_M:
            continue
        if nearest[0] in per_tag:
            previous_low, previous_high = per_tag[nearest[0]]
            per_tag[nearest[0]] = (np.minimum(previous_low, low), np.maximum(previous_high, high))
        else:
            per_tag[nearest[0]] = (low, high)

    result: list[dict] = []
    for tag in sorted(per_tag):
        low, high = per_tag[tag]
        corners = [
            [low[0] if a else high[0], low[1] if b else high[1], low[2] if c else high[2]]
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
        ]
        segments = []
        for i in range(8):
            for j in range(i + 1, 8):
                # Box edges differ in exactly one coordinate.
                if sum(1 for a, b in zip(corners[i], corners[j]) if abs(a - b) > 1e-9) != 1:
                    continue
                if float(np.linalg.norm(np.array(corners[i]) - corners[j])) < MIN_EDGE_M:
                    continue
                segments.append([corners[i], corners[j]])
        if segments:
            result.append({"id": tag, "segments": segments})
    return result


def alliance_tape(markings: list[dict]) -> list[dict]:
    """The lines in `markings` a match can be identified without already having
    a pose: the red and blue tape that runs the field's whole width.

    Colour is what makes these usable. White and black field lines are just as
    visible, but telling the centre line from a goal line needs a pose, which is
    the thing being solved for. A short strip of alliance tape elsewhere on the
    field is the wrong line to match to, so only the full-width runs are kept.
    """
    return [
        line
        for line in markings
        if ("Red," in line["name"] or "Blue," in line["name"])
        and "317.19in" in line["name"]
    ]


if __name__ == "__main__":
    import sys

    what = sys.argv[1] if len(sys.argv) > 1 else "markings"
    if what == "structure":
        print(json.dumps(field_structure(), indent=2))
    else:
        print(json.dumps(alliance_tape(field_markings()), indent=2))

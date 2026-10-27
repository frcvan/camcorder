#!/usr/bin/env python3
"""Send camcorder pose estimates and the footage they came from to Rerun.

`camcorder pose` writes one JSON report per video. This reads those reports
(plus the WPILib field layout they name) and logs, for each video:

* the field outline and every tag on it, in field coordinates;
* the camera, as a pinhole plus a transform per solved frame;
* the frame itself, at the same time, projected from that camera;
* the camera's path, and its height / heading / pitch / roll / residual over
  the timeline.

One command, then look at it:

    python scripts/rerun_poses.py path/to/poses/*.json --spawn
    python scripts/rerun_poses.py path/to/poses/ --rrd field.rrd

Each report becomes its own camera under `world/cameras/<name>`, so several
matches of the same event line up in one field with their tags and paths.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr

#: FRC field tags are 6.5 inches on a side unless the report says otherwise.
DEFAULT_TAG_SIZE_M = 0.1651
#: Where the layouts that ship with camcorder live, relative to the repo root.
DEFAULT_LAYOUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "tags"
#: Tags are thin, but not flat: give them a little depth so they are visible
#: edge-on in the viewer.
TAG_THICKNESS_M = 0.005
#: Poses this close together are the same camera, to `--station-tolerance` in
#: metres and this many degrees. A broadcast camera does not move during a
#: match; two poses that differ by more are two cameras in the same video.
STATION_ANGLE_TOLERANCE_DEG = 12.0


@dataclass
class Camera:
    """One pose report and the video it was measured from."""

    name: str
    report: dict
    video: Path | None
    layout: dict | None

    @property
    def solved(self) -> list[dict]:
        return self.report.get("solved", [])

    @property
    def position(self) -> np.ndarray:
        """Camera centre of the first solved frame, in field coordinates."""
        return np.array(self.solved[0]["position"], dtype=float)

    def intrinsics(self) -> tuple[float, float, float, int, int]:
        """(fx, cx, cy, width, height) from the report's assumed field of view."""
        width = int(self.report["frame_width"])
        height = int(self.report["frame_height"])
        fov_deg = float(self.report.get("horizontal_fov_deg", 60.0))
        fx = float((width * 0.5) / np.tan(np.radians(fov_deg) * 0.5))
        return fx, float((width - 1) * 0.5), float((height - 1) * 0.5), width, height


# ---------------------------------------------------------------------------
# reading what camcorder wrote
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def resolve_layout(report: dict, layout_dir: Path) -> dict | None:
    """The field layout a report names: a bundled name, or a path on disk."""
    spec = report.get("tag_layout")
    if not spec:
        return None
    for candidate in (layout_dir / f"{spec}.json", Path(spec)):
        if candidate.is_file():
            try:
                return load_json(candidate)
            except (OSError, ValueError) as error:
                print(f"warning: could not read {candidate}: {error}", file=sys.stderr)
                return None
    print(
        f"warning: no field layout for {spec!r} — tags will not be drawn "
        f"(looked in {layout_dir} and at that path)",
        file=sys.stderr,
    )
    return None


def layout_tags(layout: dict | None) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(id, translation, quaternion xyzw) for every tag, either schema."""
    if not layout:
        return []
    entries = layout.get("field-tags") or layout.get("tags") or []
    tags = []
    for entry in entries:
        pose = entry["pose"]
        translation = pose["translation"]
        quaternion = pose["rotation"]["quaternion"]
        tags.append(
            (
                int(entry["ID"]),
                np.array([translation["x"], translation["y"], translation["z"]], dtype=float),
                # WPILib writes WXYZ, rerun wants XYZW.
                np.array(
                    [
                        quaternion["X"],
                        quaternion["Y"],
                        quaternion["Z"],
                        quaternion["W"],
                    ],
                    dtype=float,
                ),
            )
        )
    return sorted(tags, key=lambda tag: tag[0])


def field_size(layout: dict | None) -> tuple[float, float] | None:
    if not layout:
        return None
    dimensions = layout.get("field-dimensions") or layout.get("field")
    if not dimensions:
        return None
    return float(dimensions["length"]), float(dimensions["width"])


def collect(paths: list[Path], layout_dir: Path) -> list[Camera]:
    """Every pose report named on the command line, directories expanded."""
    reports: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.glob("*.json"))
            if not found:
                print(f"warning: no JSON reports in {path}", file=sys.stderr)
            reports.extend(found)
        elif path.is_file():
            reports.append(path)
        else:
            print(f"warning: {path} does not exist", file=sys.stderr)

    cameras = []
    for path in reports:
        try:
            report = load_json(path)
        except (OSError, ValueError) as error:
            print(f"warning: skipping {path}: {error}", file=sys.stderr)
            continue
        if "solved" not in report:
            print(f"warning: skipping {path}: not a pose report", file=sys.stderr)
            continue
        video = Path(report["video"]) if report.get("video") else None
        if video is not None and not video.is_file():
            print(f"warning: {path}: video {video} is gone", file=sys.stderr)
            video = None
        cameras.append(
            Camera(
                name=path.stem,
                report=report,
                video=video,
                layout=resolve_layout(report, layout_dir),
            )
        )
    return cameras


# ---------------------------------------------------------------------------
# logging to rerun
# ---------------------------------------------------------------------------


def log_world(cameras: list[Camera], tag_size: float) -> None:
    """The field and its tags, shared by every camera in the recording."""
    layouts: dict[str, dict] = {}
    for camera in cameras:
        if camera.layout is not None:
            layouts.setdefault(camera.report["tag_layout"], camera.layout)

    for name, layout in layouts.items():
        size = field_size(layout)
        if size:
            length, width = size
            corners = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [length, 0.0, 0.0],
                    [length, width, 0.0],
                    [0.0, width, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
            rr.log(
                f"world/{name}/field/outline",
                rr.LineStrips3D([corners], colors=[(120, 120, 120)], radii=0.02),
                static=True,
            )

        tags = layout_tags(layout)
        if not tags:
            continue
        translations = np.array([tag[1] for tag in tags])
        quaternions = np.array([tag[2] for tag in tags])
        rr.log(
            f"world/{name}/tags",
            rr.Boxes3D(
                centers=translations,
                sizes=np.tile([TAG_THICKNESS_M, tag_size, tag_size], (len(tags), 1)),
                quaternions=quaternions,
                colors=[(255, 255, 255)],
                labels=[str(tag[0]) for tag in tags],
            ),
            static=True,
        )


def camera_rotation(frame: dict) -> np.ndarray:
    """The frame's rotation as rerun wants it: `xyzw`, camera axes to field axes.

    The report stores field-to-camera, because that is what projects a field
    point into the picture. A transform in rerun goes the other way — it places
    the camera's local axes in the world — so the quaternion is conjugated.
    Using it as-is still draws a frustum, just one pointing backwards.
    """
    w, x, y, z = frame["rotation"]
    return np.array([-x, -y, -z, w], dtype=float)


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two rotations, in degrees, ignoring the sign of the pair."""
    dot = min(1.0, max(-1.0, abs(float(np.dot(a, b)))))
    return math.degrees(2.0 * math.acos(dot))


def cluster_frames(
    frames: list[dict],
    position_tol: float,
    angle_tol: float,
) -> list[list[dict]]:
    """Group poses that are the same camera standing still.

    A tiled broadcast frame holds several cameras, and a report measured on the
    whole frame mixes them: tags from the overview pane and from a sub-view give
    poses metres apart. Those are different cameras, not one camera teleporting,
    so they belong on separate entities — which is also what keeps each of them
    still across the match.
    """
    clusters: list[list[dict]] = []
    for frame in frames:
        position = np.array(frame["position"], dtype=float)
        rotation = camera_rotation(frame)
        for cluster in clusters:
            reference = cluster[0]
            near = (
                np.linalg.norm(position - np.array(reference["position"], dtype=float))
                <= position_tol
            )
            if near and quaternion_angle_deg(rotation, camera_rotation(reference)) <= angle_tol:
                cluster.append(frame)
                break
        else:
            clusters.append([frame])
    return clusters


def log_camera_static(entity: str, camera: Camera) -> None:
    fx, cx, cy, width, height = camera.intrinsics()
    rr.log(
        f"{entity}/image",
        rr.Pinhole(
            width=width,
            height=height,
            focal_length=fx,
            principal_point=(cx, cy),
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
        static=True,
    )


def log_trajectory(name: str, clusters: list[list[dict]]) -> None:
    """The camera stations, in world coordinates so they never move with a pose."""
    strips = [np.array([frame["position"] for frame in cluster]) for cluster in clusters]
    if not strips:
        return
    rr.log(
        f"world/paths/{name}",
        rr.LineStrips3D(
            strips,
            colors=[(90, 200, 255)],
            radii=0.03,
            labels=[f"cam{index + 1}" for index in range(len(strips))],
            show_labels=True,
        ),
        static=True,
    )


def read_frame(video: Path, time: float, width: int, height: int) -> np.ndarray | None:
    """One RGB frame at `time`, or None if ffmpeg cannot give us one."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{time:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        output = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError:
        print("warning: ffmpeg not found — logging poses without video", file=sys.stderr)
        return None
    expected = width * height * 3
    if output.returncode != 0 or len(output.stdout) != expected:
        return None
    return np.frombuffer(output.stdout, dtype=np.uint8).reshape(height, width, 3)


def log_frames(camera: Camera, cluster: list[dict], entity: str, with_video: bool, budget: list[int]) -> None:
    """One camera station: where it stood, and what it saw while it stood there.

    The transform goes on the entity itself and the pinhole on its child, which
    is what makes the frustum sit at the solved position: a transform logged
    below a pinhole applies to that child only, leaving the camera at the origin
    looking straight up.
    """
    width = int(camera.report["frame_width"])
    height = int(camera.report["frame_height"])

    for frame in cluster:
        rr.set_time("video_time", duration=float(frame["time"]))
        rr.set_time("frame", sequence=int(float(frame["time"]) * 1000))

        rr.log(
            entity,
            rr.Transform3D(
                translation=np.array(frame["position"], dtype=float),
                rotation=rr.Quaternion(xyzw=[float(value) for value in camera_rotation(frame)]),
            ),
        )

        plot = f"plots/{camera.name}/{entity.split('/')[-1]}"
        position = frame["position"]
        rr.log(f"{plot}/x", rr.Scalars(position[0]))
        rr.log(f"{plot}/y", rr.Scalars(position[1]))
        rr.log(f"{plot}/height", rr.Scalars(position[2]))
        rr.log(f"{plot}/yaw", rr.Scalars(frame["yaw_deg"]))
        rr.log(f"{plot}/pitch", rr.Scalars(frame["pitch_deg"]))
        rr.log(f"{plot}/roll", rr.Scalars(frame["roll_deg"]))
        rr.log(f"{plot}/reprojection_rms", rr.Scalars(frame["rms_px"]))
        rr.log(f"{plot}/tags_used", rr.Scalars(len(frame["tags"])))

        if not with_video or camera.video is None or budget[0] <= 0:
            continue
        image = read_frame(camera.video, float(frame["time"]), width, height)
        if image is not None:
            rr.log(f"{entity}/image", rr.Image(image))
            budget[0] -= 1


def usable(frame: dict, min_height: float | None) -> bool:
    """Whether a frame's pose is physically possible.

    A single tag leaves a mirror ambiguity, and the mirrored branch is sometimes
    the one that fits best. It puts the camera below the carpet, which no camera
    ever was, so a height floor removes it without touching the good frames.
    """
    if min_height is None:
        return True
    return float(frame["position"][2]) >= min_height


def log_pose(
    camera: Camera,
    with_video: bool,
    max_frames: int | None,
    min_height: float | None,
    station_tolerance: float,
) -> list[list[dict]]:
    kept = [frame for frame in camera.solved if usable(frame, min_height)]
    dropped = len(camera.solved) - len(kept)
    if dropped:
        print(
            f"  {camera.name}: dropping {dropped} solved frame(s) below "
            f"{min_height:g}m, which the mirror solution produces",
            file=sys.stderr,
        )
    camera.report = dict(camera.report, solved=kept)

    clusters = cluster_frames(kept, station_tolerance, STATION_ANGLE_TOLERANCE_DEG)
    log_trajectory(camera.name, clusters)

    budget = [max_frames if max_frames is not None else len(kept)]
    for index, cluster in enumerate(clusters):
        # A single station is just the camera; several get a name each, since
        # they are different cameras that happened to be measured in one file.
        entity = (
            f"world/cameras/{camera.name}"
            if len(clusters) == 1
            else f"world/cameras/{camera.name}/cam{index + 1}"
        )
        log_camera_static(entity, camera)
        log_frames(camera, cluster, entity, with_video, budget)
    return clusters


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log camcorder pose reports and their footage to Rerun.",
        epilog="See https://rerun.io for the viewer.",
    )
    parser.add_argument(
        "poses",
        nargs="+",
        type=Path,
        metavar="REPORT",
        help="pose JSON from `camcorder pose`, or a directory of them",
    )
    parser.add_argument(
        "--rrd",
        type=Path,
        metavar="FILE",
        help="write a recording to FILE instead of opening the viewer",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="open the Rerun viewer when done (instead of only writing --rrd)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="log poses only; do not decode frames out of the videos",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        metavar="N",
        help="decode at most N frames per camera",
    )
    parser.add_argument(
        "--station-tolerance",
        type=float,
        default=1.5,
        metavar="M",
        help="poses within M metres of each other are the same fixed camera "
        "(default: 1.5); further apart means a different camera in the same "
        "frame, and they are logged separately",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.5,
        metavar="M",
        help="skip frames whose camera is below M metres (default: 0.5, which "
        "drops the mirror solution a single tag allows; use a negative value "
        "to keep everything)",
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=DEFAULT_LAYOUT_DIR,
        metavar="DIR",
        help=f"where the bundled field layouts live (default: {DEFAULT_LAYOUT_DIR})",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=DEFAULT_TAG_SIZE_M,
        metavar="M",
        help=f"tag edge length, metres (default: {DEFAULT_TAG_SIZE_M})",
    )
    parser.add_argument(
        "--application-id",
        default="camcorder",
        help="recording name shown in the viewer (default: camcorder)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cameras = collect(args.poses, args.layout_dir)
    if not cameras:
        print("error: nothing to log", file=sys.stderr)
        return 1

    rr.init(args.application_id, spawn=False)
    if args.spawn:
        rr.spawn()
    if args.rrd is not None or not args.spawn:
        destination = args.rrd or Path("poses.rrd")
        destination.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(destination))
        print(f"wrote {destination}")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_world(cameras, args.tag_size)

    min_height = None if args.min_height is None else args.min_height
    for camera in cameras:
        stations = log_pose(
            camera,
            not args.no_video,
            args.max_frames,
            min_height,
            args.station_tolerance,
        )
        report = camera.report
        solved = sum(len(cluster) for cluster in stations)
        print(
            f"{camera.name}: {solved}/{report.get('sampled', '?')} frames solved, "
            f"{len(stations)} camera station(s), tags {report.get('tags_seen', [])}, "
            f"{'video' if camera.video and not args.no_video else 'no video'}"
        )
        for index, cluster in enumerate(stations):
            heights = [float(frame["position"][2]) for frame in cluster]
            span = max(heights) - min(heights)
            print(
                f"  station {index + 1}: {len(cluster)} frame(s), "
                f"height {sum(heights) / len(heights):.2f}m (±{span / 2:.2f}m)"
            )
        if not stations:
            print(f"  note: {camera.name} has no solved frames to show", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

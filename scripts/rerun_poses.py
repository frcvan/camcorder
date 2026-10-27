#!/usr/bin/env python3
"""Send camcorder pose estimates and the footage they came from to Rerun.

`camcorder pose` writes one JSON report per video. This reads those reports
(plus the WPILib field layout they name) and logs, for each video:

* the field outline and every tag on it, in field coordinates;
* the camera, as a pinhole plus a transform per solved frame;
* the frame itself, at the same time, projected from that camera;
* every frame of the video itself, decoded once and logged at the video's own
  frame rate, so the picture and the plots run together;
* the camera's path, its height / heading / pitch / roll / residual and how
  much of a result each frame is over the timeline;
* the one pose fitted to every frame of the station, and the focal length the
  poses were solved with;
* the side of the field each frame was read as: the red and blue pixel counts
  behind the alliance verdict, that verdict, the end of the field the tags
  stand at, and a flag on the frames where colour and geometry disagree.

An overview pane has no side of the field to report — one wide shot holds both
alliances — so its frames carry none, and its counts stay empty.

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
from collections.abc import Iterator
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
#: `-q:v` for ffmpeg's mjpeg encoder, 2 (best) to 31 (smallest). A recording that
#: holds every frame of a match needs the pictures compressed: a raw 1080p frame
#: is 6MB, which is 60GB for ten minutes of 60fps footage, against a few tens of
#: kilobytes for the same frame as a JPEG.
JPEG_QUALITY = 6
#: `side.alliance` and `side.tags_end` as numbers, so they can be plotted. Zero
#: is "not read": an overview pane, or a picture with no coloured elements.
ALLIANCE_CODE = {"blue": 1.0, "mixed": 2.0, "red": 3.0}
END_CODE = {"low_x": 1.0, "both": 2.0, "high_x": 3.0}


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
                np.array(
                    [translation["x"], translation["y"], translation["z"]], dtype=float
                ),
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
            if (
                near
                and quaternion_angle_deg(rotation, camera_rotation(reference))
                <= angle_tol
            ):
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
            image_plane_distance=2.0,
        ),
        static=True,
    )


def log_trajectory(name: str, clusters: list[list[dict]]) -> None:
    """The camera stations, in world coordinates so they never move with a pose."""
    strips = [
        np.array([frame["position"] for frame in cluster]) for cluster in clusters
    ]
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


def video_fps(video: Path) -> float:
    """The video's average frame rate, or 30 when ffprobe will not say.

    Every frame of the video is decoded, so the times have to come from the
    frame rate rather than from the report: the report samples a handful of
    frames, and the recording now holds all of them.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        output = subprocess.run(command, capture_output=True, check=False, text=True)
    except FileNotFoundError:
        return 30.0
    rate = output.stdout.strip()
    try:
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            fps = float(numerator) / float(denominator)
        else:
            fps = float(rate)
    except (ValueError, ZeroDivisionError):
        return 30.0
    return fps if 1.0 < fps < 1000.0 else 30.0


def iter_video_jpegs(
    video: Path, limit: int | None = None
) -> Iterator[tuple[int, bytes]]:
    """Every frame of the video, as `(index, JPEG bytes)`, decoded once.

    One ffmpeg process for the whole file rather than a seek per frame: seeking
    to each timestamp re-runs the decoder from the nearest keyframe, which for a
    few thousand frames is minutes of work instead of seconds. ffmpeg does the
    compression as it decodes, so the frames are handed over as the JPEGs the
    recording will store rather than as raw pixels to compress here.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(video),
        "-an",
        "-sn",
        "-c:v",
        "mjpeg",
        "-q:v",
        str(JPEG_QUALITY),
        "-f",
        "mjpeg",
        "-",
    ]
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print(
            "warning: ffmpeg not found — logging poses without video", file=sys.stderr
        )
        return

    # Start of image. Everything between one and the next is one JPEG: entropy
    # coded data byte-stuffs any 0xFF it contains, so the marker cannot appear
    # inside a frame.
    soi = b"\xff\xd8\xff"
    index = 0
    buffer = bytearray()
    try:
        assert process.stdout is not None
        while limit is None or index < limit:
            chunk = process.stdout.read(1 << 20)
            if not chunk:
                break
            buffer += chunk
            while limit is None or index < limit:
                start = buffer.find(soi)
                if start < 0:
                    break
                following = buffer.find(soi, start + len(soi))
                if following < 0:
                    del buffer[:start]
                    break
                frame = bytes(buffer[start:following])
                del buffer[:following]
                if not frame.rstrip(b"\x00").endswith(b"\xff\xd9"):
                    continue
                yield index, frame
                index += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        process.kill()
        process.wait()


def log_frames(camera: Camera, cluster: list[dict], entity: str) -> None:
    """One camera station: where it stood, and what it saw while it stood there.

    The transform goes on the entity itself and the pinhole on its child, which
    is what makes the frustum sit at the solved position: a transform logged
    below a pinhole applies to that child only, leaving the camera at the origin
    looking straight up.
    """
    for frame in cluster:
        rr.set_time("video_time", duration=float(frame["time"]))
        rr.set_time("frame", sequence=int(float(frame["time"]) * 1000))

        rr.log(
            entity,
            rr.Transform3D(
                translation=np.array(frame["position"], dtype=float),
                rotation=rr.Quaternion(
                    xyzw=[float(value) for value in camera_rotation(frame)]
                ),
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
        if "confidence" in frame:
            rr.log(f"{plot}/confidence", rr.Scalars(frame["confidence"]))
        if "weakest_margin" in frame:
            rr.log(f"{plot}/weakest_margin", rr.Scalars(frame["weakest_margin"]))
        log_side(frame, plot)


def log_side(frame: dict, plot: str) -> None:
    """The side of the field this frame was read as, when it was read at all.

    The pixel counts are logged as they are, because they are the evidence: a
    verdict of "red" from 40 red pixels against 35 blue ones is not the same
    finding as one from 40,000 against two, and only the counts show which
    happened.
    """
    side = frame.get("side")
    if not side:
        return
    rr.log(f"{plot}/red_pixels", rr.Scalars(side["red_pixels"]))
    rr.log(f"{plot}/blue_pixels", rr.Scalars(side["blue_pixels"]))
    rr.log(f"{plot}/alliance", rr.Scalars(ALLIANCE_CODE.get(side["alliance"], 0.0)))
    rr.log(f"{plot}/tags_end", rr.Scalars(END_CODE.get(side["tags_end"], 0.0)))
    rr.log(
        f"{plot}/side_disagrees",
        rr.Scalars(1.0 if side.get("disagreement") else 0.0),
    )


def log_video(
    camera: Camera,
    clusters: list[list[dict]],
    entities: list[str],
    max_frames: int | None,
) -> None:
    """Every frame of the video, at its own timestamp.

    The report samples ten frames out of a match; logging only those leaves the
    viewer with ten stills and a timeline that jumps. Decoding the file once and
    logging every frame gives the recording the video's own frame rate, so the
    camera's picture and its plots run together the way the footage did.

    Each image goes to whichever station's solved frames are nearest in time, so
    a file that holds two cameras keeps each one's pictures on its own entity.
    """
    if camera.video is None or not clusters:
        return
    fps = video_fps(camera.video)
    spans = [
        (
            min(float(frame["time"]) for frame in cluster),
            max(float(frame["time"]) for frame in cluster),
        )
        for cluster in clusters
    ]

    def station_for(time: float) -> int:
        def distance(span: tuple[float, float]) -> float:
            low, high = span
            if low <= time <= high:
                return 0.0
            return min(abs(time - low), abs(time - high))

        return min(range(len(spans)), key=lambda index: distance(spans[index]))

    logged = 0
    for index, jpeg in iter_video_jpegs(camera.video, max_frames):
        time = index / fps
        rr.set_time("video_time", duration=time)
        rr.set_time("frame", sequence=index)
        rr.log(
            f"{entities[station_for(time)]}/image",
            rr.EncodedImage(contents=jpeg, media_type="image/jpeg"),
        )
        logged += 1
        if logged % 600 == 0:
            print(f"  {camera.name}: {logged} frame(s)…", file=sys.stderr)
    print(
        f"  {camera.name}: logged {logged} frame(s) of video at {fps:.0f}fps",
        file=sys.stderr,
    )


def log_station(camera: Camera) -> None:
    """The one pose fitted to every frame the camera stood still for.

    Drawn as a marker rather than replacing the per-frame transforms: a camera
    that really did pan has to keep showing as having panned, and this is the
    consensus it panned away from.
    """
    station = camera.report.get("station")
    if not station:
        return
    position = [float(value) for value in station["position"]]
    rr.log(
        f"world/stations/{camera.name}",
        rr.Points3D(
            [position],
            colors=[(255, 200, 90)],
            radii=0.08,
            labels=[
                f"{camera.name} station: {station['frames_used']} frame(s), "
                f"rms {float(station['rms_px']):.1f}px"
            ],
        ),
        static=True,
    )


def usable(frame: dict, min_height: float | None) -> bool:
    """Whether a frame's pose is physically possible.

    A single tag leaves a mirror ambiguity, and the mirrored branch is sometimes
    the one that fits best. It puts the camera below the carpet, which no camera
    ever was, so a height floor removes it without touching the good frames.
    """
    if min_height is None:
        return True
    return float(frame["position"][2]) >= min_height


def summarise_side(cluster: list[dict]) -> str:
    """One camera station's side of the field, for the terminal."""
    verdicts = [frame["side"] for frame in cluster if frame.get("side")]
    if not verdicts:
        return ""
    alliances: dict[str, int] = {}
    for side in verdicts:
        alliances[side["alliance"]] = alliances.get(side["alliance"], 0) + 1
    counts = ", ".join(
        f"{count} {alliance}" for alliance, count in sorted(alliances.items())
    )
    disagreeing = [side for side in verdicts if side.get("disagreement")]
    line = f"\n    side of the field: {counts}"
    for side in disagreeing[:3]:
        line += f"\n      read against the geometry: {side['disagreement']}"
    if len(disagreeing) > 3:
        line += f"\n      … and {len(disagreeing) - 3} more like it"
    return line


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
    log_station(camera)

    # A single station is just the camera; several get a name each, since they
    # are different cameras that happened to be measured in one file.
    entities = [
        (
            f"world/cameras/{camera.name}"
            if len(clusters) == 1
            else f"world/cameras/{camera.name}/cam{index + 1}"
        )
        for index in range(len(clusters))
    ]
    for entity, cluster in zip(entities, clusters):
        log_camera_static(entity, camera)
        log_frames(camera, cluster, entity)

    if with_video:
        log_video(camera, clusters, entities, max_frames)
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
        help="decode at most N frames of video per camera",
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
        station = report.get("station")
        if station:
            position = [float(value) for value in station["position"]]
            print(
                f"  fitted station pose: ({position[0]:.2f}, {position[1]:.2f}, "
                f"{position[2]:.2f})m from {station['frames_used']} frame(s), "
                f"{station['frames_rejected']} rejected, rms "
                f"{float(station['rms_px']):.1f}px"
            )
        if report.get("focal_source"):
            print(f"  {report['focal_source']}")
        for index, cluster in enumerate(stations):
            heights = [float(frame["position"][2]) for frame in cluster]
            span = max(heights) - min(heights)
            print(
                f"  station {index + 1}: {len(cluster)} frame(s), "
                f"height {sum(heights) / len(heights):.2f}m (±{span / 2:.2f}m)"
                f"{summarise_side(cluster)}"
            )
        if not stations:
            print(
                f"  note: {camera.name} has no solved frames to show", file=sys.stderr
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

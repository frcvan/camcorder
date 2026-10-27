# Working on camcorder

Pull FRC match stats and match video from The Blue Alliance for one team, split
each video into the individual camera views it contains, and estimate where the
camera was from the AprilTags in the picture.

## Commands

| what | command |
| --- | --- |
| build | `cargo build --release` |
| test everything | `cargo test` (62 tests: 53 unit, 3 CLI, 6 video pipeline) |
| one area | `cargo test --lib pose` / `cargo test --test views_pipeline` |
| lint | `cargo clippy --all-targets` |
| format | `rustfmt --edition 2021 <files you touched>` |
| type-check the Python | `python -m mypy scripts/rerun_poses.py` |
| pose of a video | `cargo run --release -- pose FILE.mp4 --tag-layout 2026-rebuilt-welded` |
| Rerun recording | `python scripts/rerun_poses.py poses/*.json --rrd out.rrd` |
| inspect a recording | `rerun rrd verify out.rrd`, `rerun rrd print out.rrd` |

External tools, run as child processes and required on `PATH`: `ffmpeg`,
`ffprobe`, `yt-dlp`. There is no mocking of them — tests build real videos with
`ffmpeg` and check real pixels.

Do **not** run `cargo fmt` across the repo: it is not rustfmt-clean, and a
whole-tree pass would bury the change in unrelated churn. Format only the files
you edited.

## Layout

```
src/config.rs    camcorder.toml + .env, defaults, CLI overrides
src/tba.rs       The Blue Alliance client and match models
src/pipeline.rs  fetch -> download -> split, and where files land
src/video.rs     ffmpeg: probe, scene cuts, divider detection, cropping
src/pose/        camera pose from AprilTags — self-contained subsystem
  types.rs       Vec3/Mat3/Quat/Intrinsics/CameraPose, no dependencies
  layout.rs      WPILib field layouts and tag corners
  frames.rs      greyscale frame sampling (owns all ffmpeg use)
  detect.rs      36h11 detection (the only file that knows kornia)
  solve.rs       planar init + Levenberg-Marquardt refinement
  report.rs      FramePose/PoseReport, table and JSON output
assets/tags/     WPILib field layouts, verbatim (see their README)
scripts/         rerun_poses.py
```

`src/pose/` deliberately shares nothing with the splitter except the ffmpeg
binary names, so it can be moved or lifted into its own binary by taking the
whole directory with it.

## Things that will bite you

**The tag frame is not the intuitive one.** WPILib's tag frame puts `+x` *out of
the tag face*, `+y` to the left when looking at the tag and `+z` up the tag. A
corner "right and up in the picture" is `-y, +z` locally. Getting this wrong does
not fail loudly: the solve still fits every corner and quietly returns a pose
balanced on the tag edge. `TagLayout::corners` is the single place this mapping
lives, and a test pins the corner order to what the detector reports.

**One tag cannot pin a pose.** A planar target has two mirror solutions that
reproject identically; from a single tag nothing in the picture separates them.
`pose_from_tag` returns one candidate — it normalises the homography's arbitrary
sign, and the frame loop prefers a pose above the carpet — which leaves the
mirror able to win, so some frames come out underground or far off-field. That
is a known limitation, not a bug in the layout data. Poses
from one tag are also scaled by the assumed `--fov`, so they can be
self-consistent with a sub-pixel fit and still be metres out. Trust frames with
two or more agreeing tags. `scripts/rerun_poses.py --min-height` drops the
below-carpet ones, and the `tags` column in the report says how many were used.

**Cameras in the stands are outside the field.** Positions like y = 12–14 m on
an 8 m-wide field are correct broadcast camera positions, not bugs. Underground
(z < 0) is the impossible one.

**A tiled frame is several cameras.** A broadcast frame holds an overview plus
two sub-views, so tags detected in one frame can belong to different physical
cameras — their poses are metres apart and it is not one camera teleporting.
Group by pose before treating them as one thing: that is what
`cluster_frames` in `rerun_poses.py` is for.

**In Rerun, the transform must be on the camera entity, not below the pinhole.**
A `Transform3D` logged at `<camera>/pose` while the `Pinhole` sits at `<camera>`
applies to the child only, leaving the camera stuck at the origin looking up.
Log `Transform3D` on the camera entity and `Pinhole` + `Image` on its child. The
report's quaternion is field-to-camera (it projects field points into the
picture) whereas a Rerun transform places camera axes in the world, so it is
conjugated; using it as-is draws a frustum pointing backwards.

**Field layouts are WPILib's, not ours.** `assets/tags/*.json` are byte-for-byte
copies; refresh them with the `curl` in `assets/tags/README.md` rather than
editing, and never hand-tune a tag position to make a result look better. Both
WPILib schemas parse: current (`field-dimensions`, `field-tags`) and legacy
(`field`, `tags`).

**Long jobs are long.** Scene detection and re-encoding decode every frame, so a
season is hours. Use `--event`, and run it in the background with its log
checked rather than in the foreground.

## Conventions

* Errors go through `anyhow` with `Context`; no `unwrap` outside tests.
* Public items get doc comments that explain *why*, matching the existing style.
* Tests state the physical fact they are protecting, not the mechanics. A test
  that cannot fail when the bug returns is worse than no test — check it by
  reverting the fix and watching it go red.
* Fixtures are built with `ffmpeg` inside the test, so a fixture that needs
  motion needs moving content, not a static graphic.
* Verification means running the thing: a number in a report or a `.rrd` is
  checked with `rerun rrd verify`/`print` and against the source footage.

## Output

`[output] dir`, or `--output-dir`, else `$TMPDIR/camcorder` — match JSON,
downloaded video, split views, and `poses/<file>.json`. `/output` is gitignored.

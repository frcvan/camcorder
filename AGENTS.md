# Working on camcorder

Pull FRC match stats and match video from The Blue Alliance for one team, split
each video into the individual camera views it contains, and estimate where the
camera was from the AprilTags in the picture.

## Commands

| what | command |
| --- | --- |
| build | `cargo build --release` |
| test everything | `cargo test` (102 tests: 88 unit, 3 CLI, 5 pose pipeline, 6 video pipeline) |
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

The Python tooling (`rerun-sdk`, `numpy`, `mypy`) lives in `.venv/` when there is
one; use `.venv/bin/python` and `.venv/bin/rerun` rather than the system ones.

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
  frames.rs      RGB frame sampling (owns all ffmpeg use) and luma from it
  enhance.rs     contrast stretch + unsharp, before the decoder sees a frame
  detect.rs      36h11 detection (the only file that knows kornia)
  corners.rs     sub-pixel corners: the tag's borders fitted and intersected
  solve.rs       planar init, Levenberg-Marquardt refinement, branch choice
  trust.rs       margin floors, tag-id voting, physical gates, confidence
  station.rs     focal length measured from the frames, one pose per station
  side.rs        alliance colour, field end, and the overview-pane rule
  report.rs      FramePose/PoseReport, table and JSON output
assets/tags/     WPILib field layouts, verbatim (see their README)
scripts/         rerun_poses.py
```

`src/pose/` deliberately shares nothing with the splitter except the ffmpeg
binary names, so it can be moved or lifted into its own binary by taking the
whole directory with it.

## Things that will bite you

**The tag frame is not the intuitive one.** WPILib's tag frame puts `+x` *out of
the tag face*, `+y` to the left of someone facing the tag and `+z` up the tag.
Standing in front of a tag, your left is its `+y`, so in the picture the tag's
`+y` runs to the *right* and `+z` up it: the top-right corner of the printed
pattern is `(+y, +z)`, and that is the order `TagLayout::corners` returns.
Getting the sign wrong does not fail loudly — every corner still fits a pose,
and the pose is the mirror of the real one, on the far side of the wall where no
camera could have read the tag. It *was* wrong: the only test of the order
projected the tag from behind it, so it agreed with the bug, and every pose was
a mirror. What pins it now is a fixture in `tests/pose_pipeline.rs` that draws a
tag by intersecting each pixel's ray with the tag plane from a known camera and
asserts the solve comes back to that camera's position.

**One tag is weak evidence, and its twin is impossible.** A single square has two
poses that reproject its corners to the same pixels, and with real corner noise
the wrong one often fits *better* — 0.06px against the truth's 0.39px on the
fixture that prompted this. The fit cannot choose between them, so physics does:
`trust::plausible` rejects a camera below the carpet, behind the face of a tag it
read, rolled more than 40°, or looking up at the field. When the homography's
answer is impossible, or sits at the other end of the field from the one the
colours name, `solve` restarts from a ring of seeds around the tag and keeps the
possible pose that fits within 1.5x of the best. What is left is honest: a
single-tag frame scores ~0.5 confidence (0.7 once the lens is measured) against
~1 for three tags, and `--min-confidence` drops the weak ones on request.

**The lens is measured, not assumed.** `--fov` is only a seed. Two tags a known
distance apart fix the focal length from the picture itself, and a frame that
fits better at some other focal length says so; the median across frames is used
and every frame is re-solved with it. A single tag can never do this — that is
exactly why its pose is scaled by whatever focal length you assumed. Two guards
keep the measurement honest, and both were added after real footage broke it:
**one frame never sets the lens** (a single pane frame holding two tags 36 cm
apart "measured" 50% below the guess, halving every distance in the video and
putting the camera at knee height inside the field), and **a minimum sitting on
the edge of the search range is not a measurement** — the search ran out of room,
so the number is the bound rather than anything the picture said. On broadcast
panes the tags are 15–17 px wide, which leaves the scale genuinely uncertain:
sweeping the focal from 400px to 1000px moves the camera 2–3 m with the
reprojection flat at 0.25–0.35 px. Trust the direction and the side reading from
that footage, not the metres.

**The camera stood still, so fit one pose to it.** `station` fits a single pose to
every frame's corners, dropping frames that disagree (a pan, or a bad solve). The
per-frame poses stay in the report: a camera that really panned has to keep
showing as having panned.

**Cameras in the stands are outside the field.** Positions like y = 12–14 m on
an 8 m-wide field are correct broadcast camera positions, not bugs. Underground
(z < 0) is the impossible one, and the gates reject it: a frame whose only solve
is impossible is reported in `missed` with the reason rather than in `solved`.

**The overview pane is named, not detected.** The splitter writes the top pane
of every segment as `segNN_view1.mp4`, and `pose` takes that for the wide
overview, which holds both alliances and so gets no side-of-field reading at all
(its frames carry none in the JSON). A `view1` that is really a sub-view, or a
hand-cut clip, only needs renaming to be read.

**Frame enhancement is on by default and is not cosmetic.** `pose` stretches the
contrast and sharpens each sampled frame before the detector sees it, which is
what makes a faint tag readable at all (a tag whose border is 28 luma levels
above its quiet zone is not decoded at all as the frames stand, and reads with a
decision margin of 105 once stretched). `--no-preprocess` measures
the frames as they are; use it when checking whether a change to `enhance` is
what moved a number.

**A tiled frame is several cameras.** A broadcast frame holds an overview plus
two sub-views, so tags detected in one frame can belong to different physical
cameras — their poses are metres apart and it is not one camera teleporting.
Group by pose before treating them as one thing: that is what
`cluster_frames` in `rerun_poses.py` is for.

**In Rerun, the transform must be on the camera entity, not below the pinhole.**
A `Transform3D` logged at `<camera>/pose` while the `Pinhole` sits at `<camera>`
applies to the child only, leaving the camera stuck at the origin looking up.
Log `Transform3D` on the camera entity and `Pinhole` + the frame on its child —
an `EncodedImage` of an ffmpeg-produced JPEG, since every frame of a match is
logged and raw frames would be 60GB. The
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

`output/demo.rrd` is a small recording for checking the viewer side of things: a
projected fixture of two tags in a 3 s clip, its pose report, and all 90 video
frames logged at 30fps. The video and report it was built from are in
`output/demo/`, so the recording can be rebuilt with
`.venv/bin/python scripts/rerun_poses.py output/demo/poses --rrd output/demo.rrd`
after a change to the script.

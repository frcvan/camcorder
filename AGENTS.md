# Working on camcorder

Pull FRC match stats and match video from The Blue Alliance for one team, split
each video into the individual camera views it contains, and estimate where the
camera was from the AprilTags in the picture.

## Commands

| what | command |
| --- | --- |
| build | `cargo build --release` |
| test everything | `cargo test` (134 tests: 118 unit, 3 CLI, 6 pose pipeline, 7 video pipeline) |
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
  model.rs       image-to-model line observations, from the field asset
  frames.rs      RGB frame sampling (owns all ffmpeg use) and luma from it
  enhance.rs     contrast stretch + unsharp, before the decoder sees a frame
  detect.rs      36h11 detection (the only file that knows kornia)
  corners.rs     sub-pixel corners: the tag's borders fitted and intersected
  solve.rs       planar init, Levenberg-Marquardt refinement, branch choice
  verify.rs      which of two admitted poses the picture backs, and why
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

**A square's corners cannot choose between two cameras, so the printed code and
the mount do.** The fit uses a tag's four *outer* corners, and those four points
are the corners of a square — and a square is the same square turned half round.
A lone tag therefore admits a second camera metres away: on the Orlando pane the
solver's pool held two cameras 9 m apart that fitted the same corners at 0.42px
and 0.46px (two frames fitted both answers to the same 0.29px), and the run
reported the one that was never there. `solve_all` now carries the near-ties out
instead of dropping them, and `verify` decides between them with two things the
fit cannot use. The *code inside the border*, which the
decode had already read, is not preserved by a half turn — and the *mount the tag
is bolted to* is not on the tag's plane, so it lands somewhere else for each
answer. Every solved frame carries both readings and the rival pose in its JSON;
the table says which answer won and on what. Two floors keep it honest: a check
that reads nothing (a mount behind a robot, a tag whose cells blur together)
decides nothing, and the fit's own answer stands. `verify.rs`'s own tests are
what hold this in place: a fixture whose twin fits *better* than the truth must
still lose to the mount, a tag too small to read must leave the question open
rather than pick, and the tag's cell grid read upside down must lose the frame.

**On broadcast panes the tag's code is not readable, and the mount decides.**
This one *was* measured, because it decides whether the code is used at all. The
Orlando pane's tags are 15-18px wide, and with the reading's floor taken out of
the way (`MIN_PATTERN_TAG_PX = 8` for one run) the two answers of the 28 frames
that hold a rival read 69-100% of the cells each — both high, every frame — and
the reading backed the pose the mount backs in only 20 of the 28. Two times in
three is not a decision, so the reading is refused below 24px and the report says
its pattern could not be read, leaving the mount to decide, which it does in 26
of those 28 frames (reported 73-85%, rival 44-77%). Above the floor the code
decides first and the mount second; in the fixture the right answer reads 89% of
the cells against its twin's 56%. The mount check counts only samples more than 15cm off the tag's plane,
because the parts of the mount that lie in that plane project to the same pixels
for both answers and would only add a constant to the score.

**The lens is fitted once per shot, and reported as a band.** `--fov` is only a
seed. `station::measure_lens` fits one shared focal length to every frame of the
busiest shot at once — one centre, a rotation per frame — and reports both the
fit and the band of lenses that explains the frames just as well. It is a grid
first and a golden section second, because the residual is not convex in focal
length; `k1` and a pixel aspect are offered next and kept only if they explain
the frames by 10%, and neither may sit on the edge of its range, where the number
is the clamp rather than the picture. `measured` means the band is worth less
than half a metre of camera position — the band's width times the distance to the
tags — because a ratio alone cannot say that: 5% of 1800 px is 10 cm at a metre
and 1.5 m at thirty. The guess is kept unless the fit pins the lens or explains
the frames by 25% better.

Two guards were added after real footage broke the old per-frame measurement, and
both survive in the joint fit: **one frame never sets the lens** (a single pane
frame holding two tags 36 cm apart "measured" 50% below the guess, halving every
distance in the video and putting the camera at knee height inside the field),
and **a minimum sitting on the edge of the search range is not a measurement**.
A shot whose frames disagree so badly that no lens reprojects them within
`--max-rms` has no lens at all, rather than a compromise that looks like one.

On the 2026 field the tags are clustered 35 cm apart, which at twenty metres is
no baseline. The Houston final's panes come back fitted but **not pinned** — one
overview pane allows 1057–1352 px, worth 2.3 m of camera position — and the
report says so in `focal_measured`, `focal_range_px` and `focal_source`. A single
tag seen from a fixed camera pins nothing; the same tag seen while the camera
panned does constrain the lens (the perspective across it changes and the centre
does not), which is one reason the fit is joint. Trust the direction and the side
reading from that footage, not the metres.

**The check is the model's own geometry, not a detector.** Nothing here
recognises a trench or a ladder in the picture. The model knows where the mount
of a tag it has *already decoded* is, projects it from a candidate pose, and asks
whether the picture has an edge there — a comparison between two poses, which is
a much smaller thing to ask of the field asset than identifying what is in a
frame. `structure.json` is the 2026 field's mesh, so on any other layout there is
nothing to check and `verify` says so instead of guessing.

**The camera stood still, so fit the camera, not each frame.** `station::fit`
fits one *pose* to every frame's corners, dropping frames that disagree (a pan,
or a bad solve). `station::fit_ptz` fits one shared *position* while letting each
frame keep its own orientation and focal length, because a fixed camera on a
broadcast stand pans and zooms without moving. The per-frame poses stay in the
report: a camera that really panned has to keep showing as having panned.

**Cameras in the stands are outside the field.** Positions like y = 12–14 m on
an 8 m-wide field are correct broadcast camera positions, not bugs. Underground
(z < 0) is the impossible one, and the gates reject it: a frame whose only solve
is impossible is reported in `missed` with the reason rather than in `solved`.

**A pane can hold more than one shot.** A file is one pane of the broadcast frame, and a pane is one camera, but it can be re-aimed between samples. `station::shots` splits the samples where a tag both of them see has moved a tenth of the frame across, or where samples more than 20 s apart share no tag to compare; the shared position is then fitted from the busiest shot, counted by the frames whose own solve agrees on a centre. A pan is not a new shot — the fit lets every frame keep its own rotation — while a re-aim is, and the frames of the other shot keep the poses their own solves gave them. On the Houston panes the whole file is one shot, which is the control: nothing there should move when this changes.

**One frame must not move the camera.** A lone tag solves two ways — the camera
in front of it, and a camera on the far side of the wall — and which one a frame
picks is noise. So the shared centre comes from *agreement*: the position with
the most frames within a metre of it, which on Orlando was 11 of 13 frames in one
pane and 15 of 19 in the other. Choosing by reprojection error instead is what
put both cameras in the middle of the field: two frames that agreed with each
other outvoted ten that agreed on the corner. A frame whose own solve landed
elsewhere is then **re-aimed** at the shared centre and re-solved there, because
its orientation was fitted from a position the camera was never at and from the
real centre can point straight away from the tag — where the residual is a flat
penalty with no slope to descend. Frames that still do not fit are counted in
`frames_unexplained` rather than kept quietly. A field-model round has to obey
the same rule: moving the centre metres for a hundredth of a pixel is not a
refinement.

**A low camera is not an impossible camera.** The Rerun script used to drop
frames below 0.5m, on the theory that the mirror branch is underground. Orlando's
panes stand at 0.28m and 0.43m, so that default discarded every frame that had
solved correctly and kept only the ones that had not. Only a pose below the
carpet is impossible; the viewer's floor now defaults to 0.

**A weak fit is reported, not withheld.** `pose` always reports the shared
position it fitted, together with `sigma_m`: the largest displacement of that
position which still explains the frames once each frame is free to re-aim and
re-zoom. When `sigma_m` reaches the probe's reach (2m), or the fit does not even
reproject the frames within `--max-rms`, the report marks the position
`ambiguous` and the table says it is a best estimate rather than a measurement.
This exists because an Orlando pane holding one 16px tag fits its frames to a
tenth of a pixel at more than one position: quoting either as *the* camera would
be the mistake. Read the position, then read the flag.

**The field model is evidence only where the picture agrees.** `model.rs`
matches the AdvantageScope asset's long red/blue alliance tape (identified by
colour) and the box outline of the structure each decoded tag is mounted on
(identified by tag id, so it can never be the wrong structure) — the second is
the only thing that breaks a lone tag's scale ambiguity, since the mount is not
on the tag's plane. A match is only recorded where the picture actually has an
edge, and matches are looked for afresh as the fit improves. On Orlando the
structure contributed nothing: its matches were coincidental and the fit
without them was kept, which the report shows as `model_frames: 0`.

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

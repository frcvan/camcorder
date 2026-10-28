# camcorder

Pulls FRC match data and match videos from [The Blue Alliance](https://www.thebluealliance.com/developer)
for one team, splits each video into the individual camera views it contains,
and estimates where each camera was standing from the AprilTags in the picture.

* the team number and season come from `camcorder.toml`
* the TBA read API key comes from `.env`
* videos are fetched with `yt-dlp` (YouTube) or straight over HTTP (TBA-hosted)
* each video is cut into its own views: several views tiled in one frame are
  cropped apart, and view switches over time are cut into separate files
* `pose` reads the AprilTags in a view and solves for the camera's position and
  orientation on the field

Everything below is run from the repository root.

**Contents:** [One match, end to end](#one-match-end-to-end) ·
[Requirements](#requirements) · [Setup](#setup) · [camcorder.toml](#camcordertoml) ·
[Commands](#commands) · [Output](#output) · [How views are found](#how-views-are-found) ·
[Camera pose from AprilTags](#camera-pose-from-apriltags) ·
[Looking at the result in Rerun](#looking-at-the-result-in-rerun) ·
[Troubleshooting](#troubleshooting) · [Notes](#notes)

## One match, end to end

The whole path — empty checkout to a pose report and a Rerun recording you can
fly around — for `2026cmptx_f1m1` as filmed at Houston. Substitute your own team
number, event key and match.

### 1. Build, and give it a key

```bash
cargo build --release
cp .env.example .env
$EDITOR .env                   # TBA_API_KEY=...   free from thebluealliance.com/account
```

The Rerun viewer is needed only at the very end:

```bash
python3 -m venv .venv
.venv/bin/pip install rerun-sdk numpy
```

### 2. Point it at the match

Edit `camcorder.toml` — three lines are what make it *this* event rather than a
season:

```toml
team = 386                  # your team number
year = 2026                 # season
event = "2026cmptx"         # TBA event key, from thebluealliance.com/event/<key>

[output]
dir = "output"              # everything lands here; default is $TMPDIR/camcorder
```

The same as flags, if you would rather not edit the file:

```bash
target/release/camcorder --team 386 --year 2026 --event 2026cmptx --output-dir output run
```

Leave `event` out entirely to pull the team's whole season.

### 3. Stats, then video

```bash
target/release/camcorder stats       # one row per match, plus the raw TBA JSON
target/release/camcorder download    # yt-dlp: one .mp4 per match
```

This prints the match list (`stats` writes `output/frc386/matches-2026.json`)
and then leaves:

```
output/frc386/videos/2026cmptx_f1m1.mp4
output/frc386/videos/2026cmptx_qm42.mp4
...
```

There is no per-match switch: `--event` narrows the run to the event and the
files are one per match, so delete the ones you do not want before splitting.
Matches TBA has no video for are listed with the reason and skipped.

Re-running is cheap — existing files are skipped unless you pass `--force` —
but scene detection still decodes every video, which is where the time goes.

### 4. Split the match into panes

A broadcast frame usually holds three cameras: one wide overview on top and two
sub-views below. Cut them apart:

```bash
target/release/camcorder views output/frc386/videos/2026cmptx_f1m1.mp4 \
    --out output/frc386/views/2026cmptx_f1m1
```

```
[views] 2026cmptx_f1m1.mp4 (1920x1080 · 60fps · 214.6s) -> 4 segment(s) · 6 view(s)
[views]   seg01         0.0s-3.1s   1 view   [single view]  1920x1080
[views]   seg02         3.1s-180.1s  3 views  [detected]  1920x704 + 942x338 + 942x338
[views]   seg03       180.1s-207.2s  1 view   [single view]  1920x1080
[views]   seg04       207.2s-214.6s  1 view   [single view]  1920x1080
...
output/frc386/views/2026cmptx_f1m1/seg02_view1.mp4   ← wide overview
output/frc386/views/2026cmptx_f1m1/seg02_view2.mp4   ← sub-view (left)
output/frc386/views/2026cmptx_f1m1/seg02_view3.mp4   ← sub-view (right)
```

`segNN` is the piece of the stream between scene changes, `viewN` is which pane
of that frame, in reading order. Naming matters downstream: **`segNN_view1.mp4`
is taken as the wide overview** and gets no side-of-field reading, because one
wide shot holds both alliances. A `view1` that is really a sub-view only needs
renaming to be read as one.

`target/release/camcorder run` does all three steps (stats, download, split) for
the whole event in one go, once you are happy with the settings.

### 5. Measure the camera in each pane

`pose` samples frames from one video file, decodes the AprilTags, and solves for
the camera that took them:

```bash
target/release/camcorder --output-dir output/frc386 pose \
    output/frc386/views/2026cmptx_f1m1/seg02_view2.mp4 \
    --tag-layout 2026-rebuilt-welded \
    --samples 20 \
    --window 0 15
```

`--tag-layout` is the one flag you must get right: `2026-rebuilt-welded` or
`-andymark` for this season, `2025-reefscape-*` for last, or a path to any
WPILib field layout JSON. `--window` matters while you are trying things out — a
15 s window takes seconds, a whole match takes minutes.

For every pane of the match into one directory:

```bash
mkdir -p output/frc386/poses/2026cmptx_f1m1
for f in output/frc386/views/2026cmptx_f1m1/seg02_view*.mp4; do
    target/release/camcorder --output-dir output/frc386 pose "$f" \
        --tag-layout 2026-rebuilt-welded \
        --samples 20 --window 0 15 \
        --json "output/frc386/poses/2026cmptx_f1m1/$(basename "${f%.mp4}").json"
done
```

### 6. Read the report

A real run on a Houston overview pane, output trimmed away to one block
(6 frames, which is why the lens band is wider than the 20-frame run below):

```
[pose] seg02_view1.mp4 (1920x704 · 5 tags seen · 2026-rebuilt-welded · tag 16.5cm · enhanced)
[pose] focal length fitted from 6 frame(s) but not pinned: 1229px (76°); anything from
       1081 to 1351px explains these frames as well, which is up to 2.0m of camera position
   time  tags  corners   rms  conf  camera x      y      z    yaw   pitch   roll  side
    1.2s     1        4   0.2  0.41    9.17  -4.46   3.17  102.6  -17.7   -3.1    --
    ...
    8.8s: another pose fits these corners as well, from (7.42, -4.81, 3.16)m: the picture
          did not pick one — 73% of the mount the tags stand on has an edge under it here
          against 70% from the pose that was reported
  fixed position (24 corner(s) in 6 frame(s), 3 agreeing on this centre, 3 re-aimed from
  elsewhere, no model structure matched, lens 1229px, itself only bounded to 1081–1351px,
  rms 0.3px, not pinned: a camera up to 0.9m away explains these frames too):
  (9.17, -4.46, 3.17)m
    the centre is the best fit, not a measurement: with this much footage a camera
    elsewhere fits nearly as well
overview pane: the side of the field was not read from it (one wide shot holds both alliances)
```

* **time** — where in the file the sample was taken.
* **tags / corners** — tags used, and corner points fitted (one tag = 4 corners).
* **rms** — reprojection error in pixels for that frame. Under ~1 px is a good
  fit; it does not by itself mean the pose is right.
* **conf** — `0..1`. One tag is ~0.5 because a single square fits two cameras;
  three tags pin it at ~1.
* **camera x/y/z** — metres in WPILib field coordinates. **Cameras in the stands
  are legitimately outside the field** (`y` of 12–14 m on an 8 m-wide field is a
  broadcast position, not a bug); below the carpet (`z < 0`) is impossible and
  is rejected.
* **yaw / pitch / roll** — degrees.
* **side** — which end of the field the alliance colours name, or `--`.

Then the station summary: one shared position for a camera that stood still.
Read it as two facts — **the position**, and **whether the position is a
measurement or a best guess**. `not pinned`, `ambiguous`, and "a camera up to
0.9m away explains these frames too" mean the footage cannot narrow it further:
quote the place, not the millimetres.

The JSON is the complete record — every frame pose, every rival pose and the
reading that chose between them, the lens band (`focal_px`, `focal_range_px`,
`k1`), `shots`, and `frames_unexplained`. Everything printed above is in there.

### 7. Look at it in Rerun

```bash
.venv/bin/python scripts/rerun_poses.py output/frc386/poses/2026cmptx_f1m1 \
    --rrd output/houston-f1m1.rrd \
    --max-frames 900

.venv/bin/rerun output/houston-f1m1.rrd
```

`--max-frames` is the file size: the recording is almost entirely video, so 900
frames (15 s at 60 fps) across three panes is already a couple of hundred MB.
`--no-video` gives a few MB of pure geometry. Check a recording without opening
it with `.venv/bin/rerun rrd verify FILE` and `.venv/bin/rerun rrd print FILE`.

### What that looks like on this footage

| pane | finding | reading |
| --- | --- | --- |
| `view1` (overview) | lens **measured** at 1372 px (70°), 17 % below the 60° guess; rms ~0.6 px | position and lens both usable |
| `view2` / `view3` | lens bounded only to a very wide band — metres to tens of metres of position | direction and side are fine, the **scale** is not |

The sub-views hold tags ~35 cm apart on screen, so the focal length — and with
it every distance — is genuinely undetermined there. The report says so in
words instead of printing a confident-looking coordinate.

## Requirements

| tool | why | check |
| --- | --- | --- |
| Rust (edition 2021) | builds the program | `cargo --version` |
| `ffmpeg` + `ffprobe` | probing, scene detection, cropping, pose frames | `ffmpeg -version` |
| `yt-dlp` | downloading YouTube match videos | `yt-dlp --version` |
| `rerun-sdk` + `numpy` in `.venv` | only for `scripts/rerun_poses.py` | `.venv/bin/python -c "import rerun"` |

`ffmpeg` and `yt-dlp` must be on `PATH`; they are run as child processes and
there is no bundled fallback.

## Setup

```bash
cp .env.example .env          # then put your free TBA read key in it
$EDITOR camcorder.toml        # team number, season, split settings
cargo build --release
target/release/camcorder      # full pipeline
```

A TBA read key is free from <https://www.thebluealliance.com/account>; without
it, everything except `views` and `pose` stops with *"TBA_API_KEY is not set"*.
`.env` is read next to the config file first, then from the current directory.

## camcorder.toml

```toml
team = 386           # FRC team number
year = 2025          # season to pull
# event = "2025arc"  # optional: one event instead of the whole season

[output]
dir = "/tmp/camcorder"   # default: a temp dir ($TMPDIR/camcorder)

[video]
mode = "both"           # "both" | "tiled" | "scenes"
layout = "auto"         # "auto" (broadcast layout) | "1x1" | "2x2" | ...
scene_threshold = 0.4   # lower = more cuts
scene_min_len = 2.0     # ignore cuts creating shorter clips than this
crf = 18                # x264 quality for the split files
preset = "veryfast"
```

`mode` decides what a video becomes: `tiled` crops the views out and leaves each
one whole, `scenes` splits on scene changes without cropping, `both` does one
then the other. Unknown keys are an error rather than being silently ignored, so
typos in this file are caught immediately. Anything in it can be overridden per
run with `--team`, `--year`, `--event` and `--output-dir`.

## Commands

| command | what it does |
| --- | --- |
| `camcorder run` (default) | stats, then download, then split every video |
| `camcorder stats` | pull and print match stats, write the raw JSON |
| `camcorder download` | download the match videos only |
| `camcorder views <FILE> [--out DIR]` | split one video file you already have |
| `camcorder pose <FILE> [--tag-layout …] [--samples …] [--window …]` | camera pose from the AprilTags in a video |
| `camcorder --help`, `camcorder <cmd> --help` | all flags |

Global flags, valid on any subcommand: `--config FILE`, `--team N`, `--year Y`,
`--event KEY`, `--output-dir DIR`, `--force`, `-v/--verbose`.

The stats table shows each match's alliance, result, score, opponent score and
video, then season averages and how much of the season has video and score
breakdowns. `--verbose` also dumps the raw score breakdown for your alliance.

Re-running never re-downloads a video or overwrites a view that already exists,
so it is safe to run repeatedly; `--force` redoes them.

## Output

Everything lands under the output directory, which defaults to
`$TMPDIR/camcorder`. Set `[output] dir` (or `--output-dir`) to a real path when
you want to keep the files.

```
output/frc386/
├── matches-2026.json              # the untouched TBA response
├── videos/2026cmptx_f1m1.mp4      # one file per match
├── views/2026cmptx_f1m1/
│   ├── seg01_view1.mp4            # title card, whole frame
│   ├── seg02_view1.mp4            # overview pane of the match
│   ├── seg02_view2.mp4            # bottom-left sub-view
│   ├── seg02_view3.mp4            # bottom-right sub-view
│   └── seg03_view1.mp4            # score screen, whole frame
└── poses/2026cmptx_f1m1/          # what `pose` measured, one JSON per pane
    ├── seg02_view1.json
    └── ...
```

A `pose` run with no `--json` writes to `<output dir>/poses/<stem>.json`. A
single-view video with no scene changes gives exactly one file,
`seg01_view1.mp4`.

## How views are found

FRC broadcasts come in two shapes: one full-frame feed, or one wide overview on
top of two sub-views. `layout = "auto"` looks for the second one and leaves
everything else as a single view:

1. average nine frames across the segment, then read off each row's colour as
   the median along that row, and each column's the same way, so picture
   content that varies along a line cancels out;
2. look for a horizontal divider between 55% and 74% of the height, and a
   vertical one between 42% and 58% of the width below it — the only places the
   broadcast puts them — where the strip's colour sits outside the picture on
   *both* sides;
3. check that the three panes that implies are broadcast proportions, so a
   field line or a score bar can never be mistaken for a divider.

That divider is drawn over the picture, not part of it, which is why it reads as
a colour outside the scene: the championship uses a thin bright line, the 2026
venues a wide themed bar, older district streams a magenta one. Because the
search only visits where the layout puts its dividers, it does not need to
tell a divider from a field marking in general — a general test is unreliable,
since a painted line looks exactly like a divider.

This covers every venue checked: 2025 Einstein (championship), 2025 Archimedes
district, 2026 Orlando, Tallahassee, South Florida and New England. The layout
of a match only starts after the title card, which the scene cut handles: the
card is one view, the tiled broadcast that follows is three.

### When detection will not help

Set `layout` explicitly in `camcorder.toml` when the frame holds views some
other way:

* a 2×2 or 1×2 grid of cameras, strip feeds, or anything not overview-on-top;
* views flush against each other with no divider drawn between them;
* a picture-in-picture inset or a scoreboard overlay (those are left alone, not
  cropped out).

Pinned layouts crop at exact fractions of the frame (`width / cols`,
`height / rows`), so they suit uniform grids. `views` prints the panes it used
and where they came from, so a wrong split is visible at a glance.

## Camera pose from AprilTags

```bash
# the short form: 2025 field, 60° lens seed, ten frames
target/release/camcorder pose match.mp4

# a 2026 field, a sub-view of a broadcast, 15 s in
target/release/camcorder --output-dir output pose seg02_view2.mp4 \
    --tag-layout 2026-rebuilt-welded --samples 20 --window 0 15
```

| flag | default | meaning |
| --- | --- | --- |
| `--tag-layout` | `2025-reefscape-welded` | bundled layout name, or a path to a WPILib layout JSON |
| `--fov` | 60 | horizontal field of view to seed the lens with, degrees |
| `--tag-size` | 0.1651 | tag edge length, metres (6.5 in) |
| `--samples` | 10 | frames to measure |
| `--window START END` | whole file | restrict sampling to these seconds |
| `--json FILE` | `<output>/poses/<stem>.json` | where the report goes |
| `--min-confidence` | 0 | drop solved frames below this confidence |
| `--no-preprocess` | off | measure frames as they are, unenhanced |
| `--force` | off | (global) redo regardless of existing files |

The table goes to stdout, the `[pose]` commentary to stderr, and the JSON to
disk. The tag maps are WPILib's own field layouts, copied verbatim into
`assets/tags/` — `2025-reefscape-welded`, `2025-reefscape-andymark`,
`2026-rebuilt-welded` and `2026-rebuilt-andymark` — or any layout file you
trust via `--tag-layout path/to/layout.json`. `assets/tags/README.md` records
where each came from and how to refresh them.

**The lens is measured, not assumed.** `--fov` is only a seed: frames holding
two or more tags a known distance apart fix the focal length from the picture
itself, and the median across frames becomes the lens every frame is re-solved
with. One frame never sets it, and a minimum sitting on the edge of the search
range is a bound rather than a measurement. The report also prints the band of
lenses that explains the frames just as well — `focal_px`, `focal_range_px`,
plus `k1` for radial distortion. Where that band is wide, every distance scales
with it, which is why a report can show a sub-pixel fit and still be 20 % too
far away. A frame holding a single tag can never measure the lens at all, which
is exactly why its distance depends on the seed you gave.

**One tag is weak evidence.** A single square admits two poses that reproject
its corners to the same pixels, so single-tag frames score ~0.5 confidence and
the report also names the rival pose. Three tags pin a pose at ~1. The `tags`
column says which frames are which. `CAMCORDER_DEBUG_POSE=1` prints every
detection (id, where it was, how big, decoder margin) and each frame's result,
which is the quick way to see why a video produced nothing.

This only works where the tags are big enough to read: a wide overview from the
far side of the venue puts them at a dozen pixels and nothing will decode them,
while a zoomed sub-view or a close-up camera usually will. It is a separate
subsystem (`src/pose/`) and shares nothing with the splitter except the ffmpeg
and ffprobe helpers.

## Looking at the result in Rerun

`scripts/rerun_poses.py` loads pose reports into [Rerun](https://rerun.io): the
field and its tags, each camera as a frustum with the frame it saw, its path,
and plots of position, height, heading, pitch, roll and reprojection error over
the timeline.

```bash
.venv/bin/python scripts/rerun_poses.py output/frc386/poses --rrd field.rrd
.venv/bin/python scripts/rerun_poses.py output/frc386/poses/*.json --spawn
.venv/bin/rerun field.rrd
```

A report is one video, which can hold more than one camera — a tiled broadcast
frame has an overview and two sub-views, and tags from different panes give
poses metres apart. Poses closer than `--station-tolerance` (1.5 m) are treated
as one fixed camera, and anything further is logged as its own station,
`world/cameras/<name>/cam1`, `cam2`, …, so each one stands still instead of
jumping. Frames whose camera lands below `--min-height` (0 m, i.e. only poses
below the carpet) are dropped with a note: a camera *can* legitimately sit low,
and the mirror solution a single tag allows is underground, not merely low.

| flag | effect |
| --- | --- |
| `--rrd FILE` | write a recording (without it, `poses.rrd` in the cwd) |
| `--spawn` | open the viewer when done |
| `--no-video` | poses only: small and fast |
| `--max-frames N` | at most N frames of video per camera — this is the file size |
| `--min-height M` | hide frames below M metres (default 0) |
| `--station-tolerance M` | poses within M metres are one fixed camera (default 1.5) |
| `--layout-dir DIR`, `--tag-size M` | must match what `pose` used |

It reads the report's `tag_layout` from `assets/tags/`, so tags line up with
what the solver used. `.venv/bin/rerun rrd verify FILE` checks a recording, and
`.venv/bin/rerun rrd print FILE` is the honest check that the transforms and
the pinholes are where you think they are: the camera transform belongs on the
camera entity with the `Pinhole` and the image on its child, or the frustum
draws at the origin pointing up.

The markers carry no sentences: the per-station text labels were removed because
they buried the 3D view. The reasons are printed when the recording is built and
every number is in the pose JSON.

## Troubleshooting

| symptom | cause / fix |
| --- | --- |
| `TBA_API_KEY is not set` | `.env` missing or not next to the config file |
| `error: reading config camcorder.toml` | not in the repo root, or `--config` points at a missing file |
| `frcN played M matches in YYYY, none at <event>` | wrong event key, or the team did not attend |
| `yt-dlp` failures on one match | listed with the reason and skipped; TBA often has no video for early quals |
| `yt-dlp: command not found` / `ffmpeg: No such file` | install it, or fix `PATH` |
| splitter writes one view and notes *"no tiled views found"* | the feed is not the usual tiling: pin `[video] layout` |
| `pose` finds 0 tags | wrong `--tag-layout`, a pane that is not on the field, or tags too small/blurred. Try another `--window`, more `--samples`, and `CAMCORDER_DEBUG_POSE=1` |
| every frame lands in `missed` | layout mismatch, or the video is not that season's field |
| positions mirror across the wall | almost always a tag-frame/layout mismatch, not a solve bug |
| only absurd positions survive | tags are being misread; raise `--min-confidence` and see what is left |
| `.rrd` is enormous | it is the video: cut `--max-frames`, shorten `--window`, or use `--no-video` |
| frusta at the origin pointing up | the transform was logged below the pinhole; use this repo's script |
| `rerun` not found | it is `.venv/bin/rerun`, not on `PATH` |

## Notes

* `TBA_BASE_URL` overrides the API host (useful for a mirror or a local stub).
* Scene detection and re-encoding decode every frame, so a full season is a
  long job — prefer `--event` when trying things out, and run it in the
  background with the log checked.
* Split files are re-encoded with x264; there is no stream-copy mode.
* `pose` does not detect robots, game pieces or field structures. It recognises
  AprilTags and fits a camera to them; the field model adds only the alliance
  tape and the mount of a tag it has already decoded, and only where the
  picture has an edge there. On some footage it contributes nothing
  (`model_frames: 0`) and the per-frame geometry stands alone.
* The engineering detail behind the pose solver — tag frame orientation, the two
  answers a single square admits, why the lens is measured and why one frame
  never sets it — is in `AGENTS.md`.

# camcorder

Pulls FRC match data and match videos from [The Blue Alliance](https://www.thebluealliance.com/developer)
for one team, then splits each video into the individual camera views it contains.

* the team number and season come from `camcorder.toml`
* the TBA read API key comes from `.env`
* videos are fetched with `yt-dlp` (YouTube) or straight over HTTP (TBA-hosted)
* each video is cut into its own views: several views tiled in one frame are
  cropped apart, and view switches over time are cut into separate files

## Requirements

| tool | why | check |
| --- | --- | --- |
| Rust (edition 2021) | builds the program | `cargo --version` |
| `ffmpeg` + `ffprobe` | probing, scene detection, cropping | `ffmpeg -version` |
| `yt-dlp` | downloading YouTube match videos | `yt-dlp --version` |

`ffmpeg` and `yt-dlp` must be on `PATH`; they are run as child processes.

## Setup

```sh
cp .env.example .env          # then put your free TBA read key in it
$EDITOR camcorder.toml        # team number, season, split settings
cargo run --release            # full pipeline
```

A TBA read key is free from <https://www.thebluealliance.com/account>. Anything
in `camcorder.toml` can be overridden per run with `--team`, `--year`, `--event`
and `--output-dir`.

### camcorder.toml

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

Unknown keys are an error rather than being silently ignored, so typos in this
file are caught immediately.

## Commands

| command | what it does |
| --- | --- |
| `camcorder run` (default) | stats, then download, then split every video |
| `camcorder stats` | pull and print match stats, write the raw JSON |
| `camcorder download` | download the match videos only |
| `camcorder views <FILE> [--out DIR]` | split one video file you already have |
| `camcorder pose <FILE> [--tag-layout …] [--fov …]` | camera pose from the AprilTags in a video |
| `camcorder --help` | all flags (`--force`, `--verbose`, `--config`, …) |

The stats table shows each match's alliance, result, score, opponent score and
video, then season averages and how much of the season has video and score
breakdowns. `--verbose` also dumps the raw score breakdown for your alliance.

Re-running never re-downloads a video or overwrites a view that already exists,
so it is safe to run repeatedly; `--force` redoes them. Scene detection still
scans each video on every run, which is where most of the time goes.

## Output

Everything lands under the output directory, which defaults to
`$TMPDIR/camcorder`. Set `[output] dir` (or `--output-dir`) to a real path when
you want to keep the files.

```
$TMPDIR/camcorder/frc386/
├── matches-2025.json              # the untouched TBA response
├── videos/2025arc_qm2.mp4         # one file per match
└── views/2025arc_qm2/
    ├── seg01_view1.mp4            # title card, whole frame
    ├── seg02_view1.mp4            # overview pane of the match
    ├── seg02_view2.mp4            # bottom-left sub-view
    ├── seg02_view3.mp4            # bottom-right sub-view
    └── seg03_view1.mp4            # score screen, whole frame
```

`segNN` is the piece of the stream between scene changes, `viewN` is which pane
of that frame it is, in reading order. A single-view video with no scene
changes gives exactly one file, `seg01_view1.mp4`.

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

```sh
camcorder pose match.mp4                        # 2025 field, 60° lens assumed
camcorder pose match.mp4 --tag-layout 2026-rebuilt-welded --fov 45 2>/dev/null
```

Samples frames from the video, reads the 36h11 tags in each one, and solves
where the camera must have been to see them that way: position in field
coordinates, yaw, pitch and roll, plus the reprojection error that says how
much to trust the number. The tabular result goes to stderr and a JSON report
to `<output dir>/poses/<file>.json` (or `--json`).

The tag maps are WPILib's own field layouts, copied verbatim into
`assets/tags/` — `2025-reefscape-welded`, `2025-reefscape-andymark`,
`2026-rebuilt-welded` and `2026-rebuilt-andymark` — or any layout file you
trust via `--tag-layout path/to/layout.json`. `assets/tags/README.md` records
where each came from and how to refresh them.

Two things the estimate cannot work out by itself: the lens, and the field the
video was shot on. `--fov` is the source camera's horizontal field of view
(60° is a reasonable broadcast guess), and getting it wrong scales the
distance — a pose can be self-consistent, with sub-pixel reprojection error,
and still be 20% too far away. `--tag-size` is 6.5 in for FRC tags.

### Looking at the result in Rerun

`scripts/rerun_poses.py` loads pose reports into [Rerun](https://rerun.io): the
field and its tags, each camera as a frustum with the frame it saw, its path,
and plots of position, height, heading, pitch, roll and reprojection error over
the timeline.

A report is one video, which can hold more than one camera — a tiled broadcast
frame has an overview and two sub-views, and tags from different panes give
poses metres apart. Poses closer than `--station-tolerance` (1.5 m) are treated
as one fixed camera, and anything further is logged as its own station,
`world/cameras/<name>/cam1`, `cam2`, …, so each one stands still instead of
jumping. Frames whose camera lands below `--min-height` (0.5 m) are dropped with
a note: those are the mirror solution a single tag allows, and no camera was
ever there.

```sh
pip install rerun-sdk
python scripts/rerun_poses.py $TMPDIR/camcorder/poses --spawn
python scripts/rerun_poses.py poses/*.json --rrd field.rrd      # write a file
```

It reads the report's `tag_layout` from `assets/tags/`, so tags line up with
what the solver used; `--layout-dir` and `--tag-size` cover a layout kept
elsewhere. `--no-video` leaves the frames out (much smaller and faster;
decoding frames is what takes the time), `--max-frames` caps how many come
from each.

One tag is enough for a pose but not enough to lean on: it fixes the direction
the camera looks far better than how far away it is, so a single far tag and a
wrong `--fov` can be metres out while reporting a sub-pixel fit. Frames with two
or more tags that agree are the ones worth trusting, and the `tags` column says
which is which. `CAMCORDER_DEBUG_POSE=1` prints every detection (id, where it
was, how big, decoder margin) and the result for each frame, which is the quick
way to see why a video produced nothing.

This only works where the tags are big enough to read: a wide overview from the
far side of the venue puts them at a dozen pixels and nothing will decode them,
while a zoomed sub-view or a close-up camera usually will. It is a separate
subsystem (`src/pose/`) and shares nothing with the splitter except the ffmpeg
and ffprobe helpers.

## Notes

* `TBA_BASE_URL` overrides the API host (useful for a mirror or a local stub).
* Scene detection and re-encoding decode every frame, so a full season is a
  long job — prefer `--event` or `--year` when trying things out.
* Split files are re-encoded with x264; there is no stream-copy mode.

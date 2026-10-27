//! Getting a match video onto disk and splitting it into its individual views.
//!
//! A "view" is one camera's picture inside the frame that gets published. Two
//! things make that awkward:
//!
//! * the views are tiled, and not always in a neat grid — a championship match
//!   is one wide overview on top with two subviews underneath, and
//! * the layout changes part way through, usually a full-frame title card or
//!   single-camera shot at the start and the tiled broadcast after the cut.
//!
//! So the splitter cuts the stream at scene changes first, then works out the
//! panes of *each* piece separately and crops every pane out of it. Panes come
//! from the broadcast layout itself: one wide overview on top, two sub-views
//! below, divided by a graphic that stands out from the picture on both sides.

use std::{
    fmt, fs,
    io::Write as _,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::config::{Layout, SplitMode};

pub const BIN_FFMPEG: &str = "ffmpeg";
pub const BIN_FFPROBE: &str = "ffprobe";
pub const BIN_YTDLP: &str = "yt-dlp";

/// Frames sampled and averaged per segment when looking for dividers.
const SEAM_SAMPLE_FRAMES: usize = 9;
/// Neighbour window sizes for the "stands out either side" test. Dividers run
/// from a 2px bright line to a 40px themed bar, so several sizes are needed.
const DIVIDER_SCALES: [usize; 5] = [4, 8, 16, 32, 64];
/// How far a divider must differ from the picture on both sides, in luma.
const DIVIDER_LUMA_STEP: f32 = 18.0;
/// The same as |dU| + |dV|, for coloured dividers.
const DIVIDER_CHROMA_STEP: f32 = 20.0;
/// No divider takes more than this share of the axis; anything wider is picture.
const MAX_DIVIDER_SHARE: f64 = 0.08;
/// Below this much change between samples the segment is a still graphic.
const MOTION_FLOOR: f32 = 1.0;
/// Where the horizontal divider sits, as a fraction of the frame height.
const TOP_SEAM_RANGE: (f64, f64) = (0.48, 0.78);
/// Where the divider inside the lower strip sits, as a fraction of the width.
const BOTTOM_SEAM_RANGE: (f64, f64) = (0.42, 0.58);
/// Aspect ratio range of the wide overview pane, measured from the layouts the
/// venues use (2.65–2.90). A single feed shot down the length of the field, with
/// its lines at the same place as a divider, is wider than this.
const OVERVIEW_ASPECT: (f64, f64) = (2.2, 3.1);
/// Aspect ratio range of a lower sub-view.
const SUBVIEW_ASPECT: (f64, f64) = (1.1, 3.0);
/// How closely the two sub-view widths must match each other.
const SUBVIEW_SYMMETRY: f64 = 0.96;
/// Share of sampled frame pairs that must show movement for the segment to be
/// live cameras rather than a drawn board.
const MOVING_SHARE_FLOOR: f32 = 0.75;
const MIN_FRAME_SIDE: u32 = 64;

// ---------------------------------------------------------------------------
// grids (the shorthand you can pin in camcorder.toml)
// ---------------------------------------------------------------------------

/// Rows x cols of equally sized views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Grid {
    pub rows: u32,
    pub cols: u32,
}

impl Grid {
    pub const SINGLE: Grid = Grid { rows: 1, cols: 1 };

    pub fn new(rows: u32, cols: u32) -> Result<Self> {
        if rows == 0 || cols == 0 {
            bail!("layout must have at least one row and one column, got {rows}x{cols}");
        }
        if rows > 8 || cols > 8 {
            bail!("layout {rows}x{cols} is too dense; at most 8x8 is supported");
        }
        Ok(Grid { rows, cols })
    }

    /// Parse `"2x2"`, `"2X2"` or `"2 x 2"`.
    pub fn parse(value: &str) -> Result<Self> {
        let cleaned = value.trim().to_ascii_lowercase().replace('*', "x");
        let (rows, cols) = cleaned
            .split_once('x')
            .with_context(|| format!("expected ROWSxCOLS, got {value:?}"))?;
        let rows: u32 = rows.trim().parse().context("bad row count")?;
        let cols: u32 = cols.trim().parse().context("bad column count")?;
        Grid::new(rows, cols)
    }

    pub fn tiles(self) -> u32 {
        self.rows * self.cols
    }

    /// Cell size for a frame that size: even numbers, centred when it does not divide.
    pub fn tile_dims(self, width: u32, height: u32) -> (u32, u32) {
        let tile_width = even(width / self.cols).max(2);
        let tile_height = even(height / self.rows).max(2);
        (tile_width, tile_height)
    }

    /// Where tile `index` (row-major) sits inside a frame that size.
    pub fn tile_rect(self, index: u32, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let (tile_width, tile_height) = self.tile_dims(width, height);
        let col = index % self.cols;
        let row = index / self.cols;
        let x = (col * tile_width + width.saturating_sub(self.cols * tile_width) / 2).min(width - 1);
        let y = (row * tile_height + height.saturating_sub(self.rows * tile_height) / 2)
            .min(height - 1);
        (tile_width, tile_height, x, y)
    }
}

impl fmt::Display for Grid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.rows, self.cols)
    }
}

fn even(value: u32) -> u32 {
    value - value % 2
}

// ---------------------------------------------------------------------------
// panes
// ---------------------------------------------------------------------------

/// One camera view, in pixels of the source frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pane {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Pane {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Pane {
            x,
            y,
            width,
            height,
        }
    }

    pub fn crop_filter(&self) -> String {
        format!(
            "crop={}:{}:{}:{}",
            self.width, self.height, self.x, self.y
        )
    }

    /// x264 needs even sizes, and cropping on an even boundary avoids extra chroma work.
    fn sanitized(self) -> Self {
        Pane {
            x: even(self.x),
            y: even(self.y),
            width: even(self.width).max(2),
            height: even(self.height).max(2),
        }
    }
}

/// The views tiled into one frame, in reading order (top to bottom, left to right).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameLayout {
    pub panes: Vec<Pane>,
}

impl FrameLayout {
    /// The whole frame as one view.
    pub fn single(width: u32, height: u32) -> Self {
        FrameLayout {
            panes: vec![Pane::new(0, 0, width, height)],
        }
    }

    /// A uniform rows x cols grid, sized exactly.
    pub fn from_grid(grid: Grid, width: u32, height: u32) -> Self {
        let panes = (0..grid.tiles())
            .map(|index| {
                let (w, h, x, y) = grid.tile_rect(index, width, height);
                Pane::new(x, y, w, h)
            })
            .collect();
        FrameLayout { panes }
    }

    pub fn views(&self) -> usize {
        self.panes.len()
    }

    /// `1920x664 + 954x412 + 962x412`
    pub fn describe(&self) -> String {
        self.panes
            .iter()
            .map(|pane| format!("{}x{}", pane.width, pane.height))
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

// ---------------------------------------------------------------------------
// ffprobe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    /// Seconds; 0.0 when the container does not report one.
    pub duration: f64,
    pub fps: f64,
}

#[derive(Debug, Deserialize)]
struct RawProbe {
    #[serde(default)]
    streams: Vec<RawStream>,
    #[serde(default)]
    format: RawFormat,
}

#[derive(Debug, Default, Deserialize)]
struct RawStream {
    width: Option<u32>,
    height: Option<u32>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    duration: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RawFormat {
    duration: Option<String>,
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let mut cmd = Command::new(BIN_FFPROBE);
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
        "-of",
        "json",
    ])
    .arg(path);
    let output = run_capture(&mut cmd, &format!("ffprobe {}", path.display()))?;
    let raw: RawProbe = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parsing ffprobe output for {}", path.display()))?;

    let stream = raw
        .streams
        .first()
        .with_context(|| format!("no video stream found in {}", path.display()))?;
    let width = stream.width.context("video stream has no width")?;
    let height = stream.height.context("video stream has no height")?;
    let duration = raw
        .format
        .duration
        .as_deref()
        .or(stream.duration.as_deref())
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.0);
    let fps = stream
        .r_frame_rate
        .as_deref()
        .and_then(parse_fraction)
        .or_else(|| stream.avg_frame_rate.as_deref().and_then(parse_fraction))
        .unwrap_or(0.0);

    Ok(VideoInfo {
        width,
        height,
        duration,
        fps,
    })
}

fn parse_fraction(value: &str) -> Option<f64> {
    let (numerator, denominator) = match value.split_once('/') {
        Some((n, d)) => (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?),
        None => (value.parse::<f64>().ok()?, 1.0),
    };
    (denominator != 0.0).then(|| numerator / denominator)
}

// ---------------------------------------------------------------------------
// scene detection
// ---------------------------------------------------------------------------

/// Timestamps (seconds) where the picture changes enough to count as a new shot.
pub fn scene_cuts(path: &Path, threshold: f32) -> Result<Vec<f64>> {
    let filter = format!("select='gt(scene,{threshold})',showinfo");
    let mut cmd = Command::new(BIN_FFMPEG);
    cmd.args(["-hide_banner", "-nostats", "-nostdin", "-i"])
        .arg(path)
        .args(["-filter:v", &filter, "-an", "-sn", "-f", "null", "-"])
        .stdout(Stdio::null());
    let output = run_capture(&mut cmd, "ffmpeg scene detection")?;
    Ok(parse_showinfo_times(&String::from_utf8_lossy(&output.stderr)))
}

/// Pull `pts_time:` values out of ffmpeg's `showinfo` log lines.
fn parse_showinfo_times(log: &str) -> Vec<f64> {
    let mut times = Vec::new();
    for line in log.lines() {
        if !line.contains("showinfo") {
            continue;
        }
        let Some(index) = line.find("pts_time:") else {
            continue;
        };
        let token: String = line[index + "pts_time:".len()..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        if let Ok(value) = token.parse::<f64>() {
            if value.is_finite() {
                times.push(value);
            }
        }
    }
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    times
}

/// One piece of the stream, in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub index: usize,
    pub start: f64,
    pub end: f64,
}

impl Segment {
    fn duration(&self) -> Option<f64> {
        let duration = self.end - self.start;
        (duration.is_finite() && duration > 0.001).then_some(duration)
    }
}

/// Turn cut points into segments, dropping cuts that would make a stub of a clip.
fn build_segments(cuts: &[f64], duration: f64, min_len: f64) -> Vec<Segment> {
    let mut starts: Vec<f64> = vec![0.0];
    for &cut in cuts {
        if !cut.is_finite() || cut <= 0.0 || (duration > 0.0 && cut >= duration) {
            continue;
        }
        let previous = *starts.last().expect("starts is never empty");
        if cut - previous >= min_len {
            starts.push(cut);
        }
    }
    if starts.len() > 1 && duration > 0.0 && duration - starts[starts.len() - 1] < min_len {
        starts.pop();
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, &start)| Segment {
            index,
            start,
            end: starts.get(index + 1).copied().unwrap_or(duration),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// layout detection
// ---------------------------------------------------------------------------

// FRC broadcasts use a small family of layouts: one full-frame feed (a single
// camera, or the title card before the match), or one wide overview above two
// sub-views. The championship divides those with a thin bright line and the
// regional venues with a wide themed bar, but both are graphics drawn over the
// picture, so in the time-averaged frame the divider stands out from the picture
// on both sides. Picture content — the white line down the middle of the field,
// a wall, a goal post — does not: it belongs to the scene, so its neighbours'
// colours bracket it.
//
// Detection therefore only tests the boundaries those layouts propose instead of
// scanning the frame for anything line-like. A field marking can look exactly
// like a divider, but it does not sit where the layout puts one, and the panes
// either side of it are not broadcast proportions.
//
// Profiles are medians along the divider's length: picture content that varies
// along it drops out, while a drawn line or bar survives.

/// Where the panes came from, so a run can say what it assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutSource {
    /// Pinned in the config file.
    Configured,
    /// Found by scanning the frame for dividers.
    Detected,
    /// The whole frame as one view.
    Single,
}

impl fmt::Display for LayoutSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LayoutSource::Configured => "configured",
            LayoutSource::Detected => "detected",
            LayoutSource::Single => "single view",
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detected {
    pub layout: FrameLayout,
    pub source: LayoutSource,
    pub samples: usize,
}

/// Work out how the frame is divided up during `window` (a segment's time range).
pub fn detect_layout(path: &Path, info: &VideoInfo, window: (f64, f64)) -> Result<Detected> {
    let single = Detected {
        layout: FrameLayout::single(info.width, info.height),
        source: LayoutSource::Single,
        samples: 0,
    };
    if info.width < MIN_FRAME_SIDE || info.height < MIN_FRAME_SIDE {
        return Ok(single);
    }

    let frames = sample_frames(path, info, window)?;
    if frames.len() < 3 {
        return Ok(Detected {
            samples: frames.len(),
            ..single
        });
    }
    let analysis = Analysis::new(&frames);
    let samples = frames.len();

    // A results screen is drawn from the same panels as the broadcast — a wide
    // band on top, two columns below — but it is a still graphic, not cameras.
    // A segment that *ends* on one (the celebration, then the board) has to be
    // caught too, so what counts is how much of the segment is moving, not how
    // much motion it averages.
    let moving = moving_share(&frames);
    if std::env::var_os("CAMCORDER_DEBUG_LAYOUT").is_some() {
        eprintln!("  [motion] {:.0}% of samples move", moving * 100.0);
    }
    if moving < MOVING_SHARE_FLOOR {
        return Ok(Detected { samples, ..single });
    }

    // One wide overview on top of two sub-views: the broadcast layout. Only
    // positions that would leave panes of broadcast proportions are considered,
    // so the strongest line in a single feed cannot drive the split.
    let rows = row_profile(&analysis);
    let Some(top) = find_divider(&rows, TOP_SEAM_RANGE, info.height, |y| {
        plausible(info.width, y, OVERVIEW_ASPECT)
    }) else {
        return Ok(Detected { samples, ..single });
    };

    let bottom_height = info.height - top.end;
    let lower = column_profile(&analysis, top.end as usize, info.height as usize);
    let Some(side) = find_divider(&lower, BOTTOM_SEAM_RANGE, info.width, |x| {
        plausible(x, bottom_height, SUBVIEW_ASPECT)
            && plausible(info.width - x, bottom_height, SUBVIEW_ASPECT)
    }) else {
        return Ok(Detected { samples, ..single });
    };
    if !plausible(info.width, top.start, OVERVIEW_ASPECT)
        || !plausible(side.start, bottom_height, SUBVIEW_ASPECT)
        || !plausible(info.width - side.end, bottom_height, SUBVIEW_ASPECT)
    {
        return Ok(Detected { samples, ..single });
    }
    // The two sub-views are cut from the same frame with the same margin, so
    // they come out the same width. A panel of a scoreboard splits lopsidedly.
    if !matched_widths(side.start, info.width - side.end) {
        return Ok(Detected { samples, ..single });
    }

    if std::env::var_os("CAMCORDER_DEBUG_LAYOUT").is_some() {
        eprintln!(
            "  [layout] top divider {}-{} (score {:.0}), side divider {}-{} (score {:.0})",
            top.start, top.end, top.score, side.start, side.end, side.score
        );
    }

    let layout = FrameLayout {
        panes: vec![
            Pane::new(0, 0, info.width, top.start).sanitized(),
            Pane::new(0, top.end, side.start, bottom_height).sanitized(),
            Pane::new(side.end, top.end, info.width - side.end, bottom_height).sanitized(),
        ],
    };
    Ok(Detected {
        layout,
        source: LayoutSource::Detected,
        samples,
    })
}

/// Is `width` by `height` a broadcast pane proportion rather than a sliver?
fn plausible(width: u32, height: u32, range: (f64, f64)) -> bool {
    let aspect = width as f64 / height.max(1) as f64;
    aspect >= range.0 && aspect <= range.1
}

/// Are two sub-view widths close enough to be a pair of camera feeds?
fn matched_widths(left: u32, right: u32) -> bool {
    let (left, right) = (left as f64, right as f64);
    let (narrow, wide) = if left < right {
        (left, right)
    } else {
        (right, left)
    };
    wide > 0.0 && narrow / wide >= SUBVIEW_SYMMETRY
}

/// A divider, as a half-open range of rows or columns, and how strongly it read.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Divider {
    start: u32,
    end: u32,
    score: f32,
}

/// The strongest divider within `range` (fractions of `length`) among the
/// positions `usable` accepts, widened to the full thickness of its band.
fn find_divider(
    profile: &Profile,
    range: (f64, f64),
    length: u32,
    usable: impl Fn(u32) -> bool,
) -> Option<Divider> {
    let last = profile.len().saturating_sub(2);
    let low = ((length as f64 * range.0) as usize).clamp(1, last.max(1));
    let high = ((length as f64 * range.1) as usize).min(last);
    if low >= high {
        return None;
    }

    let mut best = (0usize, 0.0f32);
    let mut taken = false;
    for position in low..high {
        if !usable(position as u32) {
            continue;
        }
        let score = divider_score(profile, position);
        if score > best.1 {
            best = (position, score);
            taken = true;
        }
    }
    if std::env::var_os("CAMCORDER_DEBUG_LAYOUT").is_some() {
        eprintln!(
            "  [divider] range {:.2}-{:.2} of {length}: best {} score {:.1}",
            range.0, range.1, best.0, best.1
        );
    }
    if !taken || best.1 < DIVIDER_LUMA_STEP.min(DIVIDER_CHROMA_STEP) {
        return None;
    }

    // Grow over the whole band, so the panes do not keep a stripe of the
    // divider. The picture either side is measured well clear of it, because a
    // 40px bar would otherwise be compared against itself, and growth follows
    // whichever signal found the divider, so a coloured bar does not bleed into
    // warm-looking picture below it.
    let far = DIVIDER_SCALES[DIVIDER_SCALES.len() - 1];
    let above = baseline(profile, best.0, far, true);
    let below = baseline(profile, best.0, far, false);
    let luma = beyond(profile.luma[best.0], above.0, below.0);
    let chroma = beyond(profile.u[best.0], above.1, below.1)
        + beyond(profile.v[best.0], above.2, below.2);
    let by_colour = chroma > luma;
    let peak = if by_colour { chroma } else { luma };
    let floor = peak * 0.4;
    let measure = |index: usize| {
        if by_colour {
            beyond(profile.u[index], above.1, below.1)
                + beyond(profile.v[index], above.2, below.2)
        } else {
            beyond(profile.luma[index], above.0, below.0)
        }
    };
    let max_width = ((length as f64 * MAX_DIVIDER_SHARE) as usize).max(8);
    let mut start = best.0;
    while start > 1 && best.0 + 1 - start < max_width && measure(start - 1) >= floor {
        start -= 1;
    }
    let mut end = best.0;
    while end + 2 < profile.len() && end + 1 - start < max_width && measure(end + 1) >= floor {
        end += 1;
    }
    if std::env::var_os("CAMCORDER_DEBUG_LAYOUT").is_some() {
        eprintln!(
            "  [divider] band {start}-{} peak {peak:.1} floor {floor:.1}",
            end + 1
        );
    }
    Some(Divider {
        start: start as u32,
        end: end as u32 + 1,
        score: best.1,
    })
}

/// The picture's colours well clear of `position`, as (luma, u, v).
fn baseline(profile: &Profile, position: usize, far: usize, above: bool) -> (f32, f32, f32) {
    let span = 32;
    let (low, high) = if above {
        (
            position.saturating_sub(far + span),
            position.saturating_sub(far),
        )
    } else {
        (
            (position + far + 1).min(profile.len()),
            (position + far + 1 + span).min(profile.len()),
        )
    };
    if low >= high {
        return (
            profile.luma[position],
            profile.u[position],
            profile.v[position],
        );
    }
    (
        median(&profile.luma[low..high]),
        median(&profile.u[low..high]),
        median(&profile.v[low..high]),
    )
}

/// How far the strip at `position` stands outside the picture on both sides,
/// measured at several window sizes so the divider's width need not be known.
fn divider_score(profile: &Profile, position: usize) -> f32 {
    let length = profile.len();
    let mut best = 0.0f32;
    for &scale in &DIVIDER_SCALES {
        if position < scale || position + scale >= length {
            continue;
        }
        let before = position - scale..position;
        let after = position + 1..position + 1 + scale;
        let luma = beyond(
            profile.luma[position],
            median(&profile.luma[before.clone()]),
            median(&profile.luma[after.clone()]),
        );
        let chroma = beyond(
            profile.u[position],
            median(&profile.u[before.clone()]),
            median(&profile.u[after.clone()]),
        ) + beyond(
            profile.v[position],
            median(&profile.v[before]),
            median(&profile.v[after]),
        );
        best = best.max(luma).max(chroma);
    }
    best
}

/// How far `value` sits outside the range spanned by `a` and `b`, or zero when
/// it lies between them (a blend of the two, i.e. a plain edge in the picture).
fn beyond(value: f32, a: f32, b: f32) -> f32 {
    let (low, high) = if a < b { (a, b) } else { (b, a) };
    if value > high {
        value - high
    } else if value < low {
        low - value
    } else {
        0.0
    }
}

/// A cross-section of the frame: one entry per row or per column, each the
/// median of the averaged frames along the other axis.
struct Profile {
    luma: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}

impl Profile {
    fn len(&self) -> usize {
        self.luma.len()
    }
}

/// One frame, at full resolution, in planar YUV.
#[derive(Clone)]
struct YuvFrame {
    width: usize,
    height: usize,
    chroma_width: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

/// Per-pixel average of the sampled frames.
struct Analysis {
    width: usize,
    height: usize,
    chroma_width: usize,
    mean_y: Vec<f32>,
    mean_u: Vec<f32>,
    mean_v: Vec<f32>,
}

impl Analysis {
    fn new(frames: &[YuvFrame]) -> Self {
        let first = &frames[0];
        let (width, height, chroma_width) = (first.width, first.height, first.chroma_width);
        let count = frames.len() as f32;

        let mut mean_y = vec![0.0f32; width * height];
        for frame in frames {
            for (slot, &value) in mean_y.iter_mut().zip(frame.y.iter()) {
                *slot += value as f32;
            }
        }
        for slot in &mut mean_y {
            *slot /= count;
        }

        let chroma_len = first.u.len().min(first.v.len());
        let mut mean_u = vec![0.0f32; chroma_len];
        let mut mean_v = vec![0.0f32; chroma_len];
        for frame in frames {
            for (slot, &value) in mean_u.iter_mut().zip(frame.u.iter()) {
                *slot += value as f32;
            }
            for (slot, &value) in mean_v.iter_mut().zip(frame.v.iter()) {
                *slot += value as f32;
            }
        }
        for slot in &mut mean_u {
            *slot /= count;
        }
        for slot in &mut mean_v {
            *slot /= count;
        }

        Analysis {
            width,
            height,
            chroma_width,
            mean_y,
            mean_u,
            mean_v,
        }
    }

    fn luma(&self, x: usize, y: usize) -> f32 {
        self.mean_y[y * self.width + x]
    }

    fn chroma(&self, x: usize, y: usize) -> (f32, f32) {
        let index = (y / 2) * self.chroma_width + x / 2;
        (self.mean_u[index], self.mean_v[index])
    }
}

/// Share of consecutive sample pairs in which the picture changes. Live footage
/// moves all the way through a segment; a results screen is still for most of
/// it, even when the segment opens on a replay.
fn moving_share(frames: &[YuvFrame]) -> f32 {
    let mut moving = 0usize;
    let mut pairs = 0usize;
    for window in frames.windows(2) {
        if let Some(change) = pair_change(&window[0], &window[1]) {
            pairs += 1;
            if change >= MOTION_FLOOR {
                moving += 1;
            }
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    moving as f32 / pairs as f32
}

/// Mean absolute luma change between two frames, sampled sparsely.
fn pair_change(before: &YuvFrame, after: &YuvFrame) -> Option<f32> {
    if before.y.len() != after.y.len() || before.y.is_empty() {
        return None;
    }
    let stride = (before.y.len() / 4096).max(1);
    let mut total = 0.0f64;
    let mut count = 0usize;
    for index in (0..before.y.len()).step_by(stride) {
        total += (before.y[index] as f32 - after.y[index] as f32).abs() as f64;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some((total / count as f64) as f32)
}

fn row_profile(analysis: &Analysis) -> Profile {
    let columns: Vec<usize> = (0..analysis.width).step_by(2).collect();
    let mut luma = Vec::with_capacity(analysis.height);
    let mut u = Vec::with_capacity(analysis.height);
    let mut v = Vec::with_capacity(analysis.height);
    for y in 0..analysis.height {
        let mut lumas = Vec::with_capacity(columns.len());
        let mut us = Vec::with_capacity(columns.len());
        let mut vs = Vec::with_capacity(columns.len());
        for &x in &columns {
            lumas.push(analysis.luma(x, y));
            let (cu, cv) = analysis.chroma(x, y);
            us.push(cu);
            vs.push(cv);
        }
        luma.push(median(&lumas));
        u.push(median(&us));
        v.push(median(&vs));
    }
    Profile { luma, u, v }
}

fn column_profile(analysis: &Analysis, y0: usize, y1: usize) -> Profile {
    let y1 = y1.min(analysis.height);
    let rows: Vec<usize> = (y0..y1.max(y0)).step_by(2).collect();
    let mut luma = Vec::with_capacity(analysis.width);
    let mut u = Vec::with_capacity(analysis.width);
    let mut v = Vec::with_capacity(analysis.width);
    for x in 0..analysis.width {
        let mut lumas = Vec::with_capacity(rows.len());
        let mut us = Vec::with_capacity(rows.len());
        let mut vs = Vec::with_capacity(rows.len());
        for &y in &rows {
            lumas.push(analysis.luma(x, y));
            let (cu, cv) = analysis.chroma(x, y);
            us.push(cu);
            vs.push(cv);
        }
        luma.push(median(&lumas));
        u.push(median(&us));
        v.push(median(&vs));
    }
    Profile { luma, u, v }
}


fn sample_frames(path: &Path, info: &VideoInfo, window: (f64, f64)) -> Result<Vec<YuvFrame>> {
    let times = sample_times(window, info.duration, SEAM_SAMPLE_FRAMES);
    let mut frames = Vec::new();
    for time in times {
        match extract_frame(path, time, info.width, info.height) {
            Ok(frame) => frames.push(frame),
            // The very end of a segment can sit past the last frame; that is not
            // fatal as long as the other samples came back.
            Err(error) if !frames.is_empty() => {
                eprintln!("  note: could not sample {time:.2}s ({error:#})");
            }
            Err(error) => return Err(error),
        }
    }
    Ok(frames)
}

fn extract_frame(path: &Path, time: f64, width: u32, height: u32) -> Result<YuvFrame> {
    let mut cmd = Command::new(BIN_FFMPEG);
    cmd.args([
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        &format!("{time:.3}"),
        "-i",
    ])
    .arg(path)
    .args([
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-vf",
        "format=yuv420p",
        "-f",
        "rawvideo",
        "-",
    ]);
    let output = run_capture(&mut cmd, "ffmpeg frame sampling")?;

    let (w, h) = (width as usize, height as usize);
    let chroma_width = (w + 1) / 2;
    let chroma_height = (h + 1) / 2;
    let luma_len = w * h;
    let chroma_len = chroma_width * chroma_height;
    let expected = luma_len + 2 * chroma_len;
    if output.stdout.len() != expected {
        bail!(
            "expected {expected} bytes of planar YUV, got {} (frame at {time:.3}s)",
            output.stdout.len()
        );
    }
    Ok(YuvFrame {
        width: w,
        height: h,
        chroma_width,
        y: output.stdout[..luma_len].to_vec(),
        u: output.stdout[luma_len..luma_len + chroma_len].to_vec(),
        v: output.stdout[luma_len + chroma_len..].to_vec(),
    })
}

fn sample_times(window: (f64, f64), duration: f64, count: usize) -> Vec<f64> {
    let start = window.0.max(0.0);
    let end = if window.1 > start {
        window.1
    } else if duration > start {
        duration
    } else {
        // Nothing known past the start: a single sample is all we can do.
        start
    };
    let span = end - start;
    if !(span > 0.5) {
        return vec![start];
    }
    (0..count)
        .map(|index| start + span * (index as f64 + 0.5) / count as f64)
        .collect()
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    }
}

// ---------------------------------------------------------------------------
// splitting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SplitOptions {
    pub mode: SplitMode,
    pub layout: Layout,
    pub scene_threshold: f32,
    pub scene_min_len: f64,
    pub crf: u8,
    pub preset: String,
    /// Re-cut files that already exist.
    pub force: bool,
}

/// What one segment of the stream turned out to contain.
#[derive(Debug, Clone)]
pub struct SegmentReport {
    pub segment: Segment,
    pub layout: FrameLayout,
    pub source: LayoutSource,
    pub outputs: Vec<PathBuf>,
    pub skipped: usize,
}

#[derive(Debug, Clone)]
pub struct SplitReport {
    pub info: VideoInfo,
    pub segments: Vec<SegmentReport>,
    pub outputs: Vec<PathBuf>,
    pub skipped: Vec<PathBuf>,
}

impl SplitReport {
    pub fn views(&self) -> usize {
        self.segments.iter().map(|report| report.layout.views()).sum()
    }

    /// True when every segment came out as one plain full-frame view.
    pub fn is_single_view_throughout(&self) -> bool {
        self.segments.iter().all(|report| report.layout.views() == 1)
    }
}

/// Cut `input` into one file per view per segment, inside `dest`.
pub fn split_views(input: &Path, dest: &Path, options: &SplitOptions) -> Result<SplitReport> {
    if !input.is_file() {
        bail!("no such video: {}", input.display());
    }
    let info = probe(input)?;

    let cuts = if options.mode.splits_scenes() {
        scene_cuts(input, options.scene_threshold)?
    } else {
        Vec::new()
    };
    let segments = build_segments(&cuts, info.duration, options.scene_min_len);

    fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;

    let mut report = SplitReport {
        info,
        segments: Vec::new(),
        outputs: Vec::new(),
        skipped: Vec::new(),
    };

    for segment in segments {
        let (layout, source) = if !options.mode.crops_views() {
            (
                FrameLayout::single(info.width, info.height),
                LayoutSource::Single,
            )
        } else {
            match options.layout {
                Layout::Grid(grid) => (
                    FrameLayout::from_grid(grid, info.width, info.height),
                    LayoutSource::Configured,
                ),
                Layout::Auto => {
                    let detected = detect_layout(input, &info, (segment.start, segment.end))?;
                    (detected.layout, detected.source)
                }
            }
        };

        let mut segment_report = SegmentReport {
            segment,
            layout,
            source,
            outputs: Vec::new(),
            skipped: 0,
        };

        for (index, pane) in segment_report.layout.panes.iter().enumerate() {
            let out = dest.join(format!(
                "seg{:02}_view{}.mp4",
                segment.index + 1,
                index + 1
            ));
            if out.is_file() && !options.force {
                report.skipped.push(out);
                segment_report.skipped += 1;
                continue;
            }
            cut_pane(
                input,
                &out,
                segment,
                segment_report.layout.views(),
                pane,
                options,
            )?;
            report.outputs.push(out.clone());
            segment_report.outputs.push(out);
        }

        report.segments.push(segment_report);
    }

    Ok(report)
}

fn cut_pane(
    input: &Path,
    out: &Path,
    segment: Segment,
    views: usize,
    pane: &Pane,
    options: &SplitOptions,
) -> Result<()> {
    let mut cmd = Command::new(BIN_FFMPEG);
    cmd.args([
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        &format!("{:.3}", segment.start),
        "-i",
    ])
    .arg(input);

    if let Some(duration) = segment.duration() {
        cmd.args(["-t", &format!("{duration:.3}")]);
    }
    if views > 1 {
        cmd.args(["-vf", &pane.crop_filter()]);
    }
    cmd.args([
        "-c:v",
        "libx264",
        "-preset",
        &options.preset,
        "-crf",
        &options.crf.to_string(),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
    ])
    .arg(out);

    run_capture(
        &mut cmd,
        &format!("ffmpeg splitting {} into {}", input.display(), out.display()),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// downloading
// ---------------------------------------------------------------------------

/// Download a YouTube video id with yt-dlp into `dest`.
pub fn download_youtube(video_key: &str, dest: &Path) -> Result<PathBuf> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let url = format!("https://www.youtube.com/watch?v={video_key}");
    let mut cmd = Command::new(BIN_YTDLP);
    cmd.args([
        "--no-playlist",
        "--no-progress",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format",
        "mp4",
        "-o",
    ])
    .arg(dest)
    .arg(&url);
    run_capture(&mut cmd, &format!("yt-dlp {url}"))?;

    resolve_downloaded(dest).with_context(|| format!("yt-dlp did not produce {}", dest.display()))
}

/// Download a file URL straight over HTTP.
pub fn download_direct(url: &str, dest: &Path) -> Result<PathBuf> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(1800))
        .build()
        .context("building the HTTP client")?;
    let mut response = client.get(url).send().with_context(|| format!("GET {url}"))?;
    if !response.status().is_success() {
        bail!("downloading {url} failed: {}", response.status());
    }
    let mut file = fs::File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    std::io::copy(&mut response, &mut file).with_context(|| format!("writing {}", dest.display()))?;
    file.flush()?;
    Ok(dest.to_path_buf())
}

/// yt-dlp writes the container it found; accept a different extension at the same stem.
fn resolve_downloaded(dest: &Path) -> Result<PathBuf> {
    if dest.is_file() {
        return Ok(dest.to_path_buf());
    }
    let stem = dest.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    if let Some(parent) = dest.parent() {
        for entry in fs::read_dir(parent)?.flatten() {
            let path = entry.path();
            if path.file_stem().and_then(|s| s.to_str()) == Some(stem)
                && path.extension().is_some()
                && !path.to_string_lossy().ends_with(".part")
            {
                return Ok(path);
            }
        }
    }
    bail!("no downloaded file next to {}", dest.display())
}

// ---------------------------------------------------------------------------
// process helpers
// ---------------------------------------------------------------------------

fn run_capture(cmd: &mut Command, what: &str) -> Result<std::process::Output> {
    let output = cmd
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("could not run {what} — is it installed and on PATH?"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(8).collect();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
        bail!("{what} failed ({})\n{tail}", output.status);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_parses_and_divides_frames() {
        assert_eq!(Grid::parse("2x2").unwrap(), Grid { rows: 2, cols: 2 });
        assert_eq!(Grid::parse(" 2 X 3 ").unwrap(), Grid { rows: 2, cols: 3 });
        assert!(Grid::parse("2").is_err());
        assert!(Grid::parse("0x2").is_err());

        let grid = Grid { rows: 2, cols: 2 };
        assert_eq!(grid.tiles(), 4);
        assert_eq!(grid.tile_dims(1920, 1080), (960, 540));
        assert_eq!(grid.tile_rect(0, 1920, 1080), (960, 540, 0, 0));
        assert_eq!(grid.tile_rect(3, 1920, 1080), (960, 540, 960, 540));
    }

    #[test]
    fn odd_frames_are_rounded_to_even_tiles() {
        let grid = Grid { rows: 1, cols: 2 };
        // 1921 wide: 960+960 with the spare pixel dropped from the middle.
        assert_eq!(grid.tile_dims(1921, 1081), (960, 1080));
        assert_eq!(grid.tile_rect(0, 1921, 1081), (960, 1080, 0, 0));
        assert_eq!(grid.tile_rect(1, 1921, 1081), (960, 1080, 960, 0));
    }

    #[test]
    fn a_grid_layout_is_exact() {
        let layout = FrameLayout::from_grid(Grid::new(2, 2).unwrap(), 1920, 1080);
        assert_eq!(layout.views(), 4);
        assert_eq!(layout.panes[0], Pane::new(0, 0, 960, 540));
        assert_eq!(layout.panes[3], Pane::new(960, 540, 960, 540));
        assert_eq!(layout.panes[3].crop_filter(), "crop=960:540:960:540");
        assert_eq!(layout.describe(), "960x540 + 960x540 + 960x540 + 960x540");
    }

    #[test]
    fn panes_are_even_sized_for_the_encoder() {
        assert_eq!(
            Pane::new(957, 663, 963, 417).sanitized(),
            Pane::new(956, 662, 962, 416)
        );
    }

    #[test]
    fn parses_fractions_and_showinfo_times() {
        assert_eq!(parse_fraction("30000/1001").unwrap().round(), 30.0);
        assert_eq!(parse_fraction("25").unwrap(), 25.0);
        assert_eq!(parse_fraction("25/0"), None);

        let log = "\
[Parsed_showinfo_1 @ 0x55] n:   0 pts:      0 pts_time:0       duration:1
[Parsed_showinfo_1 @ 0x55] n:   1 pts:     30 pts_time:1.001     duration:1
[Parsed_showinfo_1 @ 0x55] n:   2 pts:     60 pts_time:12.5      duration:1
[Parsed_showinfo_1 @ 0x55] n:   3 pts:     90 pts_time:12.5      duration:1
some other ffmpeg chatter pts_time:99";
        assert_eq!(parse_showinfo_times(log), vec![0.0, 1.001, 12.5]);
    }

    #[test]
    fn segments_respect_the_minimum_length() {
        let segments = build_segments(&[10.0, 10.4, 20.0, 100.0], 30.0, 2.0);
        // 10.4 is too close to 10.0, 100 is past the end, and the tail is long enough.
        assert_eq!(
            segments,
            vec![
                Segment { index: 0, start: 0.0, end: 10.0 },
                Segment { index: 1, start: 10.0, end: 20.0 },
                Segment { index: 2, start: 20.0, end: 30.0 },
            ]
        );
    }

    #[test]
    fn short_tail_is_merged_into_the_previous_segment() {
        let segments = build_segments(&[28.5], 30.0, 2.0);
        assert_eq!(segments, vec![Segment { index: 0, start: 0.0, end: 30.0 }]);
    }

    #[test]
    fn no_cuts_means_one_segment_covering_everything() {
        let segments = build_segments(&[], 12.0, 2.0);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].duration(), Some(12.0));

        // Unknown duration: copy to the end.
        let segments = build_segments(&[], 0.0, 2.0);
        assert_eq!(segments[0].duration(), None);
    }

    /// A profile of the given luma with neutral chroma.
    fn profile_of(luma: Vec<f32>) -> Profile {
        let length = luma.len();
        Profile {
            luma,
            u: vec![128.0; length],
            v: vec![128.0; length],
        }
    }

    #[test]
    fn a_bright_line_is_a_divider() {
        // The championship separates the panes with a thin bright line.
        let mut luma = vec![100.0; 400];
        for slot in luma.iter_mut().skip(258).take(3) {
            *slot = 160.0;
        }
        let divider =
            find_divider(&profile_of(luma), (0.55, 0.74), 400, |_| true).expect("a divider");
        assert!((258..=262).contains(&divider.start), "got {divider:?}");
        assert!(divider.end > divider.start, "got {divider:?}");
    }

    #[test]
    fn a_wide_themed_bar_is_taken_in_full() {
        // The regional venues use a ~30px bar; no strip of it may stay in a pane.
        let mut luma = vec![100.0; 400];
        for slot in luma.iter_mut().skip(255).take(30) {
            *slot = 150.0;
        }
        let divider =
            find_divider(&profile_of(luma), (0.55, 0.74), 400, |_| true).expect("a divider");
        assert!(divider.start <= 255, "got {divider:?}");
        assert!(divider.end >= 285, "got {divider:?}");
    }

    #[test]
    fn a_coloured_bar_counts_even_at_the_same_brightness() {
        // The magenta bar of the 2025 district streams is flat in luma.
        let (mut u, mut v) = (vec![128.0; 400], vec![128.0; 400]);
        for index in 258..266 {
            u[index] = 96.0;
            v[index] = 200.0;
        }
        let profile = Profile {
            luma: vec![100.0; 400],
            u,
            v,
        };
        assert!(find_divider(&profile, (0.55, 0.74), 400, |_| true).is_some());
    }

    #[test]
    fn picture_content_is_not_a_divider() {
        // A gradual change: every value lies between its neighbours.
        let ramp: Vec<f32> = (0..400).map(|i| 100.0 + i as f32 * 0.05).collect();
        assert!(find_divider(&profile_of(ramp), (0.0, 1.0), 400, |_| true).is_none());

        // A plain step is a picture edge, not a band: the value on one side is
        // inside the range the other side already covers.
        let mut stepped = vec![100.0; 400];
        for slot in stepped.iter_mut().skip(260) {
            *slot = 140.0;
        }
        assert!(find_divider(&profile_of(stepped), (0.0, 1.0), 400, |_| true).is_none());
    }

    fn grey_frame(width: usize, height: usize, level: u8) -> YuvFrame {
        let chroma = (width / 2) * (height / 2);
        YuvFrame {
            width,
            height,
            chroma_width: width / 2,
            y: vec![level; width * height],
            u: vec![128; chroma],
            v: vec![128; chroma],
        }
    }

    #[test]
    fn a_results_screen_does_not_count_as_live() {
        // Four moving samples then four still ones: the celebration, then the board.
        let mut frames = Vec::new();
        for index in 0..9 {
            let level = if index < 4 { (index * 40) as u8 } else { 200 };
            frames.push(grey_frame(64, 64, level));
        }
        assert_eq!(moving_share(&frames), 0.5);

        // A live segment moves every time.
        let live: Vec<YuvFrame> = (0..9)
            .map(|index| grey_frame(64, 64, (index * 20) as u8))
            .collect();
        assert_eq!(moving_share(&live), 1.0);

        // A title card or a board held for the whole segment never moves.
        let still = vec![grey_frame(64, 64, 90); 9];
        assert_eq!(moving_share(&still), 0.0);
    }

    #[test]
    fn sub_views_must_be_the_same_width() {
        assert!(matched_widths(952, 954));
        assert!(matched_widths(942, 944));
        assert!(!matched_widths(906, 1006));
        assert!(!matched_widths(932, 984));
        assert!(!matched_widths(0, 0));
    }

    #[test]
    fn pane_proportions_must_look_like_camera_views() {
        assert!(plausible(1920, 663, OVERVIEW_ASPECT));
        assert!(plausible(954, 414, SUBVIEW_ASPECT));
        // Halving the frame is not a broadcast layout.
        assert!(!plausible(960, 1080, SUBVIEW_ASPECT));
        assert!(!plausible(1920, 540, OVERVIEW_ASPECT));
        assert!(!plausible(100, 0, SUBVIEW_ASPECT));
    }

    #[test]
    fn sample_times_stay_inside_the_segment() {
        let times = sample_times((10.0, 20.0), 100.0, 5);
        assert_eq!(times.len(), 5);
        assert!(times.iter().all(|t| *t > 10.0 && *t < 20.0));
        // Unknown duration and no end: fall back to a single sample.
        assert_eq!(sample_times((0.0, 0.0), 0.0, 5), vec![0.0]);
    }
}

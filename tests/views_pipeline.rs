//! End-to-end tests for the split pipeline.
//!
//! The fixtures are synthetic videos built by ffmpeg itself, with known pane
//! geometry and known colours, so checking actual pixel colours at actual
//! coordinates is what shows the crops land on the right part of the frame.
//!
//! Four shapes matter:
//! * a championship broadcast — one wide overview on top, two subviews below,
//!   separated by thin *bright* lines — that only starts 4s into the stream
//!   after a full-frame title card, detected with no configuration,
//! * a single-view clip that must come out untouched,
//! * a uniform 2x2 grid separated by black gutters, and
//! * two views side by side, neither of them 16:9.
//!
//! The last two are not broadcast layouts, so they are cropped from a pinned
//! `layout` in the config rather than detected.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use camcorder::{
    config::{Layout, SplitMode},
    video::{self, Grid, LayoutSource, Pane, SplitOptions},
};

const TILE_W: u32 = 320;
const TILE_H: u32 = 180;
/// Black gutter kept to the right and below each tile.
const GUTTER: u32 = 12;
const FRAME_W: u32 = 2 * (TILE_W + GUTTER);
const FRAME_H: u32 = 2 * (TILE_H + GUTTER);

/// Championship layout: 1920x664 overview, a 4px line, then 954+4+962 across 412.
const CHAMP_W: u32 = 1920;
const CHAMP_OVERVIEW_H: u32 = 664;
const CHAMP_LINE: u32 = 4;
const CHAMP_LEFT_W: u32 = 954;
const CHAMP_RIGHT_W: u32 = 962;
const CHAMP_SUBVIEW_H: u32 = 412;
const CHAMP_INTRO_SECS: u32 = 4;

struct Temp {
    dir: tempfile::TempDir,
}

impl Temp {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().expect("temp dir"),
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}

fn ffmpeg(args: &[&str]) -> std::process::Output {
    let output = Command::new(video::BIN_FFMPEG)
        .args(args)
        .output()
        .expect("ffmpeg is required to build and check test fixtures");
    assert!(
        output.status.success(),
        "ffmpeg {args:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// 4 seconds, 2x2 tiled with black gutters, cut in half by a scene change at 2s.
///
/// Scene 1 is four flat colours (red, green, blue, yellow top-left to
/// bottom-right); scene 2 is detailed test patterns.
fn build_tiled_video(dest: &Path) {
    let tile = format!("s={TILE_W}x{TILE_H}:d=2:r=30");
    let mut args: Vec<String> = vec!["-y".into(), "-loglevel".into(), "error".into()];
    for input in [
        format!("color=c=red:{tile}"),
        format!("color=c=green:{tile}"),
        format!("color=c=blue:{tile}"),
        format!("color=c=yellow:{tile}"),
        format!("testsrc2={tile}"),
        format!("testsrc2={tile},hue=h=120"),
        format!("testsrc2={tile},hue=h=240"),
        format!("testsrc2={tile},negate"),
    ] {
        args.push("-f".into());
        args.push("lavfi".into());
        args.push("-i".into());
        args.push(input);
    }

    let width = TILE_W + GUTTER;
    let height = TILE_H + GUTTER;
    let pad = format!("pad={width}:{height}:0:0:black");
    let tiles: Vec<String> = (0..8).map(|i| format!("[{i}]{pad}[a{i}]")).collect();
    let graph = format!(
        "{};[a0][a1]hstack[t1];[a2][a3]hstack[b1];[t1][b1]vstack[s1];\
         [a4][a5]hstack[t2];[a6][a7]hstack[b2];[t2][b2]vstack[s2];\
         [s1][s2]concat=n=2:v=1:a=0[out]",
        tiles.join(";")
    );
    let path = dest.to_str().expect("utf-8 path");
    args.extend(
        [
            "-filter_complex",
            graph.as_str(),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            path,
        ]
        .map(str::to_string),
    );

    let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
    ffmpeg(&borrowed);
}

/// 8 seconds: a full-frame title card, then a championship broadcast.
///
/// The title card is dark on purpose: cutting from a bright flat fill to the
/// tiled frame is only a modest change, which puts the scene detector's decision
/// near its threshold. A dark card with a moving patch on it is unambiguous.
fn build_championship_video(dest: &Path) {
    let clip = format!("d={CHAMP_INTRO_SECS}:r=30");
    let path = dest.to_str().expect("utf-8 path");
    // Each pane is a flat colour with a patch of moving test pattern overlaid:
    // live picture is what tells a camera layout from a results screen, and the
    // flat colour keeps the pane identifiable by pixel colour. The patches stay
    // clear of the spots the test samples.
    let graph = format!(
        "[6]split=4[q0][p1][p2][p3];\
         [0][q0]overlay=x=(W-140)/2:y=(H-100)/2[intro];\
         [1][p1]overlay=x=mod(t*220\\,W-140):y=60[top];\
         [2][p2]overlay=x=mod(t*170\\,W-140):y=20[left];\
         [3][p3]overlay=x=mod(t*170\\,W-140):y=20[right];\
         [4]setsar=1[hline];[5]setsar=1[vline];\
         [top][hline]vstack[top_row];\
         [left][vline][right]hstack=inputs=3[bottom_row];\
         [top_row][bottom_row]vstack[tiled];\
         [intro][tiled]concat=n=2:v=1:a=0[out]"
    );
    ffmpeg(&[
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        &format!("color=c=black:s={CHAMP_W}x1080:{clip}"),
        "-f",
        "lavfi",
        "-i",
        &format!("color=c=red:s={CHAMP_W}x{CHAMP_OVERVIEW_H}:{clip}"),
        "-f",
        "lavfi",
        "-i",
        &format!("color=c=green:s={CHAMP_LEFT_W}x{CHAMP_SUBVIEW_H}:{clip}"),
        "-f",
        "lavfi",
        "-i",
        &format!("color=c=blue:s={CHAMP_RIGHT_W}x{CHAMP_SUBVIEW_H}:{clip}"),
        "-f",
        "lavfi",
        "-i",
        &format!("color=c=white:s={CHAMP_W}x{CHAMP_LINE}:{clip}"),
        "-f",
        "lavfi",
        "-i",
        &format!("color=c=white:s={CHAMP_LINE}x{CHAMP_SUBVIEW_H}:{clip}"),
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=140x100:r=30:d=4",
        "-filter_complex",
        &graph,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        path,
    ]);
}

/// A plain single-view 16:9 clip: detailed, always moving, no straight edges
/// and no flat patches, i.e. as close to camera footage as lavfi gets.
fn build_single_view_video(dest: &Path) {
    let path = dest.to_str().expect("utf-8 path");
    ffmpeg(&[
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "mandelbrot=s=1280x720:r=30",
        "-t",
        "3",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        path,
    ]);
}

/// Two 640x720 views side by side: red on the left, green on the right.
fn build_side_by_side_video(dest: &Path) {
    let path = dest.to_str().expect("utf-8 path");
    ffmpeg(&[
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=640x720:d=2:r=30",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=640x720:d=2:r=30",
        "-filter_complex",
        "[0][1]hstack",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        path,
    ]);
}

fn options(mode: SplitMode, layout: Layout) -> SplitOptions {
    SplitOptions {
        mode,
        layout,
        scene_threshold: 0.4,
        scene_min_len: 1.0,
        crf: 28,
        preset: "ultrafast".to_string(),
        force: true,
    }
}

/// Assert a pixel is lit on the expected channels and quiet on the others.
///
/// Channel *dominance* rather than exact values: ffmpeg's RGB conversion of a
/// limited-range YUV source does not round-trip flat colours exactly.
fn assert_colour(actual: (u8, u8, u8), dominant: &[usize], what: &str) {
    let channels = [actual.0 as i32, actual.1 as i32, actual.2 as i32];
    let top = dominant
        .iter()
        .map(|&channel| channels[channel])
        .min()
        .expect("at least one dominant channel");
    assert!(
        top > 90,
        "{what}: expected a lit channel among {dominant:?}, got {actual:?}"
    );
    for channel in (0..3).filter(|channel| !dominant.contains(channel)) {
        assert!(
            top - channels[channel] > 60,
            "{what}: expected only channels {dominant:?}, got {actual:?}"
        );
    }
}

/// Colour of one pixel of the frame at `at` seconds.
fn pixel_at(video_path: &Path, at: f64, x: usize, y: usize) -> (u8, u8, u8) {
    let output = ffmpeg(&[
        "-loglevel",
        "error",
        "-ss",
        &format!("{at:.3}"),
        "-i",
        video_path.to_str().expect("utf-8 path"),
        "-frames:v",
        "1",
        "-vf",
        "format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ]);
    let info = video::probe(video_path).expect("output probes");
    let index = (y * info.width as usize + x) * 3;
    assert!(
        output.stdout.len() >= index + 3,
        "frame is only {} bytes, wanted pixel ({x},{y})",
        output.stdout.len()
    );
    (
        output.stdout[index],
        output.stdout[index + 1],
        output.stdout[index + 2],
    )
}

fn sorted_file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("view directory")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    names
}

fn assert_near(actual: u32, expected: u32, tolerance: u32, what: &str) {
    let difference = (actual as i64 - expected as i64).abs();
    assert!(
        difference <= tolerance as i64,
        "{what}: got {actual}, expected {expected} (±{tolerance})"
    );
}

/// Every written view must be exactly the pane the report says it is.
fn assert_outputs_match_panes(report: &video::SplitReport, dest: &Path) {
    for segment in &report.segments {
        for (index, pane) in segment.layout.panes.iter().enumerate() {
            let name = format!("seg{:02}_view{}.mp4", segment.segment.index + 1, index + 1);
            let info = video::probe(&dest.join(&name)).expect("view probes");
            assert_eq!(
                (info.width, info.height),
                (pane.width, pane.height),
                "{name} does not match pane {pane:?}"
            );
        }
    }
}

#[test]
fn crops_a_tiled_grid_and_splits_scenes() {
    let temp = Temp::new();
    let input = temp.path("tiled.mp4");
    build_tiled_video(&input);
    let dest = temp.path("views");

    let report = video::split_views(
        &input,
        &dest,
        &options(SplitMode::Both, Layout::Grid(Grid::new(2, 2).unwrap())),
    )
    .expect("splits");

    assert_eq!(report.segments.len(), 2, "one cut at the scene change");
    assert!(
        (1.9..=2.2).contains(&report.segments[1].segment.start),
        "cut should land near 2s, got {}",
        report.segments[1].segment.start
    );

    // 4 views in both halves, from the pinned grid.
    for segment in &report.segments {
        assert_eq!(segment.source, LayoutSource::Configured);
        assert_eq!(segment.layout.views(), 4, "{:?}", segment.layout);
        for pane in &segment.layout.panes {
            assert_near(pane.width, TILE_W, 14, "pane width");
            assert_near(pane.height, TILE_H, 14, "pane height");
        }
    }

    assert_eq!(report.outputs.len(), 8);
    assert_eq!(
        sorted_file_names(&dest),
        vec![
            "seg01_view1.mp4",
            "seg01_view2.mp4",
            "seg01_view3.mp4",
            "seg01_view4.mp4",
            "seg02_view1.mp4",
            "seg02_view2.mp4",
            "seg02_view3.mp4",
            "seg02_view4.mp4",
        ]
    );
    assert_outputs_match_panes(&report, &dest);

    // Each view really is the camera that was in that corner, checked just
    // inside the corner and again near the far corner of the same pane.
    let pane = |segment: usize, view: usize| report.segments[segment].layout.panes[view];
    let colour = |segment: usize, view: usize, at: f64, x: u32, y: u32| {
        let name = format!("seg{:02}_view{}.mp4", segment + 1, view + 1);
        pixel_at(&dest.join(name), at, x as usize, y as usize)
    };

    for (segment_index, label) in [(0usize, "scene 1"), (1usize, "scene 2")] {
        let _ = label;
        let top_left = pane(segment_index, 0);
        let top_right = pane(segment_index, 1);
        let bottom_left = pane(segment_index, 2);
        let bottom_right = pane(segment_index, 3);
        assert!(top_left.x < top_right.x, "views are in reading order");
        assert!(top_left.y < bottom_left.y, "views are in reading order");

        if segment_index == 0 {
            assert_colour(colour(0, 0, 0.5, 10, 10), &[0], "top-left view");
            assert_colour(colour(0, 1, 0.5, 10, 10), &[1], "top-right view");
            assert_colour(colour(0, 2, 0.5, 10, 10), &[2], "bottom-left view");
            assert_colour(colour(0, 3, 0.5, 10, 10), &[0, 1], "bottom-right view");

            // Far corner of the same panes, so the whole view came along.
            assert_colour(
                colour(0, 0, 0.5, top_left.width - 20, top_left.height - 20),
                &[0],
                "top-left view, far corner",
            );
            assert_colour(
                colour(
                    0,
                    3,
                    0.5,
                    bottom_right.width - 20,
                    bottom_right.height - 20,
                ),
                &[0, 1],
                "bottom-right view, far corner",
            );
        }
    }

    // The second segment is the second scene, not a repeat of the first.
    assert_ne!(
        pixel_at(&dest.join("seg01_view1.mp4"), 0.5, 10, 10),
        pixel_at(&dest.join("seg02_view1.mp4"), 0.5, 10, 10),
        "segment 2 should show the second scene"
    );
}

#[test]
fn championship_layout_appears_after_the_title_card() {
    let temp = Temp::new();
    let input = temp.path("championship.mp4");
    build_championship_video(&input);
    let dest = temp.path("views");

    let report =
        video::split_views(&input, &dest, &options(SplitMode::Both, Layout::Auto)).expect("splits");

    // The title card and the broadcast are different shots.
    assert_eq!(report.segments.len(), 2);
    assert!(
        (CHAMP_INTRO_SECS as f64 - 0.5..=CHAMP_INTRO_SECS as f64 + 0.5)
            .contains(&report.segments[1].segment.start),
        "cut should land at {CHAMP_INTRO_SECS}s, got {}",
        report.segments[1].segment.start
    );

    // The title card is one full-frame view and must not be cropped.
    let intro = &report.segments[0];
    assert_eq!(intro.source, LayoutSource::Single);
    assert_eq!(intro.layout.views(), 1);
    assert_eq!(intro.layout.panes[0], Pane::new(0, 0, CHAMP_W, 1080));

    // The broadcast is the championship layout: wide overview on top, two
    // subviews below, split by thin bright lines.
    let tiled = &report.segments[1];
    assert_eq!(tiled.source, LayoutSource::Detected);
    assert_eq!(tiled.layout.views(), 3, "{:?}", tiled.layout);
    assert_eq!(
        tiled.layout.panes[0],
        Pane::new(0, 0, CHAMP_W, CHAMP_OVERVIEW_H),
        "the overview keeps the full width"
    );
    assert_eq!(
        tiled.layout.panes[1],
        Pane::new(
            0,
            CHAMP_OVERVIEW_H + CHAMP_LINE,
            CHAMP_LEFT_W,
            CHAMP_SUBVIEW_H
        )
    );
    assert_eq!(
        tiled.layout.panes[2],
        Pane::new(
            CHAMP_LEFT_W + CHAMP_LINE,
            CHAMP_OVERVIEW_H + CHAMP_LINE,
            CHAMP_RIGHT_W,
            CHAMP_SUBVIEW_H
        )
    );

    // 1 file for the intro, 3 for the broadcast.
    assert_eq!(report.outputs.len(), 4);
    assert_eq!(
        sorted_file_names(&dest),
        vec![
            "seg01_view1.mp4",
            "seg02_view1.mp4",
            "seg02_view2.mp4",
            "seg02_view3.mp4",
        ]
    );
    assert_outputs_match_panes(&report, &dest);

    // The intro keeps the whole frame; the three panes hold the three cameras.
    let intro_info = video::probe(&dest.join("seg01_view1.mp4")).expect("intro probes");
    assert_eq!((intro_info.width, intro_info.height), (CHAMP_W, 1080));
    assert_colour(
        pixel_at(&dest.join("seg02_view1.mp4"), 1.0, 960, 300),
        &[0],
        "overview",
    );
    assert_colour(
        pixel_at(&dest.join("seg02_view2.mp4"), 1.0, 400, 200),
        &[1],
        "bottom-left subview",
    );
    assert_colour(
        pixel_at(&dest.join("seg02_view3.mp4"), 1.0, 400, 200),
        &[2],
        "bottom-right subview",
    );
}

#[test]
fn leaves_a_single_view_video_alone() {
    let temp = Temp::new();
    let input = temp.path("single.mp4");
    build_single_view_video(&input);
    let dest = temp.path("views");

    let report =
        video::split_views(&input, &dest, &options(SplitMode::Both, Layout::Auto)).expect("splits");

    assert!(report.is_single_view_throughout());
    assert_eq!(report.segments.len(), 1, "a steady clip has no scene changes");
    assert_eq!(report.segments[0].source, LayoutSource::Single);
    assert_eq!(report.outputs.len(), 1);
    let info = video::probe(&dest.join("seg01_view1.mp4")).expect("output probes");
    assert_eq!((info.width, info.height), (1280, 720), "frame must be untouched");
}

#[test]
fn crops_side_by_side_views_with_a_configured_layout() {
    let temp = Temp::new();
    let input = temp.path("side-by-side.mp4");
    build_side_by_side_video(&input);
    let dest = temp.path("views");

    // 640x720 views are nowhere near 16:9 and there is no divider line at all,
    // so this only works because the layout is pinned in the config.
    let report = video::split_views(
        &input,
        &dest,
        &options(SplitMode::Tiled, Layout::Grid(Grid::new(1, 2).unwrap())),
    )
    .expect("splits");

    assert_eq!(report.segments[0].source, LayoutSource::Configured);
    assert_eq!(report.segments.len(), 1, "tiled mode does not cut on scenes");
    assert_eq!(report.outputs.len(), 2);

    let info = video::probe(&dest.join("seg01_view1.mp4")).expect("left view probes");
    assert_eq!((info.width, info.height), (640, 720));

    assert_colour(
        pixel_at(&dest.join("seg01_view1.mp4"), 0.5, 320, 360),
        &[0],
        "left view",
    );
    assert_colour(
        pixel_at(&dest.join("seg01_view2.mp4"), 0.5, 320, 360),
        &[1],
        "right view",
    );
}

#[test]
fn scenes_only_mode_never_crops() {
    let temp = Temp::new();
    let input = temp.path("tiled.mp4");
    build_tiled_video(&input);
    let dest = temp.path("views");

    let report =
        video::split_views(&input, &dest, &options(SplitMode::Scenes, Layout::Auto)).expect("splits");

    assert_eq!(report.outputs.len(), 2, "one file per segment");
    for name in sorted_file_names(&dest) {
        let info = video::probe(&dest.join(&name)).expect("output probes");
        assert_eq!(
            (info.width, info.height),
            (FRAME_W, FRAME_H),
            "{name} should keep the full frame"
        );
    }
}

#[test]
fn cli_splits_a_video_file() {
    let temp = Temp::new();
    let input = temp.path("championship.mp4");
    build_championship_video(&input);
    let out = temp.path("cli-views");
    let config = temp.path("camcorder.toml");
    std::fs::write(
        &config,
        format!(
            "team = 254\nyear = 2025\n\n[output]\ndir = {:?}\n",
            temp.path("output")
        ),
    )
    .expect("writes config");

    let output = Command::new(env!("CARGO_BIN_EXE_camcorder"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "views",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("runs camcorder");

    assert!(
        output.status.success(),
        "camcorder views failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("3 views [detected]"),
        "expected the championship layout to be detected in:\n{stderr}"
    );
    assert!(
        stderr.contains("1920x664 + 954x412 + 962x412"),
        "expected the pane sizes in:\n{stderr}"
    );
    assert_eq!(sorted_file_names(&out).len(), 4);

    // Printed to stdout so the paths can be piped somewhere.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("seg01_view1.mp4") && stdout.contains("seg02_view3.mp4"),
        "expected the produced paths on stdout, got: {stdout}"
    );
}

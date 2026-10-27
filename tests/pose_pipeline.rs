//! End-to-end tests for the pose pipeline, on real videos built by ffmpeg.
//!
//! The fixtures are *projections*: a tag from the bundled WPILib layout, seen
//! from a camera placed at a known point on the field, with the picture drawn by
//! intersecting each pixel's ray with the tag's plane. That matters more than it
//! sounds. A tag drawn flat into the frame, as a rectangle on a page, is not a
//! picture any camera could take of a tag on a wall: it implies a camera
//! squarely on the tag's axis, and — as this test suite discovered — lands the
//! solve on the wrong side of the wall, where no camera can read a tag at all.
//! A projected fixture has a true pose by construction, so the pose can be
//! checked against something rather than just checked for existence.
//!
//! The shapes that matter:
//! * a tag on a red background — detected through h264, solved to where the
//!   camera was, read as the red end of the field, and marked as the weak
//!   evidence a single tag is;
//! * a faint tag, whose border is 28 luma levels above its quiet zone — the raw
//!   frames do not decode it at all, and the enhanced ones do;
//! * two tags — the lens measured from the picture, which is the only way the
//!   scale of a broadcast pose can be right;
//! * the same footage named as the overview pane — solved, but not asked about
//!   the side of the field, because one wide shot holds both alliances.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use camcorder::pose::{
    self,
    types::{CameraPose, Intrinsics, Mat3, Vec3},
    Alliance, End, Pane, PoseOptions, TagLayout,
};
use kornia_apriltag::family::{TagFamily, TagFamilyKind};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

/// FRC's red, as it comes out of a broadcast.
const RED: (u8, u8, u8) = (220, 30, 40);
/// The grey a field's carpet and surrounds come out as.
const GREY: (u8, u8, u8) = (110, 112, 116);
/// A clean tag: full black and full white.
const CLEAN_INK: (u8, u8) = (0, 255);
/// A tag in a dimmer picture: still comfortably readable, but no longer
/// saturating the range.
const READABLE_INK: (u8, u8) = (95, 145);
/// A faint tag: 28 luma levels between its dark cells and its light ones, which
/// is below what the decoder reads until the contrast is stretched.
const FAINT_INK: (u8, u8) = (106, 134);

/// The field of view the fixtures are rendered at, and the one a pose run is
/// told unless the test is about the lens being wrong.
const TRUTH_FOV: f64 = 60.0;

struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().expect("temp dir"),
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// A `camcorder.toml`, which the binary reads before it looks at its
    /// subcommand even for `pose`.
    fn write_config(&self) {
        std::fs::write(
            self.path("camcorder.toml"),
            "team = 254\nyear = 2026\n\n[output]\ndir = \"output\"\n",
        )
        .expect("write config");
    }
}

fn layout() -> TagLayout {
    TagLayout::load("2026-rebuilt-welded").expect("bundled layout")
}

fn family() -> TagFamily {
    TagFamily::try_from(&TagFamilyKind::Tag36H11).expect("the 36h11 family")
}

/// A pose at `from` looking straight at `target`, with the field's `+z` up.
fn looking_at(from: Vec3, target: Vec3) -> CameraPose {
    let forward = (target - from).normalized();
    let right = forward.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
    let down = forward.cross(right);
    CameraPose {
        position: from,
        rotation: Mat3::from_rows([right.to_array(), down.to_array(), forward.to_array()])
            .to_quaternion(),
    }
}

/// Where the fixture camera stood. Close to the red wall, a little above the
/// tags, looking along the wall at a pair of them — which is what a broadcast
/// sub-view of the alliance wall looks like.
fn camera_position() -> Vec3 {
    Vec3::new(15.55, 6.85, 0.95)
}

fn tag_centre(ids: &[u16]) -> Vec3 {
    let layout = layout();
    let mut centre = Vec3::new(0.0, 0.0, 0.0);
    for id in ids {
        centre = centre
            + layout
                .tag(*id)
                .expect("the tag is in the layout")
                .translation;
    }
    centre.scaled(1.0 / ids.len() as f64)
}

fn truth_pose(ids: &[u16]) -> CameraPose {
    looking_at(camera_position(), tag_centre(ids))
}

/// One frame: what a camera at `pose` sees of the given tags, on `background`.
///
/// Every pixel is the average of four samples of the ray through it, so the
/// cells' edges are anti-aliased the way a real lens and a real codec leave
/// them. `ink` is (dark, bright) in luma, which is how a faint tag is made: the
/// same geometry with no contrast left in it.
fn render_scene(
    ids: &[u16],
    pose: CameraPose,
    camera: &Intrinsics,
    background: (u8, u8, u8),
    ink: (u8, u8),
) -> Vec<u8> {
    let layout = layout();
    let family = family();
    let tag_size = pose::FRC_TAG_SIZE_M;
    // `FRC_TAG_SIZE_M` is the printed square — the black border's outer edge,
    // which is what the decoder's quad marks — and that square is eight cells
    // across, six of code and one of border all the way round.
    let cell = tag_size / 8.0;
    let quiet_cells = 2.0;
    let to_field = Mat3::from_quaternion(pose.rotation).transpose();

    let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];
    let (pixels, _) = rgb.as_chunks_mut::<3>();
    for pixel in pixels.iter_mut() {
        pixel.copy_from_slice(&[background.0, background.1, background.2]);
    }

    let planes: Vec<(Vec3, Vec3, Mat3, usize)> = ids
        .iter()
        .map(|id| {
            let tag = layout.tag(*id).expect("the tag is in the layout");
            let rotation = Mat3::from_quaternion(tag.rotation);
            let normal = rotation.mul_vec(Vec3::new(1.0, 0.0, 0.0));
            (
                tag.translation,
                normal,
                rotation,
                family.code_data[*id as usize],
            )
        })
        .collect();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let mut colour = [0.0f64; 3];
            let mut samples = 0.0;
            for (dy, dx) in [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)] {
                let u = x as f64 + dx;
                let v = y as f64 + dy;
                let ray = to_field.mul_vec(Vec3::new(
                    (u - camera.cx) / camera.fx,
                    (v - camera.cy) / camera.fy,
                    1.0,
                ));
                // A tag is ink on paper, so it is grey; everything else keeps
                // the background's colour, which is what the side of the field
                // is read from.
                let value = luma(&planes, &family, pose.position, ray, cell, quiet_cells, ink);
                let level = match value {
                    Some(value) => [value; 3],
                    None => [
                        background.0 as f64,
                        background.1 as f64,
                        background.2 as f64,
                    ],
                };
                for channel in 0..3 {
                    colour[channel] += level[channel];
                }
                samples += 1.0;
            }
            let base = (y * WIDTH + x) * 3;
            for channel in 0..3 {
                rgb[base + channel] = (colour[channel] / samples).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    rgb
}

/// The luma one ray sees on a tag plane: the nearest tag it meets, or `None`
/// when the ray misses every tag.
fn luma(
    planes: &[(Vec3, Vec3, Mat3, usize)],
    family: &TagFamily,
    eye: Vec3,
    ray: Vec3,
    cell: f64,
    quiet_cells: f64,
    ink: (u8, u8),
) -> Option<f64> {
    let mut hit: Option<f64> = None;
    let mut value = None;
    for (centre, normal, rotation, code) in planes {
        let denominator = ray.dot(*normal);
        if denominator.abs() < 1e-9 {
            continue;
        }
        let reach = (*centre - eye).dot(*normal) / denominator;
        if reach <= 1e-6 {
            continue;
        }
        if hit.map(|nearest| reach >= nearest).unwrap_or(false) {
            continue;
        }
        let point = eye + ray.scaled(reach);
        let local = rotation.transpose().mul_vec(point - *centre);
        // The picture's right is the tag's `-y` and its down is the tag's `-z`:
        // that is the corner ordering the layout documents.
        let grid_x = local.y / cell;
        let grid_y = -local.z / cell;
        let extent = 4.0 + quiet_cells;
        if grid_x.abs() > extent || grid_y.abs() > extent {
            continue;
        }
        hit = Some(reach);
        if grid_x.abs() > 4.0 || grid_y.abs() > 4.0 {
            // The quiet zone: light, and the border has something to stand
            // against.
            value = Some(ink.1 as f64);
            continue;
        }
        let border_x = (grid_x + 4.0).floor() as usize;
        let border_y = (grid_y + 4.0).floor() as usize;
        let mut cell_value = ink.1;
        if border_x == 0 || border_y == 0 || border_x == 7 || border_y == 7 {
            cell_value = ink.0;
        } else {
            for bit in 0..family.nbits {
                if family.bit_x[bit] as usize == border_x && family.bit_y[bit] as usize == border_y
                {
                    // The family's code words are stored most significant bit
                    // first; reading them the other way round decodes to some
                    // other member of the family, if anything at all.
                    let value_bit = (code >> (family.nbits - 1 - bit)) & 1;
                    cell_value = if value_bit == 1 { ink.1 } else { ink.0 };
                }
            }
        }
        value = Some(cell_value as f64);
    }
    value
}

/// Encode one still as a two-second h264 clip, which is what the pipeline reads.
fn encode(dir: &Path, name: &str, rgb: &[u8], filter: Option<&str>) -> PathBuf {
    let raw = dir.join("frame.rgb");
    std::fs::write(&raw, rgb).expect("write the raw frame");
    let out = dir.join(name);
    let mut cmd = Command::new(camcorder::video::BIN_FFMPEG);
    cmd.args([
        "-y",
        "-loglevel",
        "error",
        "-stream_loop",
        "-1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        &format!("{WIDTH}x{HEIGHT}"),
        "-r",
        "30",
        "-i",
    ])
    .arg(&raw)
    .args(["-t", "2", "-c:v", "libx264", "-preset", "ultrafast"]);
    if let Some(filter) = filter {
        cmd.args(["-vf", filter]);
    }
    let output = cmd
        .args(["-pix_fmt", "yuv420p"])
        .arg(&out)
        .output()
        .expect("ffmpeg is required to build these fixtures");
    assert!(
        output.status.success(),
        "encoding {name} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    out
}

fn options(preprocess: bool) -> PoseOptions {
    PoseOptions {
        tag_layout: "2026-rebuilt-welded".to_string(),
        samples: 3,
        preprocess,
        ..PoseOptions::default()
    }
}

#[test]
fn a_solved_pose_lands_where_the_camera_was() {
    let sandbox = Sandbox::new();
    let camera = Intrinsics::from_horizontal_fov(WIDTH as u32, HEIGHT as u32, TRUTH_FOV);
    let truth = truth_pose(&[13]);
    let video = encode(
        sandbox.dir.path(),
        "seg01_view2.mp4",
        &render_scene(&[13], truth, &camera, RED, CLEAN_INK),
        None,
    );

    let report = pose::estimate(&video, &options(true)).expect("the estimate runs");
    assert_eq!(
        report.pane,
        Pane::View,
        "a sub-view is where the side is read"
    );
    assert!(
        !report.solved.is_empty(),
        "a clean tag must survive the codec:\n{}\n{:#?}",
        report.render_table(),
        report.missed
    );

    for frame in &report.solved {
        let error = (Vec3::from_array(frame.position) - truth.position).norm();
        assert!(
            error < 0.25,
            "pose {error:.2}m from the camera's actual position {frame:?}"
        );
        assert!(frame.position[2] > 0.0, "the camera is above the carpet");

        let side = frame.side.as_ref().expect("a view gets a side reading");
        assert_eq!(side.alliance, Alliance::Red, "the picture is red");
        assert_eq!(side.tags_end, End::HighX, "tag 13 is at the red end");
        assert_eq!(side.disagreement, None, "colour and tags agree: {side:?}");

        // One tag cannot pin a pose: its scale is the focal length, and with
        // one tag the focal length is the guess. The report must say so.
        assert!(
            frame.confidence < 0.7,
            "a single-tag frame should read as weak evidence, got {}",
            frame.confidence
        );
        assert!(
            frame.weakest_margin > 30.0,
            "margin {}",
            frame.weakest_margin
        );
    }

    let station = report.station.as_ref().expect("a station pose");
    let error = (Vec3::from_array(station.position) - truth.position).norm();
    assert!(
        error < 0.25,
        "station pose {error:.2}m from the camera's actual position"
    );
    assert_eq!(station.frames_used, report.solved.len());
    assert_eq!(station.frames_rejected, 0);

    // A single tag cannot measure the lens either, so the guess is kept.
    assert!(
        report.focal_source.contains("assumed"),
        "{}",
        report.focal_source
    );
}

#[test]
fn two_tags_measure_the_lens_the_run_was_told_wrong() {
    let sandbox = Sandbox::new();
    let camera = Intrinsics::from_horizontal_fov(WIDTH as u32, HEIGHT as u32, TRUTH_FOV);
    let truth = truth_pose(&[13, 14]);
    let video = encode(
        sandbox.dir.path(),
        "pair.mp4",
        &render_scene(&[13, 14], truth, &camera, GREY, CLEAN_INK),
        None,
    );

    // The run is told a field of view far too wide. The two tags are a known
    // distance apart on the wall, so the picture says what the lens is.
    let wrong = PoseOptions {
        horizontal_fov_deg: 85.0,
        ..options(true)
    };
    let report = pose::estimate(&video, &wrong).expect("the estimate runs");
    assert!(
        !report.solved.is_empty(),
        "two clean tags must solve:\n{}\n{:#?}",
        report.render_table(),
        report.missed
    );
    assert!(
        report.focal_source.contains("measured"),
        "two tags should measure the lens, got: {}",
        report.focal_source
    );
    assert!(
        (report.focal_px - camera.fx).abs() / camera.fx < 0.05,
        "measured {:.0}px against the camera's {:.0}px",
        report.focal_px,
        camera.fx
    );

    for frame in &report.solved {
        let error = (Vec3::from_array(frame.position) - truth.position).norm();
        assert!(
            error < 0.35,
            "pose {error:.2}m out after the lens was measured"
        );
        assert!(
            frame.confidence > 0.7,
            "two tags should be stronger evidence than one, got {}",
            frame.confidence
        );
    }
}

#[test]
fn a_faint_tag_is_only_read_once_the_frame_is_enhanced() {
    let sandbox = Sandbox::new();
    let camera = Intrinsics::from_horizontal_fov(WIDTH as u32, HEIGHT as u32, TRUTH_FOV);
    let truth = truth_pose(&[13]);
    let file = encode(
        sandbox.dir.path(),
        "faint.mp4",
        &render_scene(&[13], truth, &camera, GREY, FAINT_INK),
        None,
    );

    let enhanced = pose::estimate(&file, &options(true)).expect("the estimate runs");
    assert!(
        !enhanced.solved.is_empty(),
        "the stretch is what makes this tag readable:\n{}\n{:#?}",
        enhanced.render_table(),
        enhanced.missed
    );

    // The same footage with the frames as they are. This is the fact the
    // enhancement exists for: if the raw decode ever starts reading these, the
    // fixture has stopped being faint and needs more fade, not a weaker
    // assertion.
    let raw = pose::estimate(&file, &options(false)).expect("the estimate runs");
    assert!(
        raw.solved.is_empty(),
        "the faint fixture is no longer faint: raw frames solved {} of them",
        raw.solved.len()
    );

    // …and it must not cost a tag that was already readable.
    let readable = encode(
        sandbox.dir.path(),
        "readable.mp4",
        &render_scene(&[13], truth, &camera, GREY, READABLE_INK),
        None,
    );
    for preprocess in [false, true] {
        let report = pose::estimate(&readable, &options(preprocess)).expect("the estimate runs");
        assert!(
            !report.solved.is_empty(),
            "a readable tag was lost (preprocess: {preprocess}):\n{}",
            report.render_table()
        );
    }
}

#[test]
fn the_overview_pane_is_not_asked_which_side_of_the_field_it_is_on() {
    let sandbox = Sandbox::new();
    let camera = Intrinsics::from_horizontal_fov(WIDTH as u32, HEIGHT as u32, TRUTH_FOV);
    let truth = truth_pose(&[13]);
    let video = encode(
        sandbox.dir.path(),
        "seg01_view1.mp4",
        &render_scene(&[13], truth, &camera, RED, CLEAN_INK),
        None,
    );

    let report = pose::estimate(&video, &options(true)).expect("the estimate runs");
    assert_eq!(report.pane, Pane::Overview);
    assert!(
        !report.solved.is_empty(),
        "the overview still gets a pose:\n{}",
        report.render_table()
    );
    for frame in &report.solved {
        assert!(
            frame.side.is_none(),
            "an overview holds both alliances, so it must not be read: {:?}",
            frame.side
        );
    }
    assert!(
        report.render_table().contains("overview pane"),
        "the table should say why the side is missing:\n{}",
        report.render_table()
    );
}

#[test]
fn the_cli_flags_turn_the_enhancement_and_the_weak_frames_off() {
    let sandbox = Sandbox::new();
    sandbox.write_config();
    let camera = Intrinsics::from_horizontal_fov(WIDTH as u32, HEIGHT as u32, TRUTH_FOV);
    let truth = truth_pose(&[13]);
    let video = encode(
        sandbox.dir.path(),
        "faint.mp4",
        &render_scene(&[13], truth, &camera, GREY, FAINT_INK),
        None,
    );
    let report_path = sandbox.path("pose.json");

    let run = |extra: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_camcorder"))
            .arg("pose")
            .arg(&video)
            .args([
                "--tag-layout",
                "2026-rebuilt-welded",
                "--json",
                report_path.to_str().expect("utf-8 path"),
            ])
            .args(extra)
            .current_dir(sandbox.dir.path())
            .output()
            .expect("run camcorder");
        assert!(
            output.status.success(),
            "camcorder pose failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report_path).expect("the report was written"))
                .expect("the report is JSON");
        (String::from_utf8_lossy(&output.stderr).to_string(), report)
    };

    let (stderr, report) = run(&[]);
    assert!(stderr.contains("enhanced"), "stderr:\n{stderr}");
    assert!(
        stderr.contains("focal length"),
        "the run should say where its focal length came from:\n{stderr}"
    );
    let first = report["solved"]
        .as_array()
        .expect("solved")
        .first()
        .expect("the default run should read the faint tag")
        .clone();
    // The side of the field travels with the pose into the report.
    assert_eq!(
        first["side"]["alliance"], "unknown",
        "a grey picture picks no side"
    );
    assert_eq!(
        first["side"]["tags_end"], "high_x",
        "tag 13 is at the red end"
    );
    assert!(first["side"]["red_pixels"].is_number());
    // …and so does how much of a result the frame is.
    let confidence = first["confidence"].as_f64().expect("a confidence");
    assert!((0.0..1.0).contains(&confidence), "confidence {confidence}");
    assert!(first["weakest_margin"].is_number());
    assert!(report["station"]["position"].is_array());

    let (stderr, report) = run(&["--no-preprocess"]);
    assert!(stderr.contains("raw frames"), "stderr:\n{stderr}");
    assert!(
        report["solved"].as_array().expect("solved").is_empty(),
        "raw frames should not read the faint tag:\n{stderr}"
    );

    // A single-tag frame is worth about half a result; asking for more than
    // that has to drop it, and say so.
    let (stderr, report) = run(&["--min-confidence", "0.9"]);
    assert!(
        report["solved"].as_array().expect("solved").is_empty(),
        "a 0.9 floor should drop single-tag frames:\n{stderr}"
    );
    let missed = report["missed"].as_array().expect("missed");
    assert!(
        missed.iter().any(|frame| frame["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("confidence")),
        "the report should say why the frame was dropped: {missed:?}"
    );
}

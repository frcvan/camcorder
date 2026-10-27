//! Where the camera was, worked out from the AprilTags in the picture.
//!
//! This is a self-contained subsystem: it takes a video file and a tag layout
//! and returns a [`PoseReport`], and it knows nothing about TBA, splitting or
//! the config file. The pieces are:
//!
//! - [`layout`]: where the tags are on the field (WPILib's own JSON layouts).
//! - [`frames`]: pulling greyscale frames out of the video.
//! - [`detect`]: finding tag corners in a frame.
//! - [`solve`]: turning corners into a camera pose.
//! - [`report`]: the result, as text or JSON.
//!
//! Everything else in the crate can be changed without touching this, and this
//! can be lifted into its own binary by keeping these five modules together.

pub mod detect;
pub mod frames;
pub mod layout;
pub mod report;
pub mod solve;
pub mod types;

use std::path::Path;

use anyhow::{bail, Result};

pub use detect::TagDetector;
pub use layout::{TagLayout, TagPose};
pub use report::{FramePose, MissedFrame, PoseReport};
pub use types::{CameraPose, Intrinsics, Quat, TagObservation, Vec3};

/// Tag edge length of the FRC field tags, in metres (6.5 inches).
pub const FRC_TAG_SIZE_M: f64 = 0.1651;

/// A frame whose solve reprojects worse than this is not trusted.
const DEFAULT_MAX_RMS_PX: f64 = 8.0;

/// How to estimate a pose.
#[derive(Debug, Clone)]
pub struct PoseOptions {
    /// Bundled layout name or a path to a WPILib layout JSON file.
    pub tag_layout: String,
    /// Physical edge length of one tag, metres.
    pub tag_size_m: f64,
    /// Horizontal field of view of the source camera, degrees. This is the one
    /// number the estimate cannot work out for itself: broadcast cameras differ
    /// and the video carries no focal length.
    pub horizontal_fov_deg: f64,
    /// How many frames to sample from the video.
    pub samples: usize,
    /// Restrict sampling to this `(start, end)` in seconds.
    pub window: Option<(f64, f64)>,
    /// Drop detections the decoder was less sure about than this.
    pub min_margin: f32,
    /// Reject a solve that reprojects worse than this, in pixels.
    pub max_rms_px: f64,
}

impl Default for PoseOptions {
    fn default() -> Self {
        PoseOptions {
            tag_layout: "2025-reefscape-welded".to_string(),
            tag_size_m: FRC_TAG_SIZE_M,
            horizontal_fov_deg: 60.0,
            samples: 10,
            window: None,
            min_margin: 30.0,
            max_rms_px: DEFAULT_MAX_RMS_PX,
        }
    }
}

/// Estimate the camera pose in every sampled frame of `video`.
pub fn estimate(video: &Path, options: &PoseOptions) -> Result<PoseReport> {
    if !(options.horizontal_fov_deg > 1.0 && options.horizontal_fov_deg < 179.0) {
        bail!(
            "horizontal field of view must be between 1 and 179 degrees, got {}",
            options.horizontal_fov_deg
        );
    }
    if !options.tag_size_m.is_finite() || options.tag_size_m <= 0.0 {
        bail!("tag size must be positive, got {}", options.tag_size_m);
    }
    if !(options.samples > 0) {
        bail!("sample at least one frame, got {}", options.samples);
    }
    if !video.is_file() {
        bail!("no such video: {}", video.display());
    }

    let layout = TagLayout::load(&options.tag_layout)?;
    let info = crate::video::probe(video)?;
    let times = frames::sample_times(info.duration, options.samples, options.window);
    if times.is_empty() {
        bail!("nothing to sample: the video is {:.1}s long", info.duration);
    }
    let sampled = frames::grab(video, &info, &times)?;

    let camera = Intrinsics::from_horizontal_fov(
        sampled[0].width,
        sampled[0].height,
        options.horizontal_fov_deg,
    );
    let mut detector = TagDetector::new(sampled[0].width, sampled[0].height)?;

    let mut report = PoseReport {
        video: video.to_path_buf(),
        tag_layout: layout.name.clone(),
        frame_width: sampled[0].width,
        frame_height: sampled[0].height,
        horizontal_fov_deg: options.horizontal_fov_deg,
        tag_size_m: options.tag_size_m,
        sampled: sampled.len(),
        solved: Vec::new(),
        missed: Vec::new(),
        tags_seen: Vec::new(),
    };

    for frame in &sampled {
        let detections = detector.detect(&frame.pixels, options.min_margin)?;
        if debug_enabled() {
            eprintln!("  [pose] {:.2}s: {}", frame.time, describe(&detections));
        }
        if detections.is_empty() {
            report.missed.push(MissedFrame {
                time: frame.time,
                tags: Vec::new(),
                reason: "no tags in the frame".to_string(),
            });
            continue;
        }
        for detection in &detections {
            if !report.tags_seen.contains(&detection.id) {
                report.tags_seen.push(detection.id);
            }
        }
        report.tags_seen.sort_unstable();

        match solve::solve(&detections, &layout, options.tag_size_m, &camera) {
            Some(solution) if solution.rms_px <= options.max_rms_px => {
                report.solved.push(FramePose::new(
                    frame.time,
                    &solution.pose,
                    solution.tags,
                    solution.corners,
                    solution.rms_px,
                ));
            }
            Some(solution) => report.missed.push(MissedFrame {
                time: frame.time,
                tags: solution.tags,
                reason: format!(
                    "reprojection error {:.1}px is above the {:.1}px limit",
                    solution.rms_px, options.max_rms_px
                ),
            }),
            None => report.missed.push(MissedFrame {
                time: frame.time,
                tags: detections.iter().map(|detection| detection.id).collect(),
                reason: "the tags seen do not determine a pose".to_string(),
            }),
        }
    }

    Ok(report)
}

/// `CAMCORDER_DEBUG_POSE=1` prints what was read in each frame. A pose that
/// never appears is usually a detection problem, not a solver problem, and this
/// says which.
fn debug_enabled() -> bool {
    std::env::var_os("CAMCORDER_DEBUG_POSE").is_some()
}

fn describe(detections: &[TagObservation]) -> String {
    if detections.is_empty() {
        return "no detections".to_string();
    }
    let parts: Vec<String> = detections
        .iter()
        .map(|detection| {
            let centre_x = detection
                .corners
                .iter()
                .map(|corner| corner[0])
                .sum::<f64>()
                / 4.0;
            let centre_y = detection
                .corners
                .iter()
                .map(|corner| corner[1])
                .sum::<f64>()
                / 4.0;
            format!(
                "id {} at ({centre_x:.0},{centre_y:.0}) {:.0}px, margin {:.0}",
                detection.id,
                detection.pixel_width(),
                detection.margin
            )
        })
        .collect();
    format!("{} detection(s): {}", detections.len(), parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_describe_a_broadcast_camera_and_the_2025_field() {
        let options = PoseOptions::default();
        assert_eq!(options.tag_layout, "2025-reefscape-welded");
        assert!((options.tag_size_m - FRC_TAG_SIZE_M).abs() < 1e-12);
        assert!(options.horizontal_fov_deg > 0.0);
        assert!(options.samples > 0);
    }

    #[test]
    fn a_missing_video_is_reported_before_anything_else() {
        let error = estimate(Path::new("no-such-video.mp4"), &PoseOptions::default())
            .expect_err("must fail");
        assert!(error.to_string().contains("no such video"), "got: {error}");
    }

    #[test]
    fn a_nonsense_field_of_view_is_rejected() {
        let options = PoseOptions {
            horizontal_fov_deg: 400.0,
            ..PoseOptions::default()
        };
        let error = estimate(Path::new("whatever.mp4"), &options).expect_err("must fail");
        assert!(error.to_string().contains("field of view"), "got: {error}");
    }
}

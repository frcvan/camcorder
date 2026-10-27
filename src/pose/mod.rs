//! Where the camera was, worked out from the AprilTags in the picture.
//!
//! This is a self-contained subsystem: it takes a video file and a tag layout
//! and returns a [`PoseReport`], and it knows nothing about TBA, splitting or
//! the config file. The pieces are:
//!
//! - [`layout`]: where the tags are on the field (WPILib's own JSON layouts).
//! - [`frames`]: pulling frames out of the video.
//! - [`enhance`]: making a frame easier for the decoder to read.
//! - [`detect`]: finding tag corners in a frame.
//! - [`corners`]: putting those corners where the borders actually cross.
//! - [`solve`]: turning corners into a camera pose.
//! - [`trust`]: which detections and which poses deserve to be believed.
//! - [`station`]: what is only knowable across frames — the lens, and where the
//!   camera stood.
//! - [`side`]: which end of the field the frame is looking at.
//! - [`report`]: the result, as text or JSON.
//!
//! The estimate runs in three passes over the sampled frames, and each one
//! exists because the previous one could not know something:
//!
//! 1. read every frame, so the frames can vote on which tag ids are real;
//! 2. solve every frame against the assumed lens, so the frames with two tags
//!    can measure the lens;
//! 3. solve every frame again against the measured lens, then fit one pose to
//!    the whole station.
//!
//! Everything else in the crate can be changed without touching this, and this
//! can be lifted into its own binary by keeping these modules together.

pub mod corners;
pub mod detect;
pub mod enhance;
pub mod frames;
pub mod layout;
mod model;
pub mod report;
pub mod side;
pub mod solve;
pub mod station;
pub mod trust;
pub mod types;

use std::{collections::HashMap, path::Path};

use anyhow::{bail, Result};

pub use detect::TagDetector;
pub use layout::{TagLayout, TagPose};
pub use report::{FramePose, MissedFrame, PoseReport, StationReport};
pub use side::{Alliance, End, Pane, SideVerdict};
pub use types::{CameraPose, Intrinsics, Quat, TagObservation, Vec3};

/// Tag edge length of the FRC field tags, in metres (6.5 inches).
pub const FRC_TAG_SIZE_M: f64 = 0.1651;

/// How far a field-model round may move the fitted camera before it counts as a
/// different camera rather than a refinement of the same one.
const MODEL_ROUND_MAX_MOVE_M: f64 = 1.0;

/// A frame whose solve reprojects worse than this is not trusted.
const DEFAULT_MAX_RMS_PX: f64 = 8.0;

/// How to estimate a pose.
#[derive(Debug, Clone)]
pub struct PoseOptions {
    /// Bundled layout name or a path to a WPILib layout JSON file.
    pub tag_layout: String,
    /// Physical edge length of one tag, metres.
    pub tag_size_m: f64,
    /// Horizontal field of view of the source camera, degrees. Only a starting
    /// guess: frames holding two or more tags measure the focal length from the
    /// picture itself, and the measurement replaces this for the final solve.
    pub horizontal_fov_deg: f64,
    /// How many frames to sample from the video.
    pub samples: usize,
    /// Restrict sampling to this `(start, end)` in seconds.
    pub window: Option<(f64, f64)>,
    /// The decision margin a full-size tag needs before it is used. Smaller
    /// tags are held to less, down to a third of this, because a tag eight
    /// pixels across cannot reach the margin a tag filling the frame does.
    pub min_margin: f32,
    /// Reject a solve that reprojects worse than this, in pixels.
    pub max_rms_px: f64,
    /// Work on a frame before the detector sees it: contrast stretch and an
    /// unsharp mask, so a small tag in the shade is not lost. Off makes a run
    /// reproduce what the raw picture gives, which is what to reach for when
    /// checking whether a change to [`enhance`] is the thing that moved.
    pub preprocess: bool,
    /// Drop solved frames below this confidence, `0..1`. Zero keeps everything:
    /// a weak frame is still evidence, and the report says how weak it is.
    pub min_confidence: f64,
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
            preprocess: true,
            min_confidence: 0.0,
        }
    }
}

/// One frame, read as far as knowing what is in it needs.
struct Prepared {
    time: f64,
    /// Detections that earned their place, with refined corners.
    detections: Vec<TagObservation>,
    /// The alliance the frame's colours show.
    alliance: Alliance,
    red_pixels: usize,
    blue_pixels: usize,
    /// The end of the field those colours name, for the solver to prefer.
    prefer: Option<End>,
}

/// Everything one frame is judged by, gathered so both solve passes run the
/// same code and cannot drift apart.
struct Judge<'a> {
    layout: &'a TagLayout,
    camera: &'a Intrinsics,
    pane: Pane,
    support: &'a HashMap<u16, trust::IdSupport>,
    options: &'a PoseOptions,
    /// Whether `camera` came from a measurement rather than the `--fov` guess.
    calibrated: bool,
}

/// What one pass over the frames produced.
struct Evaluated {
    views: Vec<station::View>,
    frames: Vec<FramePose>,
    missed: Vec<MissedFrame>,
}

impl Judge<'_> {
    fn evaluate(&self, prepared: &[Prepared]) -> Evaluated {
        let mut evaluated = Evaluated {
            views: Vec::new(),
            frames: Vec::new(),
            missed: Vec::new(),
        };

        for frame in prepared {
            if frame.detections.is_empty() {
                evaluated.missed.push(MissedFrame {
                    time: frame.time,
                    tags: Vec::new(),
                    reason: "no tags in the frame".to_string(),
                });
                continue;
            }
            let seen: Vec<u16> = frame
                .detections
                .iter()
                .map(|detection| detection.id)
                .collect();
            let solution = match solve::solve_preferring(
                &frame.detections,
                self.layout,
                self.options.tag_size_m,
                self.camera,
                frame.prefer,
            ) {
                Ok(solution) => solution,
                Err(reason) => {
                    evaluated.missed.push(MissedFrame {
                        time: frame.time,
                        tags: seen,
                        reason,
                    });
                    continue;
                }
            };
            if solution.rms_px > self.options.max_rms_px {
                evaluated.missed.push(MissedFrame {
                    time: frame.time,
                    tags: solution.tags,
                    reason: format!(
                        "reprojection error {:.1}px is above the {:.1}px limit",
                        solution.rms_px, self.options.max_rms_px
                    ),
                });
                continue;
            }

            // How much of the frame's evidence is behind the answer: the
            // weakest margin among the tags used, and the least corroborated
            // of their ids.
            let weakest_margin = solution
                .tags
                .iter()
                .filter_map(|id| {
                    frame
                        .detections
                        .iter()
                        .find(|detection| detection.id == *id)
                        .map(|detection| detection.margin)
                })
                .fold(f32::INFINITY, f32::min);
            let weakest_support = solution
                .tags
                .iter()
                .map(|id| self.support.get(id).map_or(0.0, |support| support.fraction))
                .fold(f64::INFINITY, f64::min);
            let confidence = trust::confidence(
                &trust::Evidence {
                    tags: solution.tags.len(),
                    rms_px: solution.rms_px,
                    weakest_margin: if weakest_margin.is_finite() {
                        weakest_margin
                    } else {
                        0.0
                    },
                    support: if weakest_support.is_finite() {
                        weakest_support
                    } else {
                        0.0
                    },
                    calibrated: self.calibrated,
                },
                self.options.max_rms_px,
            );
            if confidence < self.options.min_confidence {
                evaluated.missed.push(MissedFrame {
                    time: frame.time,
                    tags: solution.tags,
                    reason: format!(
                        "confidence {confidence:.2} is below the {:.2} limit",
                        self.options.min_confidence
                    ),
                });
                continue;
            }

            // Read after the solve, because half the answer — the end the pose
            // puts the camera at — only exists once there is a pose. The
            // overview pane is not asked: one wide shot holds both alliances.
            let verdict = match self.pane {
                Pane::Overview => None,
                Pane::View => Some(side::verdict(
                    frame.alliance,
                    frame.red_pixels,
                    frame.blue_pixels,
                    &solution.tags,
                    self.layout,
                    Some(&solution.pose),
                )),
            };

            evaluated.views.push(station::View {
                corners: solve::correspondences(
                    &frame.detections,
                    self.layout,
                    self.options.tag_size_m,
                )
                .into_iter()
                .filter(|corner| solution.tags.contains(&corner.tag))
                .collect(),
                pose: solution.pose,
                tags: solution.tags.clone(),
                lines: Vec::new(),
            });
            evaluated.frames.push(FramePose::new(report::FrameOutcome {
                time: frame.time,
                pose: &solution.pose,
                tags: solution.tags,
                corners: solution.corners,
                rms_px: solution.rms_px,
                confidence,
                weakest_margin: if weakest_margin.is_finite() {
                    weakest_margin
                } else {
                    0.0
                },
                side: verdict,
            }));
        }
        evaluated
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
    if options.samples == 0 {
        bail!("sample at least one frame, got {}", options.samples);
    }
    if !options.min_confidence.is_finite() || !(0.0..=1.0).contains(&options.min_confidence) {
        bail!(
            "minimum confidence must be between 0 and 1, got {}",
            options.min_confidence
        );
    }
    if !options.max_rms_px.is_finite() || options.max_rms_px <= 0.0 {
        bail!(
            "maximum reprojection error must be positive, got {}",
            options.max_rms_px
        );
    }
    if !options.min_margin.is_finite() || options.min_margin < 0.0 {
        bail!(
            "minimum margin must be non-negative, got {}",
            options.min_margin
        );
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
    let mut sampled = frames::grab(video, &info, &times)?;

    let width = sampled[0].width;
    let height = sampled[0].height;
    let assumed = Intrinsics::from_horizontal_fov(width, height, options.horizontal_fov_deg);
    // The overview pane holds both alliances at once, so nothing can be read
    // about the side of the field from it; the splitter names it `view1`.
    let pane = side::Pane::of(video);
    let mut detector = TagDetector::new(width, height)?;

    // Pass one: read every frame. Nothing is decided yet — the frames have to
    // vote on which ids are real before any of them can be trusted.
    let mut reads: Vec<Vec<TagObservation>> = Vec::with_capacity(sampled.len());
    for frame in sampled.iter_mut() {
        if options.preprocess {
            enhance::enhance(
                &mut frame.pixels,
                frame.width as usize,
                frame.height as usize,
            );
        }
        reads.push(detector.detect(&frame.pixels)?);
    }
    let support = trust::id_support(&reads);

    // Pass two: decide what is in each frame. A detection that only makes it
    // through on the small-tag allowance has to be backed by another frame.
    let mut prepared = Vec::with_capacity(sampled.len());
    let mut tags_seen: Vec<u16> = Vec::new();
    for (frame, detections) in sampled.iter().zip(reads) {
        let mut kept: Vec<TagObservation> = detections
            .into_iter()
            .filter(|detection| trust::keep(detection, &support, options.min_margin))
            .map(|mut detection| {
                // The detector's quad is good to a fraction of a pixel; the
                // border fit takes it the rest of the way.
                detection.corners = corners::refine(
                    &frame.pixels,
                    frame.width as usize,
                    frame.height as usize,
                    &detection.corners,
                );
                detection
            })
            .collect();
        if debug_enabled() {
            eprintln!("  [pose] {:.2}s: {}", frame.time, describe(&kept));
        }
        for detection in &kept {
            if !tags_seen.contains(&detection.id) {
                tags_seen.push(detection.id);
            }
        }
        let (red_pixels, blue_pixels) = if pane == side::Pane::Overview {
            (0, 0)
        } else {
            side::count_colours(&frame.rgb, frame.width, frame.height)
        };
        let alliance = side::alliance_from(red_pixels, blue_pixels);
        prepared.push(Prepared {
            time: frame.time,
            detections: std::mem::take(&mut kept),
            alliance,
            red_pixels,
            blue_pixels,
            prefer: match pane {
                side::Pane::Overview => None,
                side::Pane::View => side::expected_end(alliance),
            },
        });
    }
    tags_seen.sort_unstable();

    // Pass three: solve against the assumed lens, measure the lens from the
    // frames that can, then solve again with what was measured.
    let judge = Judge {
        layout: &layout,
        camera: &assumed,
        pane,
        support: &support,
        options,
        calibrated: false,
    };
    let provisional = judge.evaluate(&prepared);
    let calibration = station::calibrate(&provisional.views, assumed);

    let calibrated;
    let mut evaluated = if calibration.measured {
        calibrated = Judge {
            layout: &layout,
            camera: &calibration.intrinsics,
            pane,
            support: &support,
            options,
            calibrated: true,
        };
        calibrated.evaluate(&prepared)
    } else {
        provisional
    };

    let station = station::fit(
        &evaluated.views,
        &calibration.intrinsics,
        &layout,
        options.max_rms_px,
    )
    .map(|fitted| StationReport {
        position: [
            fitted.pose.position.x,
            fitted.pose.position.y,
            fitted.pose.position.z,
        ],
        rotation: [
            fitted.pose.rotation.w,
            fitted.pose.rotation.x,
            fitted.pose.rotation.y,
            fitted.pose.rotation.z,
        ],
        rms_px: fitted.rms_px,
        frames_used: fitted.frames_used,
        frames_rejected: fitted.frames_rejected,
    });

    // The shared centre is fitted from the tags first, then re-fitted with the
    // field model: the first fit is what makes the model match meaningful, since
    // the structure can only be looked for once there is a pose to project it
    // from. Each later round starts from the pose the previous one fitted, so its
    // search window can be tighter and a match cannot be to a nearby stranger.
    let ptz_views = evaluated.views.clone();
    let mut ptz_station = station::fit_ptz(
        &ptz_views,
        &calibration.intrinsics,
        &layout,
        options.max_rms_px,
    );
    if options.tag_layout == "2026-rebuilt-welded" {
        let mut rematched = ptz_views.clone();
        for (round, search_px) in [model::EDGE_SEARCH_WIDE_PX, model::EDGE_SEARCH_TIGHT_PX]
            .into_iter()
            .enumerate()
        {
            let Some(current) = ptz_station.as_ref() else {
                break;
            };
            if current.views.len() != rematched.len() {
                break;
            }
            for (index, (view, (pose, camera))) in
                rematched.iter_mut().zip(&current.views).enumerate()
            {
                let Some(frame) = sampled
                    .iter()
                    .find(|frame| frame.time == evaluated.frames[index].time)
                else {
                    continue;
                };
                view.lines = model::observations(frame, pose, camera, &view.tags, search_px);
            }
            let Some(next) = station::fit_ptz(
                &rematched,
                &calibration.intrinsics,
                &layout,
                options.max_rms_px,
            ) else {
                break;
            };
            // Keep a round only if it explains the frames at least as well: a
            // worse joint fit means the new matches pulled the camera away.
            // A round has to earn its keep. Model matches move the centre a
            // little as they pull it onto real edges; a round that moves it
            // metres for a hundredth of a pixel has been captured by a handful
            // of coincidental matches and is not a refinement of the same
            // camera.
            let moved = (next.position - current.position).norm();
            let better = next.rms_px <= current.rms_px
                && (moved <= MODEL_ROUND_MAX_MOVE_M || next.rms_px <= current.rms_px * 0.9);
            if debug_enabled() {
                eprintln!(
                    "  [model] round {} (±{search_px:.0}px): {} line match(es), rms {:.2}px, sigma {:.2}m{}",
                    round + 1,
                    next.line_samples,
                    next.rms_px,
                    next.sigma_m,
                    if better { "" } else { " (kept the previous fit)" }
                );
            }
            if better {
                ptz_station = Some(next);
            } else {
                break;
            }
        }
    }

    // One camera filmed this file, so every row carries the centre that was
    // fitted for it. A frame whose own solve landed on the other side of the
    // wall — which a lone tag allows, and which happened on Orlando — would
    // otherwise be drawn from somewhere the camera never was, and the side of
    // the field read from it would be read from the wrong end.
    if let Some(fitted) = &ptz_station {
        let (frames, views) = (&mut evaluated.frames, &evaluated.views);
        for (index, (pose, frame_camera)) in fitted.views.iter().enumerate() {
            let (Some(row), Some(view)) = (frames.get_mut(index), views.get(index)) else {
                break;
            };
            row.adopt(pose);
            row.rms_px = solve::reprojection_rms(pose, &view.corners, frame_camera);
            if let Some(entry) = prepared
                .iter()
                .find(|entry| entry.time == row.time)
                .filter(|_| pane == side::Pane::View)
            {
                let verdict = side::verdict(
                    entry.alliance,
                    entry.red_pixels,
                    entry.blue_pixels,
                    &row.tags,
                    &layout,
                    Some(pose),
                );
                row.side = Some(verdict);
            }
        }
    }

    let ptz_station = ptz_station.map(|fitted| report::PtzStationReport {
        position: [fitted.position.x, fitted.position.y, fitted.position.z],
        rms_px: fitted.rms_px,
        model_frames: fitted.model_frames,
        corner_samples: fitted.corner_samples,
        line_samples: fitted.line_samples,
        sigma_m: fitted.sigma_m,
        frames_agreeing: fitted.frames_agreeing,
        frames_stray: fitted.frames_stray,
        frames_unexplained: fitted.frames_unexplained,
        ambiguous: fitted.ambiguous,
        frames: fitted
            .views
            .iter()
            .zip(&evaluated.frames)
            .map(|((pose, camera), frame)| report::PtzFrameReport {
                time: frame.time,
                rotation: [
                    pose.rotation.w,
                    pose.rotation.x,
                    pose.rotation.y,
                    pose.rotation.z,
                ],
                focal_px: camera.fx,
            })
            .collect(),
    });

    Ok(PoseReport {
        video: video.to_path_buf(),
        tag_layout: layout.name.clone(),
        frame_width: width,
        frame_height: height,
        horizontal_fov_deg: 2.0
            * (width as f64 * 0.5 / calibration.intrinsics.fx)
                .atan()
                .to_degrees(),
        focal_px: calibration.intrinsics.fx,
        focal_source: calibration.describe(),
        tag_size_m: options.tag_size_m,
        sampled: sampled.len(),
        pane,
        solved: evaluated.frames,
        missed: evaluated.missed,
        tags_seen,
        min_confidence: options.min_confidence,
        station,
        ptz_station,
    })
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
        // Making the frames easier for the decoder is the default, because the
        // footage it is for is the footage a broadcast actually delivers.
        assert!(options.preprocess);
        // Every solved frame is kept by default; the confidence is reported
        // rather than used to throw evidence away unasked.
        assert_eq!(options.min_confidence, 0.0);
    }

    #[test]
    fn a_missing_video_is_reported_before_anything_else() {
        let error = estimate(Path::new("no-such-video.mp4"), &PoseOptions::default())
            .expect_err("must fail");
        assert!(error.to_string().contains("no such video"), "got: {error}");
    }

    #[test]
    fn invalid_confidence_and_error_limits_are_rejected() {
        for minimum in [-0.1, 1.1, f64::NAN] {
            let options = PoseOptions {
                min_confidence: minimum,
                ..PoseOptions::default()
            };
            assert!(estimate(Path::new("unused"), &options)
                .unwrap_err()
                .to_string()
                .contains("confidence"));
        }
        for maximum in [0.0, -1.0, f64::NAN] {
            let options = PoseOptions {
                max_rms_px: maximum,
                ..PoseOptions::default()
            };
            assert!(estimate(Path::new("unused"), &options)
                .unwrap_err()
                .to_string()
                .contains("reprojection"));
        }
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

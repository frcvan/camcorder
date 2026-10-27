//! Treating the frames of one file as one camera that did not move.
//!
//! Two things are only knowable across frames, and both are worth more than any
//! single frame's answer:
//!
//! * **The focal length.** `--fov` is a guess, and for a single tag the whole
//!   pose is scaled by it: a five percent error in the assumed field of view
//!   moves the camera by half a metre and the corners still fit perfectly. A
//!   frame holding two tags does not have to take the guess — its tags are a
//!   known distance apart on the field, so the only focal length that puts them
//!   there is measurable from the picture. [`calibrate`] measures it and the
//!   estimate then re-solves every frame with it.
//! * **The camera's position.** A broadcast camera is bolted to a stand and
//!   does not wander between sampled frames; the spread of the per-frame poses
//!   is corner noise, not motion. [`fit`] fits one pose to every frame's corners
//!   at once, which averages that noise away, and rejects the frames that do not
//!   belong to it — a pan, or a stray solve.
//!
//! Both are reported next to the per-frame poses rather than replacing them: a
//! camera that really did pan should still be visible as having panned.

use super::{
    solve::{self, Correspondence},
    types::{CameraPose, Intrinsics},
};

/// How far the measured focal length may be from the assumed one. A broadcast
/// camera covers a handful of focal lengths in its zoom range, so a "measurement"
/// outside this is a bad frame, not a lens.
const FOCAL_RANGE: (f64, f64) = (0.5, 2.5);
/// How much better the fit has to explain the corners at the measured focal
/// length before the measurement is believed. Two tags a metre apart pin the
/// focal length well; two tags whose corners are a pixel noisy do not, and this
/// keeps a noise minimum from being reported as a calibration.
const CALIBRATION_GAIN: f64 = 0.75;
/// How many frames have to measure the lens before it is re-solved with their
/// median. One frame is not evidence: on a real broadcast pane, a single frame
/// holding two tags 36cm apart "measured" a focal length at the floor of the
/// allowed range, halved every distance in the video, and put the camera at
/// knee height inside the field. Two frames that have to agree cannot do that.
const MIN_CALIBRATION_FRAMES: usize = 2;
/// A frame is dropped from the station pose when it reprojects worse than this
/// many times the median frame, or worse than the absolute floor. A pan shows
/// up as exactly this: most frames agree, a few are elsewhere.
const FRAME_OUTLIER_FACTOR: f64 = 2.5;
const FRAME_OUTLIER_PX: f64 = 4.0;
const MOST_ROUNDS: usize = 3;

/// One frame's contribution to the cross-frame fits.
#[derive(Debug, Clone)]
pub(crate) struct View {
    /// The corners the per-frame solve agreed on.
    pub corners: Vec<Correspondence>,
    /// The pose the per-frame solve found.
    pub pose: CameraPose,
    /// Tags those corners came from.
    pub tags: Vec<u16>,
}

/// The camera's intrinsics for the whole video, and where they came from.
#[derive(Debug, Clone)]
pub(crate) struct Calibration {
    pub intrinsics: Intrinsics,
    /// Whether the focal length was measured from the footage or assumed.
    pub measured: bool,
    /// Frames that contributed a usable measurement.
    pub frames: usize,
    /// The measured focal length as a multiple of the assumed one.
    pub ratio: f64,
}

impl Calibration {
    /// One line for the report: the focal length and how it was arrived at.
    pub fn describe(&self) -> String {
        if self.measured {
            format!(
                "focal length measured from {} frame(s) with two or more tags: {:.0}px ({:.0}°), {} the --fov guess",
                self.frames,
                self.intrinsics.fx,
                horizontal_fov_deg(&self.intrinsics),
                match (self.ratio - 1.0).abs() {
                    off if off < 0.005 => "within 1% of".to_string(),
                    off if self.ratio > 1.0 => format!("{:.0}% above", off * 100.0),
                    off => format!("{:.0}% below", off * 100.0),
                }
            )
        } else {
            format!(
                "focal length assumed from --fov: {:.0}px ({:.0}°)",
                self.intrinsics.fx,
                horizontal_fov_deg(&self.intrinsics)
            )
        }
    }
}

/// The horizontal field of view the intrinsics describe.
fn horizontal_fov_deg(camera: &Intrinsics) -> f64 {
    2.0 * (camera.width as f64 * 0.5 / camera.fx).atan().to_degrees()
}

/// The same camera with its focal length scaled.
fn with_focal(camera: &Intrinsics, ratio: f64) -> Intrinsics {
    Intrinsics {
        fx: camera.fx * ratio,
        fy: camera.fy * ratio,
        ..*camera
    }
}

/// Measure the focal length from the frames that can measure it.
///
/// A single tag cannot: its four corners fix only the ratio of size to distance,
/// so every focal length has a pose that fits them. Two tags a known distance
/// apart do fix it, which is why only those frames are used. The median across
/// frames is taken rather than the mean, because one frame with a mis-read tag
/// should not move the lens.
pub(crate) fn calibrate(views: &[View], assumed: Intrinsics) -> Calibration {
    let mut ratios = Vec::new();
    for view in views {
        if view.tags.len() < 2 || view.corners.len() < 8 {
            continue;
        }
        let Some(measured) = focal_ratio(view, &assumed) else {
            continue;
        };
        ratios.push(measured);
    }
    if ratios.len() < MIN_CALIBRATION_FRAMES {
        // Below the floor the frames that did measure are kept out of the
        // report's count: nothing was adopted, so nothing was measured.
        return Calibration {
            intrinsics: assumed,
            measured: false,
            frames: 0,
            ratio: 1.0,
        };
    }
    ratios.sort_by(f64::total_cmp);
    let ratio = ratios[ratios.len() / 2];
    let intrinsics = with_focal(&assumed, ratio);
    Calibration {
        intrinsics,
        measured: true,
        frames: ratios.len(),
        ratio,
    }
}

/// The focal-length multiple that best explains one frame's corners.
fn focal_ratio(view: &View, assumed: &Intrinsics) -> Option<f64> {
    let at_assumed = fit_rms(&view.corners, assumed, view.pose)?;
    // A fit that is already perfect cannot be improved on, and a search that
    // starts there would report whatever it landed on as the measurement.
    if at_assumed < 1e-6 {
        return None;
    }

    // Golden-section search in log space: the fit is smooth in the focal
    // length but varies over a factor of a few, so equal steps in the log are
    // equal steps in what the search can see.
    let (low, high) = (FOCAL_RANGE.0.ln(), FOCAL_RANGE.1.ln());
    let golden = 0.618_033_988_749_895;
    let mut a = low;
    let mut b = high;
    let mut c = b - golden * (b - a);
    let mut d = a + golden * (b - a);
    let mut cost_c = cost_at(&view.corners, assumed, view.pose, c);
    let mut cost_d = cost_at(&view.corners, assumed, view.pose, d);
    for _ in 0..28 {
        if cost_c <= cost_d {
            b = d;
            d = c;
            cost_d = cost_c;
            c = b - golden * (b - a);
            cost_c = cost_at(&view.corners, assumed, view.pose, c);
        } else {
            a = c;
            c = d;
            cost_c = cost_d;
            d = a + golden * (b - a);
            cost_d = cost_at(&view.corners, assumed, view.pose, d);
        }
        if !cost_c.is_finite() || !cost_d.is_finite() {
            return None;
        }
    }
    let best = (a + b) * 0.5;
    // A minimum still sitting on the edge of the range is not a measurement:
    // the search ran out of room, so the frame wants a lens outside what a
    // broadcast camera covers. Reporting the bound would hand every other frame
    // the number the search stopped at rather than the one the picture asks for.
    let edge = 1e-3;
    if (best - low).abs() < edge || (high - best).abs() < edge {
        return None;
    }
    let ratio = best.exp();
    let at_best = cost_at(&view.corners, assumed, view.pose, best);
    // The measurement has to be a real improvement: otherwise this frame's
    // corners are too noisy to say anything about the lens, and pretending
    // otherwise would hand every other frame a worse focal length.
    if !at_best.is_finite() || at_best >= at_assumed * CALIBRATION_GAIN {
        return None;
    }
    Some(ratio)
}

fn cost_at(
    corners: &[Correspondence],
    assumed: &Intrinsics,
    start: CameraPose,
    ln_ratio: f64,
) -> f64 {
    let camera = with_focal(assumed, ln_ratio.exp());
    fit_rms(corners, &camera, start).unwrap_or(f64::INFINITY)
}

/// The best fit's reprojection error at one focal length.
fn fit_rms(corners: &[Correspondence], camera: &Intrinsics, start: CameraPose) -> Option<f64> {
    solve::refine(start, corners, camera).map(|solution| solution.rms_px)
}

/// One camera pose for the frames that share it.
#[derive(Debug, Clone)]
pub(crate) struct Station {
    pub pose: CameraPose,
    /// Reprojection error over the frames that were kept, pixels.
    pub rms_px: f64,
    pub frames_used: usize,
    /// Frames whose pose was too far from the others to be the same camera:
    /// a pan, or a solve that went wrong.
    pub frames_rejected: usize,
}

/// Fit one pose to every frame's corners, dropping the frames that disagree.
///
/// Hard rejection of whole frames rather than down-weighting individual
/// corners: the thing that separates one broadcast camera's samples from each
/// other is motion of the whole frame, not a bad corner here and there, and a
/// corner rejected across frames would move the station pose for no reason.
pub(crate) fn fit(views: &[View], camera: &Intrinsics) -> Option<Station> {
    let usable: Vec<&View> = views
        .iter()
        .filter(|view| view.corners.len() >= 4)
        .collect();
    if usable.is_empty() {
        return None;
    }
    // Start from the frame that fits best on its own: it is the one most likely
    // to be the camera's real position, and it puts the bundle fit in the right
    // basin.
    let start = usable
        .iter()
        .filter_map(|view| {
            solve::refine(view.pose, &view.corners, camera)
                .map(|solution| (solution.rms_px, view.pose))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, pose)| pose)
        .or(Some(usable[0].pose))?;

    let mut keep: Vec<usize> = (0..usable.len()).collect();
    let mut solution = None;
    for _ in 0..MOST_ROUNDS {
        let corners: Vec<Correspondence> = keep
            .iter()
            .flat_map(|index| usable[*index].corners.iter().copied())
            .collect();
        let fitted = solve::refine(start, &corners, camera)?;
        solution = Some(fitted.clone());
        if keep.len() == 1 {
            break;
        }

        let errors: Vec<f64> = keep
            .iter()
            .map(|index| solve::reprojection_rms(&fitted.pose, &usable[*index].corners, camera))
            .collect();
        let mut sorted = errors.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        let limit = (median * FRAME_OUTLIER_FACTOR).max(median.max(FRAME_OUTLIER_PX));
        let kept: Vec<usize> = keep
            .iter()
            .zip(errors.iter())
            .filter(|(_, error)| **error <= limit)
            .map(|(index, _)| *index)
            .collect();
        if kept.len() == keep.len() || kept.is_empty() {
            break;
        }
        keep = kept;
    }

    let solution = solution?;
    Some(Station {
        pose: solution.pose,
        rms_px: solution.rms_px,
        frames_used: keep.len(),
        frames_rejected: usable.len() - keep.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{
        layout::TagLayout,
        solve,
        types::{Mat3, Quat, Vec3},
    };

    fn facing_tag(id: u16, at: Vec3) -> (u16, Vec3, Quat) {
        let rotation =
            Mat3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -std::f64::consts::FRAC_PI_2)
                .to_quaternion();
        (id, at, rotation)
    }

    fn test_layout(tags: &[(u16, Vec3, Quat)]) -> TagLayout {
        let entries: Vec<serde_json::Value> = tags
            .iter()
            .map(|(id, translation, rotation)| {
                serde_json::json!({
                    "ID": id,
                    "pose": {
                        "translation": {"x": translation.x, "y": translation.y, "z": translation.z},
                        "rotation": {"quaternion": {"W": rotation.w, "X": rotation.x, "Y": rotation.y, "Z": rotation.z}}
                    }
                })
            })
            .collect();
        let json = serde_json::json!({"field": {"length": 16.0, "width": 8.0}, "tags": entries});
        TagLayout::from_json("test", &json.to_string()).expect("test layout")
    }

    /// Corners of the given tags, as seen from `pose` through `camera`.
    fn corners_of(
        layout: &TagLayout,
        ids: &[u16],
        tag_size: f64,
        pose: CameraPose,
        camera: &Intrinsics,
    ) -> Vec<Correspondence> {
        let rotation = Mat3::from_quaternion(pose.rotation);
        let mut corners = Vec::new();
        for id in ids {
            for corner in layout.corners(*id, tag_size).expect("corners") {
                let point = rotation.mul_vec(corner - pose.position);
                let (u, v) = camera.project(point).expect("in front of the camera");
                corners.push(Correspondence {
                    field: corner,
                    image: (u, v),
                    tag: *id,
                });
            }
        }
        corners
    }

    fn look_at(position: Vec3, target: Vec3) -> CameraPose {
        solve::look_towards(position, target)
    }

    #[test]
    fn two_tags_measure_the_focal_length_the_assumed_one_got_wrong() {
        // A 1920-wide frame at 55° is 1841px. If the run was told 75°, it would
        // assume 1253px and put every pose 40% too close. The tag pair is the
        // only thing in the picture that knows better.
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
        ]);
        let truth = Intrinsics::from_horizontal_fov(1920, 1080, 55.0);
        let pose = look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(7.5, 6.0, 0.5));
        let assumed = Intrinsics::from_horizontal_fov(1920, 1080, 75.0);

        let corners = corners_of(&layout, &[3, 4], 0.1651, pose, &truth);
        let views: Vec<View> = (0..3)
            .map(|_| View {
                corners: corners.clone(),
                pose,
                tags: vec![3, 4],
            })
            .collect();
        let calibration = calibrate(&views, assumed);
        assert!(calibration.measured, "two tags should measure the lens");
        assert_eq!(calibration.frames, 3);
        assert!(
            (calibration.intrinsics.fx - truth.fx).abs() / truth.fx < 0.02,
            "measured {:.0}px against {:.0}px",
            calibration.intrinsics.fx,
            truth.fx
        );

        // …and the re-solve through the measured lens recovers the camera,
        // which the assumed one could not have.
        let before = solve::refine(pose, &views[0].corners, &assumed).expect("solves");
        let after =
            solve::refine(pose, &views[0].corners, &calibration.intrinsics).expect("solves");
        assert!(
            (after.pose.position - pose.position).norm() < 0.05,
            "measured lens left the camera {:.2}m out",
            (after.pose.position - pose.position).norm()
        );
        assert!(
            (before.pose.position - pose.position).norm() > 0.5,
            "the assumed lens should have been visibly wrong, was {:.2}m out",
            (before.pose.position - pose.position).norm()
        );
    }

    #[test]
    fn one_frame_of_two_tags_is_not_enough_to_set_the_lens() {
        // The same clean pair, in a single frame. One sample is not a
        // measurement: there is nothing for it to agree with, so a frame whose
        // corners happen to favour another lens would move every other frame.
        // On real footage that is exactly what happened — one pane frame with
        // two nearby tags halved the focal length of the whole video.
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
        ]);
        let truth = Intrinsics::from_horizontal_fov(1920, 1080, 55.0);
        let pose = look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(7.5, 6.0, 0.5));
        let assumed = Intrinsics::from_horizontal_fov(1920, 1080, 75.0);
        let views = vec![View {
            corners: corners_of(&layout, &[3, 4], 0.1651, pose, &truth),
            pose,
            tags: vec![3, 4],
        }];

        let calibration = calibrate(&views, assumed);
        assert!(
            !calibration.measured,
            "one frame must not calibrate the lens"
        );
        assert_eq!(calibration.intrinsics, assumed);
    }

    #[test]
    fn a_measurement_that_runs_out_of_range_is_not_a_measurement() {
        // A lens nearly three times the guess is not a lens any broadcast
        // camera has; it is a frame that fits better when the search is pushed
        // to its limit. The bound is where the search stopped, not what the
        // picture says, so the guess is kept instead.
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
        ]);
        let truth = Intrinsics::from_horizontal_fov(1920, 1080, 30.0);
        let pose = look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(7.5, 6.0, 0.5));
        let assumed = Intrinsics::from_horizontal_fov(1920, 1080, 72.0);
        let corners = corners_of(&layout, &[3, 4], 0.1651, pose, &truth);
        let views: Vec<View> = (0..3)
            .map(|_| View {
                corners: corners.clone(),
                pose,
                tags: vec![3, 4],
            })
            .collect();

        let calibration = calibrate(&views, assumed);
        assert!(
            !calibration.measured,
            "measured {:.0}px, which is the edge of the range",
            calibration.intrinsics.fx
        );
        assert_eq!(calibration.intrinsics, assumed);
    }

    #[test]
    fn one_tag_cannot_measure_a_lens() {
        // A single tag's four corners fit at every focal length, each with its
        // own distance. Nothing in the picture says which is right, so the
        // assumed value is kept rather than a number invented from noise.
        let layout = test_layout(&[facing_tag(13, Vec3::new(15.5, 6.0, 0.5))]);
        let truth = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let pose = look_at(Vec3::new(11.0, 2.0, 4.0), Vec3::new(15.5, 6.0, 0.5));
        let views = vec![View {
            corners: corners_of(&layout, &[13], 0.1651, pose, &truth),
            pose,
            tags: vec![13],
        }];
        let assumed = Intrinsics::from_horizontal_fov(1280, 720, 45.0);
        let calibration = calibrate(&views, assumed);
        assert!(!calibration.measured, "one tag must not calibrate the lens");
        assert_eq!(calibration.intrinsics, assumed);
        assert_eq!(calibration.frames, 0);
    }

    #[test]
    fn the_station_pose_averages_the_frames_that_share_it() {
        // Four frames of one camera with corner noise: each frame's own pose is
        // a metre or so out, and the pose fitted to all of them together is not.
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
        ]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let truth = look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(7.5, 6.0, 0.5));

        let mut seed = 11u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f64 / (1u32 << 24) as f64) - 0.5
        };
        let mut views = Vec::new();
        for _ in 0..4 {
            let mut corners = corners_of(&layout, &[3, 4], 0.1651, truth, &camera);
            for corner in corners.iter_mut() {
                corner.image.0 += noise();
                corner.image.1 += noise();
            }
            let solution = solve::refine(truth, &corners, &camera).expect("solves");
            views.push(View {
                corners,
                pose: solution.pose,
                tags: vec![3, 4],
            });
        }

        // One frame from another place entirely: a pan, or a bad solve.
        let elsewhere = look_at(Vec3::new(10.0, -3.0, 6.0), Vec3::new(7.5, 6.0, 0.5));
        let mut stray = corners_of(&layout, &[3, 4], 0.1651, elsewhere, &camera);
        for corner in stray.iter_mut() {
            corner.image.0 += noise();
            corner.image.1 += noise();
        }
        views.push(View {
            corners: stray,
            pose: elsewhere,
            tags: vec![3, 4],
        });

        let station = fit(&views, &camera).expect("a station");
        assert_eq!(station.frames_used, 4, "the stray frame should be dropped");
        assert_eq!(station.frames_rejected, 1);
        let error = (station.pose.position - truth.position).norm();
        assert!(error < 0.1, "station pose {error:.2}m from the truth");
        assert!(station.rms_px < 1.0, "station rms {:.2}px", station.rms_px);
    }

    #[test]
    fn a_segment_of_one_frame_is_still_a_station() {
        let layout = test_layout(&[facing_tag(13, Vec3::new(15.5, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let pose = look_at(Vec3::new(11.0, 2.0, 4.0), Vec3::new(15.5, 6.0, 0.5));
        let corners = corners_of(&layout, &[13], 0.1651, pose, &camera);
        let views = vec![View {
            corners,
            pose,
            tags: vec![13],
        }];
        let station = fit(&views, &camera).expect("a station");
        assert_eq!(station.frames_used, 1);
        assert_eq!(station.frames_rejected, 0);
        assert!(fit(&[], &camera).is_none());
    }
}

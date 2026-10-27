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
    layout::TagLayout,
    solve::{self, Correspondence},
    trust,
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
/// Measurements farther apart than 20% do not describe the same lens.
const MAX_FOCAL_DISAGREEMENT: f64 = 1.2;
/// A frame is dropped from the station pose when it reprojects worse than this
/// many times the median frame, or worse than the absolute floor. A pan shows
/// up as exactly this: most frames agree, a few are elsewhere.
const FRAME_OUTLIER_FACTOR: f64 = 2.5;
const FRAME_OUTLIER_PX: f64 = 4.0;
/// Similar-looking isolated tags can reproject under cameras metres apart.
/// A station needs agreement in 3D as well as in the image.
const STATION_POSITION_TOLERANCE_M: f64 = 3.0;
const STATION_ROTATION_TOLERANCE_DEG: f64 = 25.0;
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
    /// Image samples associated with unambiguous field-model lines.
    pub lines: Vec<solve::LineObservation>,
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
    let Some((ratio, frames)) = agreeing_focal_ratio(&mut ratios) else {
        return Calibration {
            intrinsics: assumed,
            measured: false,
            frames: 0,
            ratio: 1.0,
        };
    };
    Calibration {
        intrinsics: with_focal(&assumed, ratio),
        measured: true,
        frames,
        ratio,
    }
}

/// A majority of independent measurements must agree before setting the lens.
fn agreeing_focal_ratio(ratios: &mut [f64]) -> Option<(f64, usize)> {
    ratios.sort_by(f64::total_cmp);
    let mut best = (0, 0);
    let mut end = 0;
    for start in 0..ratios.len() {
        while end < ratios.len() && ratios[end] <= ratios[start] * MAX_FOCAL_DISAGREEMENT {
            end += 1;
        }
        if end - start > best.1 - best.0 {
            best = (start, end);
        }
    }
    let count = best.1 - best.0;
    if count < MIN_CALIBRATION_FRAMES || count * 2 <= ratios.len() {
        return None;
    }
    Some((ratios[best.0 + count / 2], count))
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
fn same_station(a: &CameraPose, b: &CameraPose) -> bool {
    let a = a.rotation;
    let b = b.rotation;
    let dot = (a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z)
        .abs()
        .clamp(0.0, 1.0);
    2.0 * dot.acos().to_degrees() <= STATION_ROTATION_TOLERANCE_DEG
}

pub(crate) fn fit(
    views: &[View],
    camera: &Intrinsics,
    layout: &TagLayout,
    max_rms_px: f64,
) -> Option<Station> {
    let usable: Vec<&View> = views
        .iter()
        .filter(|view| view.corners.len() >= 4)
        .collect();
    if usable.is_empty() {
        return None;
    }
    // Seed from the pose explaining the most frames before any joint fit. A
    // least-squares compromise over different cameras inflates the median error
    // and can make every frame look like an inlier.
    let (mut start, mut keep) = usable
        .iter()
        .map(|seed| {
            let inliers: Vec<usize> = usable
                .iter()
                .enumerate()
                .filter(|(_, view)| {
                    (seed.pose.position - view.pose.position).norm() <= STATION_POSITION_TOLERANCE_M
                        && same_station(&seed.pose, &view.pose)
                        && solve::reprojection_rms(&seed.pose, &view.corners, camera)
                            <= FRAME_OUTLIER_PX
                })
                .map(|(index, _)| index)
                .collect();
            let error: f64 = inliers
                .iter()
                .map(|index| solve::reprojection_rms(&seed.pose, &usable[*index].corners, camera))
                .sum();
            (seed.pose, inliers, error)
        })
        .max_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| b.2.total_cmp(&a.2)))
        .map(|(pose, inliers, _)| (pose, inliers))?;
    if keep.is_empty() || keep.len() * 2 <= usable.len() {
        return None;
    }
    let mut solution = None;
    for _ in 0..MOST_ROUNDS {
        let corners: Vec<Correspondence> = keep
            .iter()
            .flat_map(|index| usable[*index].corners.iter().copied())
            .collect();
        let fitted = solve::refine(start, &corners, camera)?;
        start = fitted.pose;
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
        if kept.len() == keep.len() {
            break;
        }
        if kept.is_empty() || kept.len() * 2 <= usable.len() {
            return None;
        }
        keep = kept;
        // The last rejection still needs a fit to the retained frames.
        solution = None;
    }

    let solution = match solution {
        Some(solution) => solution,
        None => {
            let corners: Vec<Correspondence> = keep
                .iter()
                .flat_map(|index| usable[*index].corners.iter().copied())
                .collect();
            solve::refine(start, &corners, camera)?
        }
    };
    if solution.rms_px > max_rms_px
        || keep
            .iter()
            .any(|index| trust::plausible(&solution.pose, layout, &usable[*index].tags).is_err())
    {
        return None;
    }
    Some(Station {
        pose: solution.pose,
        rms_px: solution.rms_px,
        frames_used: keep.len(),
        frames_rejected: usable.len() - keep.len(),
    })
}

/// How close two frames' own solved positions have to be to count as agreeing
/// on the same camera. A fixed camera's frames scatter by their corner noise,
/// which for a small tag is tens of centimetres; a different camera is metres.
const CONSENSUS_RADIUS_M: f64 = 1.0;
/// How far a joint refinement may move the centre from where the frames agree
/// before it is not believed. A fit that wanders across the field has been
/// captured by the handful of frames that fit its basin, not by the camera.
const RUNAWAY_M: f64 = 3.0;
/// Farthest displacement the uncertainty probe tests, metres. A position whose
/// error bound reaches this is reported as "not pinned" rather than given a
/// number that the footage cannot support.
const PTZ_PROBE_M: f64 = 2.0;
/// How many bisection steps turn the probe into a bound: 2m / 2^5 = 6cm.
const PTZ_PROBE_STEPS: usize = 5;
/// Squared reprojection error a displaced camera may add and still count as an
/// explanation of the same footage. 0.5px per corner is the precision small
/// broadcast tags actually carry.
const PTZ_SLACK_PX2: f64 = 0.25;
/// A centre not pinned any better than this is reported, but flagged.
const PTZ_AMBIGUOUS_M: f64 = 0.5;

/// A fixed centre and per-frame orientation and focal length. This is separate
/// from the static-pose station fit: panning does not imply camera translation.
#[derive(Debug, Clone)]
pub(crate) struct PtzStation {
    pub position: super::types::Vec3,
    pub rms_px: f64,
    /// Frames carrying at least one image-to-model line match.
    pub model_frames: usize,
    /// Tag corner observations the fit was based on.
    pub corner_samples: usize,
    /// Image-to-model line observations the fit was based on.
    pub line_samples: usize,
    /// Largest displacement the footage could not rule out, metres. Censored at
    /// [`PTZ_PROBE_M`]. This is the honest width of the answer: a broadcast pane
    /// holding one small tag leaves metres, and the report has to say so rather
    /// than quote a centre to the centimetre.
    pub sigma_m: f64,
    /// Frames whose own solve put the camera within a metre of this centre: the
    /// evidence the centre rests on. Note a lone tag solves as well from the far
    /// side of the wall, so on a single-tag pane the frames split into two
    /// basins and this counts only the one the answer came from.
    pub frames_agreeing: usize,
    /// Frames whose own solve put the camera somewhere else and which were
    /// therefore re-aimed at this centre before being re-solved.
    pub frames_stray: usize,
    /// Frames the fitted camera still does not explain, after re-aiming.
    pub frames_unexplained: usize,
    /// Whether `sigma_m` is wide enough that the centre is not a measurement.
    pub ambiguous: bool,
    pub views: Vec<(CameraPose, Intrinsics)>,
}

/// Fit one camera centre to every frame, allowing each frame its own orientation
/// and zoom, and report the best estimate together with what the footage could
/// not rule out.
///
/// A fit is not rejected for being weak. On real broadcast panes the evidence is
/// often a couple of small tags, and returning nothing there loses the one thing
/// the run did measure; instead the answer ships with `sigma_m`, the largest
/// displacement that still explains the pictures. Callers that need a real
/// measurement must check `ambiguous`.
///
/// The probe has to re-optimize rotation and zoom at each displaced centre: the
/// displacement can be absorbed by re-aiming, so a probe that held the other
/// parameters fixed would report a precision the fit does not have.
pub(crate) fn fit_ptz(
    views: &[View],
    camera: &Intrinsics,
    layout: &TagLayout,
    max_rms: f64,
) -> Option<PtzStation> {
    if views.len() < 2 {
        return None;
    }
    // Where the frames independently say the camera was, and how many of them
    // say it. A single tag is solved twice as well from the camera on the far
    // side of the wall, so the per-frame answers split into basins; the camera
    // is the basin most of the frames are in. Choosing by reprojection error
    // instead lets two frames that agree with each other and disagree with the
    // rest pull the answer into the empty middle of the field.
    let (agreed, support) = consensus_position(views);
    // A frame whose own solve put the camera somewhere else does not get to keep
    // its orientation: that orientation was fitted from a position the camera was
    // never at, so from the real centre it points nowhere useful — on Orlando it
    // pointed away from the tag, leaving it behind the camera entirely. The
    // nearest frame that did agree has an aim worth starting from, and the
    // refinement below moves it onto the corners this frame actually saw.
    let stray: Vec<bool> = views
        .iter()
        .map(|view| (view.pose.position - agreed).norm() > CONSENSUS_RADIUS_M)
        .collect();
    let seed_at = |position: super::types::Vec3| -> Vec<solve::PtzView> {
        views
            .iter()
            .enumerate()
            .map(|(index, view)| solve::PtzView {
                corners: view.corners.clone(),
                lines: view.lines.clone(),
                pose: CameraPose {
                    position,
                    rotation: if stray[index] {
                        nearby_aim(views, &stray, view.pose.position).unwrap_or(view.pose.rotation)
                    } else {
                        view.pose.rotation
                    },
                },
                camera: *camera,
            })
            .collect()
    };

    // Refine the centre as well as each frame's aim and zoom — but only believe a
    // refinement that stays where the frames put it. Otherwise the centre is
    // held at the agreement and only the aims and zoom are fitted.
    let refined = solve::refine_ptz(&seed_at(agreed), false)
        .filter(|fitted| {
            (fitted[0].pose.position - agreed).norm() <= RUNAWAY_M
                && fitted[0].pose.position.z >= 0.0
        })
        .filter(|fitted| !implausible(fitted, views, camera, layout));
    let fitted = match refined {
        Some(fitted) => fitted,
        None => {
            let fixed = solve::refine_ptz(&seed_at(agreed), true);
            let held = fixed.filter(|fitted| !implausible(fitted, views, camera, layout))?;
            if super::debug_enabled() {
                eprintln!(
                    "  [fixed] refinement ran away from the frames' agreement at ({:.2}, {:.2}, {:.2})m; holding the centre",
                    agreed.x, agreed.y, agreed.z
                );
            }
            held
        }
    };
    if super::debug_enabled() {
        eprintln!(
            "  [fixed] {} frame(s): {support} agree on ({:.2}, {:.2}, {:.2})m, fitted ({:.2}, {:.2}, {:.2})m at rms {:.2}px",
            views.len(),
            agreed.x, agreed.y, agreed.z,
            fitted[0].pose.position.x, fitted[0].pose.position.y, fitted[0].pose.position.z,
            ptz_rms(&fitted)
        );
    }
    let rms = ptz_rms(&fitted);
    if !rms.is_finite() {
        return None;
    }
    // Frames the fitted camera still cannot explain are counted rather than
    // quietly kept: they are frames whose own solve was on the other branch, and
    // if re-aiming at the real centre did not bring them back, the honest answer
    // is that this run does not know where those frames were taken from.
    let unexplained = fitted
        .iter()
        .filter(|view| {
            let (cost, count) = solve::ptz_error(view);
            (cost / count.max(1) as f64).sqrt() > max_rms
        })
        .count();
    let centre = fitted[0].pose.position;
    let tolerance = rms * rms + PTZ_SLACK_PX2;

    let mut sigma = 0.0f64;
    let directions = probe_directions();
    for direction in directions {
        let bound = displacement_bound(&fitted, centre, direction, tolerance);
        if bound > sigma {
            sigma = bound;
        }
        if sigma >= PTZ_PROBE_M {
            break;
        }
    }
    let sigma_m = sigma.min(PTZ_PROBE_M);

    let corner_samples = fitted.iter().map(|view| view.corners.len()).sum::<usize>();
    let line_samples = fitted
        .iter()
        .map(|view| view.lines.len().min(4))
        .sum::<usize>();
    Some(PtzStation {
        position: centre,
        rms_px: rms,
        model_frames: fitted.iter().filter(|view| !view.lines.is_empty()).count(),
        corner_samples,
        line_samples,
        sigma_m,
        frames_agreeing: support,
        frames_stray: stray.iter().filter(|stray| **stray).count(),
        frames_unexplained: unexplained,
        // A fit that does not even reproject the frames within the run's own
        // error limit is a best estimate, not a measurement, however tight the
        // displacement probe looks around it.
        ambiguous: sigma_m >= PTZ_AMBIGUOUS_M || rms > max_rms || unexplained > 0,
        views: fitted.iter().map(|view| (view.pose, view.camera)).collect(),
    })
}

/// Where most of the frames put the camera, and how many put it there.
///
/// The candidate is each frame's own solved centre in turn; the winner is the
/// one with the most other frames within [`CONSENSUS_RADIUS_M`], and the answer
/// is the mean of that group. A mean over a majority is what makes this robust
/// where a joint fit is not: agreement is counted, so two frames that agree with
/// each other cannot outvote ten that agree on somewhere else, however well
/// those two reproject.
/// The aim of the nearest frame that agreed with the centre, for re-solving a
/// frame that landed elsewhere.
fn nearby_aim(
    views: &[View],
    stray: &[bool],
    from: super::types::Vec3,
) -> Option<super::types::Quat> {
    views
        .iter()
        .zip(stray)
        .filter(|(_, stray)| !**stray)
        .min_by(|(a, _), (b, _)| {
            (a.pose.position - from)
                .norm()
                .total_cmp(&(b.pose.position - from).norm())
        })
        .map(|(view, _)| view.pose.rotation)
}

fn consensus_position(views: &[View]) -> (super::types::Vec3, usize) {
    let mut best = (views[0].pose.position, 0usize);
    for view in views {
        let agreeing: Vec<super::types::Vec3> = views
            .iter()
            .map(|other| other.pose.position)
            .filter(|other| (*other - view.pose.position).norm() <= CONSENSUS_RADIUS_M)
            .collect();
        if agreeing.len() > best.1 {
            let centre = agreeing
                .iter()
                .fold(super::types::Vec3::new(0.0, 0.0, 0.0), |sum, p| sum + *p)
                * (1.0 / agreeing.len() as f64);
            best = (centre, agreeing.len());
        }
    }
    best
}

/// Whether any frame's fitted pose is one no camera could have had, or a lens no
/// camera has. Returns the reason for the report's debug output.
fn implausible(
    fitted: &[solve::PtzView],
    views: &[View],
    camera: &Intrinsics,
    layout: &TagLayout,
) -> bool {
    fitted.iter().zip(views).any(|(view, original)| {
        trust::plausible(&view.pose, layout, &original.tags).is_err()
            || view.camera.fx < camera.fx * FOCAL_RANGE.0
            || view.camera.fx > camera.fx * FOCAL_RANGE.1
    })
}

/// The directions the centre is pushed to find what the footage rules out: six
/// axes and the four space diagonals, so a slope no axis test can see is caught.
fn probe_directions() -> Vec<super::types::Vec3> {
    use super::types::Vec3;
    let d = 1.0 / 3.0f64.sqrt();
    vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(d, d, d),
        Vec3::new(d, d, -d),
        Vec3::new(d, -d, d),
        Vec3::new(-d, d, d),
    ]
}

/// The farthest the centre can move along `direction` and still explain the
/// frames, by bisection between "no displacement" (which always fits, being the
/// fit itself) and [`PTZ_PROBE_M`]. Returns the probe distance when even that far
/// is not ruled out — the answer is then "more than this", and the caller
/// censors it.
fn displacement_bound(
    fitted: &[solve::PtzView],
    centre: super::types::Vec3,
    direction: super::types::Vec3,
    tolerance: f64,
) -> f64 {
    let fits_within = |distance: f64| -> bool {
        let mut displaced = fitted.to_vec();
        for view in &mut displaced {
            view.pose.position = centre + direction * distance;
        }
        match solve::refine_ptz(&displaced, true) {
            Some(alternative) => ptz_rms(&alternative) <= tolerance.sqrt(),
            // A displaced pose that will not even optimize is not evidence that
            // the displacement is wrong, so it does not narrow the bound.
            None => false,
        }
    };
    if fits_within(PTZ_PROBE_M) {
        return PTZ_PROBE_M;
    }
    let (mut low, mut high) = (0.0f64, PTZ_PROBE_M);
    for _ in 0..PTZ_PROBE_STEPS {
        let middle = 0.5 * (low + high);
        if fits_within(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn ptz_rms(views: &[solve::PtzView]) -> f64 {
    let mut cost = 0.0;
    let mut count = 0;
    for view in views {
        let (error, samples) = solve::ptz_error(view);
        cost += error;
        count += samples;
    }
    (cost / count.max(1) as f64).sqrt()
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
    fn fixed_position_survives_pan_and_zoom_when_geometry_constrains_it() {
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(3.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
            facing_tag(5, Vec3::new(5.0, 6.0, 2.5)),
        ]);
        let assumed = Intrinsics::from_horizontal_fov(1920, 1080, 65.0);
        let position = Vec3::new(2.0, 1.0, 4.0);
        let views: Vec<View> = [(4.0, 50.0), (6.0, 65.0), (8.0, 80.0)]
            .into_iter()
            .map(|(target, fov)| {
                let pose = look_at(position, Vec3::new(target, 6.0, 0.5));
                let camera = Intrinsics::from_horizontal_fov(1920, 1080, fov);
                View {
                    corners: corners_of(&layout, &[3, 4, 5], 0.1651, pose, &camera),
                    pose,
                    tags: vec![3, 4, 5],
                    lines: vec![],
                }
            })
            .collect();
        let fitted = fit_ptz(&views, &assumed, &layout, 8.0).expect("well-constrained camera");
        assert!((fitted.position - position).norm() < 0.05);
        assert!(
            !fitted.ambiguous,
            "three tags across three zoom levels should pin the centre, got ±{:.1}m",
            fitted.sigma_m
        );
        for ((_, camera), fov) in fitted.views.iter().zip([50.0, 65.0, 80.0]) {
            let truth = Intrinsics::from_horizontal_fov(1920, 1080, fov);
            assert!((camera.fx - truth.fx).abs() / truth.fx < 0.01);
        }
    }

    #[test]
    fn small_single_tags_do_not_claim_a_measured_fixed_centre() {
        let layout = test_layout(&[facing_tag(3, Vec3::new(6.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(640, 360, 60.0);
        let position = Vec3::new(2.0, -8.0, 4.0);
        let views: Vec<View> = [5.5, 6.0, 6.5]
            .into_iter()
            .map(|target| {
                let pose = look_at(position, Vec3::new(target, 6.0, 0.5));
                View {
                    corners: corners_of(&layout, &[3], 0.1651, pose, &camera),
                    pose,
                    tags: vec![3],
                    lines: vec![],
                }
            })
            .collect();
        // The answer is still reported — the run did fit something — but it has
        // to say that a camera metres away explains the same pixels, which is
        // what happens when one small tag is the whole picture. Quoting this
        // centre as a position would be the mistake the flag exists to prevent.
        let fitted = fit_ptz(&views, &camera, &layout, 8.0).expect("a fit is still reported");
        assert!(
            fitted.ambiguous,
            "one small tag pinned the centre to ±{:.2}m, which it cannot",
            fitted.sigma_m
        );
        assert!(fitted.sigma_m >= super::PTZ_AMBIGUOUS_M);
    }

    /// The model's structure box for a tag, as field-coordinate edges.
    fn structure_segments(id: u16) -> Vec<[[f64; 3]; 2]> {
        #[derive(serde::Deserialize)]
        struct Entry {
            id: u16,
            segments: Vec<[[f64; 3]; 2]>,
        }
        let entries: Vec<Entry> = serde_json::from_str(include_str!(
            "../../assets/field/2026-rebuilt-advantagescope/structure.json"
        ))
        .expect("the bundled structure parses");
        entries
            .into_iter()
            .find(|entry| entry.id == id)
            .expect("the tag has structure")
            .segments
    }

    #[test]
    fn a_frame_solved_to_the_other_side_of_the_wall_is_re_aimed_at_the_camera() {
        // Four frames of one fixed camera watching one small tag. Three solved
        // where the camera was; the fourth, whose four corners fit equally well
        // from the far side of the wall, landed at the field centre instead —
        // which is what happened on Orlando, where two broadcast panes came back
        // with their cameras in the middle of the field. The camera is where the
        // frames agree, not where the best reprojection is: one frame must not
        // be able to move it, and the stray frame is re-aimed at the real centre
        // rather than left pointing at nothing.
        let layout = TagLayout::load("2026-rebuilt-welded").expect("the bundled layout");
        let tag = layout.tag(5).expect("tag 5 is in the layout");
        let normal = Mat3::from_quaternion(tag.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        let truth_position = tag.translation + normal * 9.0 + Vec3::new(0.0, 0.0, 6.0);
        let assumed = Intrinsics::from_horizontal_fov(1280, 720, 60.0);

        let mut views = Vec::new();
        for offset in [-1.5, 0.0, 1.5] {
            let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
            let pose = look_at(
                truth_position,
                Vec3::new(
                    tag.translation.x,
                    tag.translation.y + offset,
                    tag.translation.z,
                ),
            );
            views.push(View {
                corners: corners_of(&layout, &[5], 0.1651, pose, &camera),
                pose,
                tags: vec![5],
                lines: Vec::new(),
            });
        }
        // The same camera, but this frame's solve picked the other branch: the
        // corners are the ones the camera really saw, the pose is not. Its aim
        // points the wrong way from the real centre — on Orlando those frames
        // ended up with the tag *behind* the camera — so keeping that aim and
        // only moving the centre cannot explain the frame, which is what makes
        // re-aiming it from a frame that did agree the necessary step.
        // The aim is the part that matters: from the real centre it points
        // straight away from the tag, the way the Orlando strays did. There the
        // tag is behind the camera, and the solver's residual for a point behind
        // it is a flat penalty — no slope to descend — so moving the centre
        // alone leaves the frame unexplained forever.
        let away = look_at(truth_position, truth_position + normal * 10.0);
        views.push(View {
            corners: views[1].corners.clone(),
            pose: CameraPose {
                position: Vec3::new(8.0, 0.0, 2.3),
                rotation: away.rotation,
            },
            tags: vec![5],
            lines: Vec::new(),
        });

        let station = fit_ptz(&views, &assumed, &layout, 8.0).expect("a fit is reported");
        let error = (station.position - truth_position).norm();
        assert!(
            error < 0.25,
            "the stray frame moved the camera {error:.2}m off, to ({:.2}, {:.2}, {:.2})",
            station.position.x,
            station.position.y,
            station.position.z
        );
        assert_eq!(station.frames_agreeing, 3);
        assert_eq!(station.frames_stray, 1);
        assert_eq!(
            station.frames_unexplained, 0,
            "the stray frame's corners are the camera's own and must be explained from it"
        );
        // …and every frame now carries the one centre the camera stood at.
        for (pose, _) in &station.views {
            assert!((pose.position - station.position).norm() < 1e-9);
        }
    }

    #[test]
    fn the_structure_of_a_small_tag_is_what_fixes_the_camera() {
        // One 16cm tag, eleven metres away, in three frames of a panning and
        // zooming camera. Its four corners are all the same plane and all the
        // same size-to-distance ratio, so a camera further down the view ray
        // with a longer lens reproduces them, and the fit is flat along that
        // ray. The structure the tag is bolted to is not on that plane: when the
        // camera moves, the top of the mount sweeps across the picture. That is
        // what these observations add, and without them the run has to say so.
        let layout = TagLayout::load("2026-rebuilt-welded").expect("the bundled layout");
        // Tag 5 stands on the three-metre structure at the red end of the field,
        // which is the kind of thing a broadcast pane actually shows: the tag
        // itself is a few pixels, the tower it is bolted to is not.
        let segments = structure_segments(5);
        assert!(segments.len() >= 8, "a box outline, got {}", segments.len());
        let tag = layout.tag(5).expect("tag 5 is in the layout");
        let aim = tag.translation;
        // Stand where the tag can actually be read: out along the front of its
        // face, up where a fixed camera on a stand would be.
        let normal = Mat3::from_quaternion(tag.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        let truth_position = aim + normal * 9.0 + Vec3::new(0.0, 0.0, 6.0);
        let assumed = Intrinsics::from_horizontal_fov(1280, 720, 60.0);

        let mut plain = Vec::new();
        let mut with_model = Vec::new();
        for (offset, fov) in [(-1.5, 55.0), (0.0, 60.0), (1.5, 70.0)] {
            let camera = Intrinsics::from_horizontal_fov(1280, 720, fov);
            let pose = look_at(truth_position, Vec3::new(aim.x, aim.y + offset, aim.z));
            let corners = corners_of(&layout, &[5], 0.1651, pose, &camera);
            assert!(
                corners.len() == 4,
                "one tag should give four corners, got {}",
                corners.len()
            );
            let lines: Vec<solve::LineObservation> = segments
                .iter()
                .filter_map(|segment| {
                    let start = Vec3::from_array(segment[0]);
                    let end = Vec3::from_array(segment[1]);
                    let middle = start + (end - start) * 0.5;
                    let image = camera.project(pose.to_camera(middle))?;
                    Some(solve::LineObservation { start, end, image })
                })
                .collect();
            assert!(!lines.is_empty(), "the structure should project into shot");
            plain.push(View {
                corners: corners.clone(),
                pose,
                tags: vec![5],
                lines: Vec::new(),
            });
            with_model.push(View {
                corners,
                pose,
                tags: vec![5],
                lines,
            });
        }

        let tags_only = fit_ptz(&plain, &assumed, &layout, 8.0).expect("a fit is reported");
        assert!(
            tags_only.ambiguous,
            "one small tag pinned the camera to ±{:.2}m on its own, which one tag cannot do",
            tags_only.sigma_m
        );

        let pinned = fit_ptz(&with_model, &assumed, &layout, 8.0).expect("a fit is reported");
        assert!(
            !pinned.ambiguous,
            "the structure should fix the camera, still ±{:.2}m ({} line sample(s))",
            pinned.sigma_m, pinned.line_samples
        );
        assert!(
            pinned.sigma_m < tags_only.sigma_m,
            "structure knowledge left the bound no tighter: {:.2}m against {:.2}m",
            pinned.sigma_m,
            tags_only.sigma_m
        );
        let error = (pinned.position - truth_position).norm();
        assert!(
            error < 0.25,
            "structural evidence left the camera {error:.2}m from where it was"
        );
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
                lines: vec![],
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
    fn disagreeing_frames_cannot_set_the_lens() {
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
        ]);
        let pose = look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(7.5, 6.0, 0.5));
        let assumed = Intrinsics::from_horizontal_fov(1920, 1080, 75.0);
        let views: Vec<View> = [50.0, 65.0]
            .into_iter()
            .map(|fov| View {
                corners: corners_of(
                    &layout,
                    &[3, 4],
                    0.1651,
                    pose,
                    &Intrinsics::from_horizontal_fov(1920, 1080, fov),
                ),
                pose,
                tags: vec![3, 4],
                lines: vec![],
            })
            .collect();
        for view in &views {
            assert!(
                focal_ratio(view, &assumed).is_some(),
                "each frame must independently measure"
            );
        }
        let calibration = calibrate(&views, assumed);
        assert!(
            !calibration.measured,
            "conflicting measurements must not set the lens"
        );
        assert_eq!(calibration.intrinsics, assumed);
        assert_eq!(agreeing_focal_ratio(&mut [0.8, 1.4, 1.42]), Some((1.42, 2)));
        assert_eq!(agreeing_focal_ratio(&mut [0.8, 0.81, 1.4, 1.42]), None);
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
            lines: vec![],
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
                lines: vec![],
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
            lines: vec![],
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
                lines: vec![],
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
            lines: vec![],
        });

        let station = fit(&views, &camera, &layout, 8.0).expect("a station");
        assert_eq!(station.frames_used, 4, "the stray frame should be dropped");
        assert_eq!(station.frames_rejected, 1);
        let error = (station.pose.position - truth.position).norm();
        assert!(error < 0.1, "station pose {error:.2}m from the truth");
        assert!(station.rms_px < 1.0, "station rms {:.2}px", station.rms_px);
    }

    #[test]
    fn a_station_rejects_a_second_camera_before_fitting() {
        // Competing cameras must not pull the initial least-squares fit between
        // them: doing that inflates the median residual and retains both groups.
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
        ]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let first = look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(7.5, 6.0, 0.5));
        let second = look_at(Vec3::new(10.0, -3.0, 6.0), Vec3::new(7.5, 6.0, 0.5));
        let views: Vec<View> = [first, first, first, second, second]
            .into_iter()
            .map(|pose| View {
                corners: corners_of(&layout, &[3, 4], 0.1651, pose, &camera),
                pose,
                tags: vec![3, 4],
                lines: vec![],
            })
            .collect();
        let station = fit(&views, &camera, &layout, 8.0).expect("majority camera");
        assert_eq!(station.frames_used, 3);
        assert_eq!(station.frames_rejected, 2);
        assert!((station.pose.position - first.position).norm() < 0.05);
        assert!(station.rms_px < 1.0);
    }

    #[test]
    fn a_single_tag_fit_does_not_hide_far_away_solutions() {
        let layout = test_layout(&[facing_tag(3, Vec3::new(6.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let first = look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(6.0, 6.0, 0.5));
        let distant = look_at(Vec3::new(10.0, 1.0, 4.0), Vec3::new(6.0, 6.0, 0.5));
        let corners = corners_of(&layout, &[3], 0.1651, first, &camera);
        let views: Vec<View> = [first, first, first, distant, distant]
            .into_iter()
            .map(|pose| View {
                corners: corners.clone(),
                pose,
                tags: vec![3],
                lines: vec![],
            })
            .collect();
        let station = fit(&views, &camera, &layout, 8.0).expect("majority camera");
        assert_eq!(station.frames_used, 3);
        assert_eq!(station.frames_rejected, 2);
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
            lines: vec![],
        }];
        let station = fit(&views, &camera, &layout, 8.0).expect("a station");
        assert_eq!(station.frames_used, 1);
        assert_eq!(station.frames_rejected, 0);
        assert!(fit(&[], &camera, &layout, 8.0).is_none());
    }
}

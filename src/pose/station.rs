//! Treating the frames of one file as one camera that did not move.
//!
//! Three things are only knowable across frames, and all are worth more than any
//! single frame's answer:
//!
//! * **The focal length.** `--fov` is a guess, and the whole pose is scaled by
//!   it: a five percent error in the assumed field of view moves the camera by
//!   half a metre and the corners still fit perfectly. [`measure_lens`] fits one
//!   lens to the frames of a shot at once — a shared centre, a rotation per
//!   frame, one focal length, and a radial distortion term only if the picture
//!   asks for one — and reports the band of lenses that explains them just as
//!   well. A band is not a measurement, and the report says which it has.
//! * **Where one shot ends and the next begins.** A pane is one camera, but it
//!   can be re-aimed between samples; [`shots`] splits the file where a tag both
//!   samples see has moved across the frame, so one station is not fitted to two
//!   framings.
//! * **The camera's position.** A broadcast camera is bolted to a stand and
//!   does not wander between sampled frames; the spread of the per-frame poses
//!   is corner noise, not motion. [`fit`] fits one pose to every frame's corners
//!   at once, which averages that noise away, and rejects the frames that do not
//!   belong to it — a pan, or a stray solve.
//!
//! All are reported next to the per-frame poses rather than replacing them: a
//! camera that really did pan should still be visible as having panned.

use std::ops::Range;

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
/// How much better the fitted lens has to explain the frames before it is used
/// in place of the `--fov` guess when the footage cannot pin it exactly. A
/// number that only wins by noise would move every distance in the video.
const LENS_GAIN: f64 = 0.75;
/// …and the least improvement in pixels that counts at all, so a fixture solved
/// to a thousandth of a pixel does not "find" a lens it cannot see.
const LENS_MIN_DROP_PX: f64 = 0.2;
/// A lens is *measured* only if the band of lenses the footage allows is worth
/// less than this many metres of camera position, which is the band's width
/// times the distance to the tags. A ratio alone cannot say that: 5% of
/// 1841px is 10cm of a camera a metre away and 1.5m of one thirty metres away,
/// and it is the metres a reader acts on.
const LENS_REACH_M: f64 = 0.5;
/// Lenses tried across the whole range before the search narrows in. A grid
/// rather than a pure bisection because the residual as a function of focal
/// length is not convex: a lone tag's corners fit at every focal length, so the
/// surface has flat valleys, and a search that starts in one reports its floor.
const LENS_GRID: usize = 11;
const LENS_STEPS: usize = 18;
/// How many bisections turn "still explains the frames this far out" into the
/// censored bound a report can quote: 2.5% of the range per step.
const LENS_BISECTIONS: usize = 6;
/// Radial distortion the picture is allowed to ask for, dimensionless.
const DISTORTION_RANGE: (f64, f64) = (-0.25, 0.25);
/// Pixels out of square the picture is allowed to ask for, as a ratio.
const ASPECT_RANGE: (f64, f64) = (0.95, 1.05);
/// A distortion or pixel-aspect term has to explain the frames this much better
/// before it is kept. Most footage has neither, and fitting them anyway hands
/// the solver two more ways to absorb a noisy corner.
const EXTRA_PARAM_GAIN: f64 = 0.9;
/// Frames a lens fit needs before it can measure anything: one frame has
/// nothing to agree with, and on real footage a single pane frame holding two
/// tags 36cm apart once halved the focal length of the whole video.
const MIN_LENS_FRAMES: usize = 2;
/// A shot ends when a tag that both samples see has moved this far, as a
/// fraction of the frame's width: the camera was re-aimed, not merely noisy.
const SHOT_SHIFT_FRACTION: f64 = 0.10;
/// …or when the samples are this far apart and share no tag to compare.
const SHOT_GAP_S: f64 = 20.0;
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
    /// Seconds into the video this frame was taken. The lens fit is one per
    /// station, but a pane holds more than one shot when the camera is re-aimed,
    /// and telling those apart needs the clock as well as the picture.
    pub time: f64,
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
    /// Whether the footage pins the focal length closely enough to quote it as
    /// a measurement. A best fit over a wide band is not a measurement, however
    /// much better it explains the frames than the guess did.
    pub measured: bool,
    /// Whether a lens fit was made at all, and so whether `range` says anything.
    /// A measurement is a fit that is both made and pinned.
    pub fitted: bool,
    /// Frames the lens fit was based on.
    pub frames: usize,
    /// The fitted focal length as a multiple of the assumed one.
    pub ratio: f64,
    /// The focal lengths the footage could not rule out, pixels, censored at the
    /// search range. `None` when no lens fit was possible at all. This is the
    /// number that stops a scale-limited pane from being read as a position: on
    /// the Houston final's overview pane the frames are explained equally well
    /// 550px and 1200px along, and two metres of camera position apart.
    pub range: Option<(f64, f64)>,
    /// Whether radial distortion was fitted rather than left at zero.
    pub k1_measured: bool,
    /// Whether the pixels were found to be out of square.
    pub aspect_measured: bool,
    /// What the focal band is worth in camera position at the distance the tags
    /// are seen from, metres: the band's width times that distance. This is the
    /// number that decides whether the lens counts as measured.
    pub reach_m: Option<f64>,
}

impl Calibration {
    fn assumed(intrinsics: Intrinsics) -> Self {
        Calibration {
            intrinsics,
            measured: false,
            fitted: false,
            frames: 0,
            ratio: 1.0,
            range: None,
            k1_measured: false,
            aspect_measured: false,
            reach_m: None,
        }
    }

    /// One line for the report: the focal length and how it was arrived at.
    pub fn describe(&self) -> String {
        let mut said = if self.measured {
            format!(
                "focal length measured from {} frame(s) of one station: {:.0}px ({:.0}°), {} the --fov guess",
                self.frames,
                self.intrinsics.fx,
                horizontal_fov_deg(&self.intrinsics),
                match (self.ratio - 1.0).abs() {
                    off if off < 0.005 => "within 1% of".to_string(),
                    off if self.ratio > 1.0 => format!("{:.0}% above", off * 100.0),
                    off => format!("{:.0}% below", off * 100.0),
                }
            )
        } else if self.fitted && (self.ratio - 1.0).abs() > 0.02 {
            // The fit found a lens that explains the frames by a real margin,
            // but the footage cannot say which lens: report where it could be
            // rather than pretending the best point is it. The metres are the
            // point — at twenty metres a 10% band is two metres of camera.
            let (low, high) = self
                .range
                .unwrap_or((self.intrinsics.fx, self.intrinsics.fx));
            format!(
                "focal length fitted from {} frame(s) but not pinned: {:.0}px ({:.0}°); anything from {:.0} to {:.0}px explains these frames as well, which is up to {:.1}m of camera position",
                self.frames,
                self.intrinsics.fx,
                horizontal_fov_deg(&self.intrinsics),
                low,
                high,
                self.reach_m.unwrap_or(0.0)
            )
        } else if let Some((low, high)) = self.range.filter(|_| self.fitted) {
            format!(
                "focal length assumed from --fov: {:.0}px ({:.0}°); the frames would fit anywhere from {:.0} to {:.0}px, which is up to {:.1}m of camera position",
                self.intrinsics.fx,
                horizontal_fov_deg(&self.intrinsics),
                low,
                high,
                self.reach_m.unwrap_or(0.0)
            )
        } else {
            format!(
                "focal length assumed from --fov: {:.0}px ({:.0}°)",
                self.intrinsics.fx,
                horizontal_fov_deg(&self.intrinsics)
            )
        };
        if self.k1_measured {
            said.push_str(&format!(
                ", radial distortion k1 = {:.3}",
                self.intrinsics.k1
            ));
        }
        if self.aspect_measured {
            let taller = self.intrinsics.fy / self.intrinsics.fx;
            said.push_str(&format!(
                ", pixels {:.1}% {} than square",
                (taller - 1.0).abs() * 100.0,
                if taller > 1.0 { "taller" } else { "wider" }
            ));
        }
        said
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

/// Measure the lens from the frames of one station.
///
/// A single frame cannot: a lone tag's four corners fix only the ratio of size
/// to distance, so every focal length has a pose that fits them. The fit here is
/// therefore joint — one centre, one lens, a rotation for each frame — and it is
/// run across a band of candidate lenses, a grid first and then a golden section
/// inside the best bracket, because the residual is not convex in focal length:
/// a lone tag leaves whole flat valleys, and a search that starts in one
/// descends to its floor and calls it the lens.
///
/// What comes back is not only a number. The band of lenses that still explains
/// the frames is the honest width of the answer. On the Houston final's overview
/// pane the tags stand in clusters 35cm apart, which at twenty metres is no
/// baseline at all, and the frames are explained equally well by 550px and
/// 1200px — two metres of camera position apart. A single number there would be
/// a lie with a decimal point on it, so the band is reported and the position is
/// read as the scaled thing it is.
///
/// The guess is kept unless the fit either pins the lens or explains the frames
/// by a real margin better, because a lens chosen by noise moves every distance
/// in the video.
pub(crate) fn measure_lens(views: &[View], assumed: Intrinsics, max_rms_px: f64) -> Calibration {
    let usable: Vec<View> = views
        .iter()
        .filter(|view| view.corners.len() >= 4)
        .cloned()
        .collect();
    if usable.len() < MIN_LENS_FRAMES {
        return Calibration::assumed(assumed);
    }
    let cost = |camera: &Intrinsics| joint_rms(&usable, camera);

    let Some(at_assumed) = cost(&assumed) else {
        return Calibration::assumed(assumed);
    };

    let (low, high) = (FOCAL_RANGE.0.ln(), FOCAL_RANGE.1.ln());
    let step = (high - low) / (LENS_GRID - 1) as f64;
    let mut grid = Vec::with_capacity(LENS_GRID);
    for index in 0..LENS_GRID {
        let ln = low + step * index as f64;
        let value = cost(&with_focal(&assumed, ln.exp())).unwrap_or(f64::INFINITY);
        if super::debug_enabled() {
            eprintln!(
                "  [lens] {:.0}px ({:.0}°): {:.3}px rms over {} frame(s)",
                ln.exp() * assumed.fx,
                horizontal_fov_deg(&with_focal(&assumed, ln.exp())),
                value,
                usable.len()
            );
        }
        grid.push((ln, value));
    }
    let Some(&(start, _)) = grid
        .iter()
        .filter(|(_, value)| value.is_finite())
        .min_by(|a, b| a.1.total_cmp(&b.1))
    else {
        return Calibration::assumed(assumed);
    };
    // A minimum sitting on the edge of the range is the search running out of
    // room, not the picture saying what the lens is.
    if start - low < step * 0.5 || high - start < step * 0.5 {
        return Calibration::assumed(assumed);
    }
    let (ln_ratio, mut best) = golden((start - step).max(low), (start + step).min(high), |ln| {
        cost(&with_focal(&assumed, ln.exp())).unwrap_or(f64::INFINITY)
    });

    // Distortion and pixel aspect, each only if it buys something real. A pane
    // whose tags never move in the picture cannot tell either apart from a
    // slightly different pose, and would hand the solver two free parameters to
    // absorb corner noise with.
    let mut camera = with_focal(&assumed, ln_ratio.exp());
    let mut k1_measured = false;
    let mut aspect_measured = false;
    let (k1, k1_cost) = golden(DISTORTION_RANGE.0, DISTORTION_RANGE.1, |k1| {
        cost(&Intrinsics { k1, ..camera }).unwrap_or(f64::INFINITY)
    });
    if super::debug_enabled() {
        eprintln!(
            "  [lens] radial k1 {:.3}: {:.3}px rms, against {:.3}px at k1 = 0",
            k1, k1_cost, best
        );
    }
    // …and neither may sit on the edge of what it is allowed to be: a bend the
    // search ran out of room for is the bound, not the picture, and taking it
    // hands every frame a distortion chosen by the clamp.
    let interior = |value: f64, (low, high): (f64, f64)| {
        let margin = (high - low) * 0.05;
        value > low + margin && value < high - margin
    };
    if worth(best, k1_cost) && interior(k1, DISTORTION_RANGE) {
        camera.k1 = k1;
        best = k1_cost;
        k1_measured = true;
    }
    let focal = camera.fx;
    let (aspect, aspect_cost) = golden(ASPECT_RANGE.0, ASPECT_RANGE.1, |aspect| {
        cost(&Intrinsics {
            fy: focal * aspect,
            ..camera
        })
        .unwrap_or(f64::INFINITY)
    });
    if worth(best, aspect_cost) && interior(aspect, ASPECT_RANGE) {
        camera.fy = focal * aspect;
        aspect_measured = true;
    }
    // One more look at the focal length now the extra terms have settled, so
    // the three are not left in the order they happened to be searched in. The
    // step here is relative to the camera as it now stands, so it is centred on
    // zero rather than on the first search's answer.
    let low_now = (FOCAL_RANGE.0 * assumed.fx / camera.fx).ln();
    let high_now = (FOCAL_RANGE.1 * assumed.fx / camera.fx).ln();
    let (ln_shift, best) = golden(
        (-step * 0.5).max(low_now),
        (step * 0.5).min(high_now),
        |ln| cost(&with_focal(&camera, ln.exp())).unwrap_or(f64::INFINITY),
    );
    // A lens fit that does not even reproject the frames within the run's own
    // error limit has not measured a lens: the frames do not agree on a camera,
    // and their disagreement is not evidence about the glass.
    if !best.is_finite() || best > max_rms_px {
        return Calibration::assumed(assumed);
    }
    camera = with_focal(&camera, ln_shift.exp());

    // How far the lens can move before the frames stop explaining themselves: a
    // bound measured the same way as `sigma_m`, since it is the same question
    // asked about a different unknown.
    let tolerance = (best * best + PTZ_SLACK_PX2).sqrt();
    let floor = (FOCAL_RANGE.0 * assumed.fx / camera.fx).ln();
    let ceiling = (FOCAL_RANGE.1 * assumed.fx / camera.fx).ln();
    let lo = focal_limit(&cost, &camera, floor, tolerance).exp();
    let hi = focal_limit(&cost, &camera, ceiling, tolerance).exp();
    let (lo_px, hi_px) = (camera.fx * lo, camera.fx * hi);

    // What the band is worth where the tags actually are: a narrow band on a
    // far camera is a wide band in metres, and metres are what the report is
    // read for.
    let distance = joint_fit(&usable, &camera).map_or(0.0, |(_, distance)| distance);
    let reach_m = (hi / lo - 1.0) * distance;
    let measured = reach_m <= LENS_REACH_M;
    let improvement = at_assumed - best;
    let used = measured || (improvement > LENS_MIN_DROP_PX && best <= at_assumed * LENS_GAIN);
    let intrinsics = if used { camera } else { assumed };
    Calibration {
        intrinsics,
        measured,
        fitted: true,
        frames: usable.len(),
        ratio: intrinsics.fx / assumed.fx,
        range: Some((lo_px, hi_px)),
        k1_measured,
        aspect_measured,
        reach_m: Some(reach_m),
    }
}

/// Whether a term the frames did not have to pay for is worth keeping: it has to
/// explain them by a clear margin, both in proportion and in pixels, or a noisy
/// corner is enough to conjure it.
fn worth(best: f64, candidate: f64) -> bool {
    candidate.is_finite()
        && candidate <= best * EXTRA_PARAM_GAIN
        && best - candidate > LENS_MIN_DROP_PX
}

/// Golden-section minimum of `cost` on `[a, b]`, in whatever space the caller
/// passed — the lens is searched in the log of the focal length, so a factor of
/// two is the same distance wherever it sits.
fn golden(mut a: f64, mut b: f64, mut cost: impl FnMut(f64) -> f64) -> (f64, f64) {
    const PHI: f64 = 0.618_033_988_749_895;
    let mut c = b - PHI * (b - a);
    let mut d = a + PHI * (b - a);
    let (mut cost_c, mut cost_d) = (cost(c), cost(d));
    for _ in 0..LENS_STEPS {
        if !cost_c.is_finite() && !cost_d.is_finite() {
            break;
        }
        if cost_c <= cost_d {
            b = d;
            d = c;
            cost_d = cost_c;
            c = b - PHI * (b - a);
            cost_c = cost(c);
        } else {
            a = c;
            c = d;
            cost_c = cost_d;
            d = a + PHI * (b - a);
            cost_d = cost(d);
        }
    }
    let middle = 0.5 * (a + b);
    let value = cost(middle);
    (
        middle,
        if value.is_finite() {
            value
        } else {
            f64::INFINITY
        },
    )
}

/// How far the lens can be scaled from the fitted one before the frames stop
/// explaining themselves, by bisection towards `edge_ln`. Returns `edge_ln`
/// when even the edge is not ruled out — the answer is then "more than this",
/// and the caller quotes the edge as the censored bound.
fn focal_limit(
    cost: &impl Fn(&Intrinsics) -> Option<f64>,
    base: &Intrinsics,
    edge_ln: f64,
    tolerance: f64,
) -> f64 {
    let within =
        |ln: f64| cost(&with_focal(base, ln.exp())).is_some_and(|value| value <= tolerance);
    if within(edge_ln) {
        return edge_ln;
    }
    let (mut good, mut bad) = (0.0, edge_ln);
    for _ in 0..LENS_BISECTIONS {
        let middle = 0.5 * (good + bad);
        if within(middle) {
            good = middle;
        } else {
            bad = middle;
        }
    }
    good
}

/// The reprojection error of one station fitted with `camera` as its lens.
///
/// Each frame is solved on its own at the candidate lens first, and the shared
/// centre starts from where those solutions agree. That matters more than it
/// sounds: starting every candidate from the pose the *assumed* lens found would
/// compare lenses by how close they sit to the guess rather than by what the
/// picture says, and a fit that only reached a local minimum would then look
/// like a measurement of the lens.
///
/// The field-model lines are deliberately left out: they arrive after this, and
/// they are evidence about the mount rather than about the lens.
fn joint_rms(views: &[View], camera: &Intrinsics) -> Option<f64> {
    joint_fit(views, camera).map(|(rms, _)| rms)
}

/// The same fit, together with how far the tags are from it: the reprojection
/// error says how well it explains the frames, the distance says what a
/// focal-length error is worth out on the field.
fn joint_fit(views: &[View], camera: &Intrinsics) -> Option<(f64, f64)> {
    let solved: Vec<CameraPose> = views
        .iter()
        .map(|view| solve::refine(view.pose, &view.corners, camera).map(|solution| solution.pose))
        .collect::<Option<Vec<_>>>()?;
    let positions: Vec<super::types::Vec3> = solved.iter().map(|pose| pose.position).collect();
    let (centre, _) = consensus(&positions);
    // A frame that solved to the other side of the wall has an aim fitted from
    // a place the camera was never at; from the real centre it points away, and
    // the residual behind the camera is flat, with no slope to descend. The
    // nearest frame that did agree has an aim worth starting from.
    let stray: Vec<bool> = solved
        .iter()
        .map(|pose| (pose.position - centre).norm() > CONSENSUS_RADIUS_M)
        .collect();
    let ptz: Vec<solve::PtzView> = views
        .iter()
        .zip(&solved)
        .enumerate()
        .map(|(index, (view, pose))| solve::PtzView {
            corners: view.corners.clone(),
            lines: Vec::new(),
            pose: CameraPose {
                position: centre,
                rotation: if stray[index] {
                    nearest_agreeing_aim(&solved, &stray, pose.position).unwrap_or(pose.rotation)
                } else {
                    pose.rotation
                },
            },
            camera: *camera,
        })
        .collect();
    let fitted = solve::refine_ptz_pinned(&ptz, false)?;
    let mut total = super::types::Vec3::new(0.0, 0.0, 0.0);
    let mut count = 0usize;
    for view in views {
        for corner in &view.corners {
            total = total + corner.field;
            count += 1;
        }
    }
    let middle = total * (1.0 / count.max(1) as f64);
    Some((ptz_rms(&fitted), (centre - middle).norm()))
}

/// Split a pane into the shots it actually holds.
///
/// A file is one pane of the broadcast frame, and on a fixed camera the whole
/// file is one shot. But a pane can be re-aimed, and a long gap between samples
/// can hide a cut; fitting one station to two shots puts the camera between them,
/// which is what a pan mixed into a static shot looked like before this existed.
/// The per-frame orientations stay in the report either way — a camera that
/// really panned has to keep showing as having panned — but the shared position
/// belongs to the shot it was fitted from.
///
/// The test is the picture, not the per-frame pose: a tag that both samples see
/// has moved a tenth of the frame across means the camera was re-aimed, while a
/// lone tag's own solve can be metres out and on either side of the wall without
/// the camera having moved at all.
pub(crate) fn shots(views: &[View], camera: &Intrinsics) -> Vec<Range<usize>> {
    let mut shots = Vec::new();
    if views.is_empty() {
        return shots;
    }
    let mut start = 0;
    for index in 1..views.len() {
        if !same_shot(&views[index - 1], &views[index], camera) {
            shots.push(start..index);
            start = index;
        }
    }
    shots.push(start..views.len());
    shots
}

fn same_shot(a: &View, b: &View, camera: &Intrinsics) -> bool {
    match shared_shift(a, b) {
        Some(shift) => shift <= camera.width as f64 * SHOT_SHIFT_FRACTION,
        // Nothing in the picture to compare. A short gap is still the same shot;
        // a long one, with the camera free to have been moved in between, is not.
        None => b.time - a.time <= SHOT_GAP_S,
    }
}

/// How far the farthest corner of a tag both frames saw has moved, in pixels.
fn shared_shift(a: &View, b: &View) -> Option<f64> {
    let mut shift: Option<f64> = None;
    for corner in &a.corners {
        for other in &b.corners {
            if (corner.field - other.field).norm() > 1e-9 {
                continue;
            }
            let moved = (corner.image.0 - other.image.0).hypot(corner.image.1 - other.image.1);
            shift = Some(shift.map_or(moved, |biggest: f64| biggest.max(moved)));
        }
    }
    shift
}

/// The shot the camera spent most of the file in, and the one the reported
/// station is fitted from.
///
/// Counted by the frames whose own solve agrees with the shot's centre, not by
/// the number of samples: a shot is evidence only where the frames in it are
/// evidence about the same place.
pub(crate) fn busiest_shot(views: &[View], shots: &[Range<usize>]) -> Range<usize> {
    shots
        .iter()
        .max_by_key(|shot| {
            (
                consensus_position(&views[shot.start..shot.end]).1,
                shot.len(),
            )
        })
        .cloned()
        .unwrap_or(0..0)
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
    let poses: Vec<CameraPose> = views.iter().map(|view| view.pose).collect();
    nearest_agreeing_aim(&poses, stray, from)
}

/// The same, over already-solved poses: the lens fit works with those rather
/// than with the frames' own solves.
fn nearest_agreeing_aim(
    poses: &[CameraPose],
    stray: &[bool],
    from: super::types::Vec3,
) -> Option<super::types::Quat> {
    poses
        .iter()
        .zip(stray)
        .filter(|(_, stray)| !**stray)
        .min_by(|(a, _), (b, _)| {
            (a.position - from)
                .norm()
                .total_cmp(&(b.position - from).norm())
        })
        .map(|(pose, _)| pose.rotation)
}

fn consensus_position(views: &[View]) -> (super::types::Vec3, usize) {
    let positions: Vec<super::types::Vec3> = views.iter().map(|view| view.pose.position).collect();
    consensus(&positions)
}

/// The mean of the largest group of positions within [`CONSENSUS_RADIUS_M`] of
/// each other, and how many positions that is.
fn consensus(positions: &[super::types::Vec3]) -> (super::types::Vec3, usize) {
    let mut best = (positions[0], 0usize);
    for position in positions {
        let agreeing: Vec<super::types::Vec3> = positions
            .iter()
            .copied()
            .filter(|other| (*other - *position).norm() <= CONSENSUS_RADIUS_M)
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
                    time: 0.0,
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
                    time: 0.0,
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
                time: 0.0,
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
            time: 0.0,
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
                time: 0.0,
                corners: corners.clone(),
                pose,
                tags: vec![5],
                lines: Vec::new(),
            });
            with_model.push(View {
                time: 0.0,
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
                time: 0.0,
                corners: corners.clone(),
                pose,
                tags: vec![3, 4],
                lines: vec![],
            })
            .collect();
        let calibration = measure_lens(&views, assumed, 8.0);
        assert!(calibration.fitted, "two tags should pin the lens to a band");
        assert_eq!(calibration.frames, 3);
        let (low, high) = calibration.range.expect("a band");
        // The band has to contain the lens the picture was drawn with, and be
        // narrow enough to be worth acting on. It is not a point: two tags on
        // one wall, seen from one place, leave a scale the principal point
        // barely separates, which is why this is a band and not a number.
        assert!(
            low <= truth.fx && truth.fx <= high,
            "the true lens {:.0}px is outside the reported {low:.0}–{high:.0}px",
            truth.fx
        );
        assert!(
            (calibration.intrinsics.fx - truth.fx).abs() / truth.fx < 0.10,
            "fitted {:.0}px against {:.0}px",
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
                time: 0.0,
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
        let calibration = measure_lens(&views, assumed, 8.0);
        // Neither frame can be the lens's evidence when they want different
        // ones: the joint fit cannot explain both, so there is no lens to
        // report rather than a compromise the reader would take as measured.
        assert!(
            !calibration.fitted,
            "two frames that want different lenses still produced a lens"
        );
        assert!(!calibration.measured);
        assert_eq!(calibration.intrinsics, assumed);
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
            time: 0.0,
            corners: corners_of(&layout, &[3, 4], 0.1651, pose, &truth),
            pose,
            tags: vec![3, 4],
            lines: vec![],
        }];

        let calibration = measure_lens(&views, assumed, 8.0);
        assert!(!calibration.fitted, "one frame must not calibrate the lens");
        assert!(!calibration.measured);
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
                time: 0.0,
                corners: corners.clone(),
                pose,
                tags: vec![3, 4],
                lines: vec![],
            })
            .collect();

        let calibration = measure_lens(&views, assumed, 8.0);
        assert!(
            !calibration.fitted,
            "measured {:.0}px, which is the edge of the range",
            calibration.intrinsics.fx
        );
        assert_eq!(calibration.intrinsics, assumed);
    }

    #[test]
    fn one_tag_cannot_measure_a_lens() {
        // A fixed camera watching one tag: every sample holds the same four
        // corners, and those four corners fit at every focal length, each with
        // its own distance. Nothing in the picture says which is right, so the
        // assumed value is kept and the band is reported as wide. (Frames of a
        // *panning* camera do constrain the lens — the perspective across the
        // tag changes while the centre does not — which is why the fit is joint
        // rather than one frame's business.)
        let layout = test_layout(&[facing_tag(13, Vec3::new(15.5, 6.0, 0.5))]);
        let truth = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let position = Vec3::new(11.0, 2.0, 4.0);
        let pose = look_at(position, Vec3::new(15.5, 6.0, 0.5));
        let corners = corners_of(&layout, &[13], 0.1651, pose, &truth);
        let views: Vec<View> = (0..3)
            .map(|index| View {
                time: index as f64 * 7.0,
                corners: corners.clone(),
                pose,
                tags: vec![13],
                lines: vec![],
            })
            .collect();
        let assumed = Intrinsics::from_horizontal_fov(1280, 720, 45.0);
        let calibration = measure_lens(&views, assumed, 8.0);
        assert!(!calibration.measured, "one tag must not calibrate the lens");
        assert_eq!(calibration.intrinsics, assumed);
        assert_eq!(calibration.frames, 3, "the fit used every frame it could");
        let (low, high) = calibration
            .range
            .expect("the fit still says where it could be");
        assert!(
            high / low > 1.5,
            "one tag narrowed the lens to {low:.0}–{high:.0}px, which it cannot"
        );
    }

    #[test]
    fn a_cluster_of_tags_does_not_pin_the_lens_at_broadcast_distance() {
        // The Houston final's panes in miniature: two hub tags 35cm apart, both
        // at the same height, seen from nineteen metres. Moving the camera back
        // and lengthening the lens reproduces both almost exactly, so the band
        // of lenses that explains the frames is wide. That is the fact the
        // report exists to state — and the reason a position from those panes
        // has to be read as a direction whose scale is uncertain.
        let layout = TagLayout::load("2026-rebuilt-welded").expect("the bundled layout");
        let first = layout.tag(8).expect("tag 8 is in the layout");
        let second = layout.tag(9).expect("tag 9 is in the layout");
        let normal = Mat3::from_quaternion(first.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        let position = first.translation + normal * 19.0 + Vec3::new(0.0, 0.0, 6.0);
        let middle = (first.translation + second.translation) * 0.5;
        let truth = Intrinsics::from_horizontal_fov(1280, 720, 70.0);
        let mut seed = 7u32;
        let views: Vec<View> = [-0.6, 0.0, 0.6]
            .into_iter()
            .enumerate()
            .map(|(index, offset)| {
                let pose = look_at(position, Vec3::new(middle.x, middle.y + offset, middle.z));
                let mut corners = corners_of(&layout, &[8, 9], 0.1651, pose, &truth);
                jitter(&mut corners, &mut seed);
                View {
                    time: index as f64 * 7.0,
                    corners,
                    pose,
                    tags: vec![8, 9],
                    lines: vec![],
                }
            })
            .collect();
        let assumed = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let calibration = measure_lens(&views, assumed, 8.0);
        // Whatever the band is, it has to contain the lens the picture was
        // drawn with: a report that excludes the truth is worse than one that
        // admits it does not know.
        if let Some((low, high)) = calibration.range {
            assert!(
                low <= truth.fx && truth.fx <= high,
                "the lens the picture was drawn with ({:.0}px) is outside the reported {low:.0}–{high:.0}px",
                truth.fx
            );
        }
        // And two tags this close together at this distance must not be quoted
        // as a measurement: the band they leave is several percent wide, which
        // on a twenty-metre ray is metres of camera position.
        assert!(
            !calibration.measured
                || calibration
                    .range
                    .is_some_and(|range| range.1 / range.0 < 1.02),
            "a 35cm cluster at nineteen metres is not a lens: reported {:.0}px, band {:.0}–{:.0}px",
            calibration.intrinsics.fx,
            calibration.range.map_or(0.0, |range| range.0),
            calibration.range.map_or(0.0, |range| range.1)
        );
    }

    #[test]
    fn a_bent_lens_is_fitted_only_when_the_picture_asks_for_one() {
        // A broadcast long lens bends straight lines near the edge of the
        // frame, and a tag sitting there lands where a pinhole says it cannot.
        // No pose absorbs that: what separates the bend from a slightly
        // different camera is the same tags seen at different distances from
        // the principal point, which is what a pan gives.
        // Tags across the width of the frame are what a lens bend shows up in:
        // moving the outermost ones and not the middle is a shape no pose can
        // reproduce, whereas a tight cluster is nearly a zoom of itself.
        let layout = TagLayout::load("2026-rebuilt-welded").expect("the bundled layout");
        let ids = [5u16, 8, 18, 27];
        let base = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let bent = Intrinsics { k1: 0.12, ..base };
        let position = Vec3::new(8.27, -9.0, 4.0);
        let shots = |camera: &Intrinsics| -> Vec<View> {
            [2.2, 3.4, 4.6]
                .into_iter()
                .enumerate()
                .map(|(index, target)| {
                    let pose = look_at(position, Vec3::new(8.27, target, 1.12));
                    View {
                        time: index as f64 * 7.0,
                        corners: corners_of(&layout, &ids, 0.1651, pose, camera),
                        pose,
                        tags: ids.to_vec(),
                        lines: vec![],
                    }
                })
                .collect()
        };

        let fitted = measure_lens(&shots(&bent), base, 8.0);
        assert!(
            fitted.k1_measured,
            "a 0.12 bend was left at k1 = {:.3}",
            fitted.intrinsics.k1
        );
        assert!(
            (fitted.intrinsics.k1 - 0.12).abs() < 0.06,
            "fitted k1 = {:.3} against 0.12",
            fitted.intrinsics.k1
        );

        // …and straight footage must not be given one, which is the half of the
        // test that keeps the term from swallowing corner noise.
        let plain = measure_lens(&shots(&base), base, 8.0);
        assert!(
            !plain.k1_measured,
            "straight footage was bent by k1 = {:.3}",
            plain.intrinsics.k1
        );
    }

    #[test]
    fn a_re_aimed_camera_is_two_shots_and_a_small_pan_is_one() {
        // A pane is one camera, but it can be re-aimed between samples. The
        // evidence is a tag both frames see: a few degrees of pan leaves it
        // where it was to within a hundred pixels, and is one shot; a re-aim
        // throws it across the frame, and averaging the two positions would put
        // the camera between them.
        let layout = test_layout(&[facing_tag(3, Vec3::new(6.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let position = Vec3::new(2.0, 1.0, 4.0);
        let at = |degrees: f64, time: f64| {
            let level = look_at(position, Vec3::new(6.0, 6.0, 0.5));
            // A yaw is a rotation of the *field* frame, so it multiplies the
            // field-to-camera rotation on the right; on the left it would roll
            // the camera about its own optical axis and barely move the tag.
            let turned = Mat3::from_quaternion(level.rotation)
                .mul_mat(&Mat3::from_axis_angle(
                    Vec3::new(0.0, 0.0, 1.0),
                    degrees.to_radians(),
                ))
                .to_quaternion();
            let pose = CameraPose {
                position,
                rotation: turned,
            };
            View {
                time,
                corners: corners_of(&layout, &[3], 0.1651, pose, &camera),
                pose,
                tags: vec![3],
                lines: vec![],
            }
        };
        let views = vec![at(-2.0, 0.0), at(2.0, 7.0), at(40.0, 14.0)];
        let split = shots(&views, &camera);
        assert_eq!(
            split,
            vec![0..2, 2..3],
            "four degrees of pan is one shot and a forty-degree re-aim is not"
        );
    }

    #[test]
    fn a_long_gap_with_no_tag_in_common_is_a_new_shot() {
        // Two samples thirty seconds apart that share no tag have nothing in
        // the picture to say whether the camera moved, so they are not treated
        // as one shot: a station is evidence about one place, and this is the
        // case where the evidence is missing rather than absent-mindedly
        // assumed. Ten seconds apart is a different matter.
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(9.0, 6.0, 0.5)),
        ]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let position = Vec3::new(2.0, 1.0, 4.0);
        let view = |id: u16, time: f64| {
            let pose = look_at(position, layout.tag(id).expect("tag").translation);
            View {
                time,
                corners: corners_of(&layout, &[id], 0.1651, pose, &camera),
                pose,
                tags: vec![id],
                lines: vec![],
            }
        };
        let close = vec![view(3, 0.0), view(4, 10.0)];
        assert_eq!(shots(&close, &camera), vec![0..2]);
        let apart = vec![view(3, 0.0), view(4, 30.0)];
        assert_eq!(shots(&apart, &camera), vec![0..1, 1..2]);
    }

    #[test]
    fn the_busiest_shot_is_the_one_the_station_is_fitted_from() {
        let layout = TagLayout::load("2026-rebuilt-welded").expect("the bundled layout");
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let position = Vec3::new(12.0, -8.0, 5.0);
        let view = |id: u16, time: f64, target: Vec3| {
            let pose = look_at(position, target);
            View {
                time,
                corners: corners_of(&layout, &[id], 0.1651, pose, &camera),
                pose,
                tags: vec![id],
                lines: vec![],
            }
        };
        let long = layout.tag(5).expect("tag").translation;
        let short = layout.tag(18).expect("tag").translation;
        // The two halves share no tag, so only the clock can tell them apart.
        let views = vec![
            view(5, 0.0, long),
            view(5, 7.0, long),
            view(5, 14.0, long),
            view(18, 44.0, short),
            view(18, 51.0, short),
        ];
        let split = shots(&views, &camera);
        assert_eq!(split.len(), 2, "{split:?}");
        assert_eq!(busiest_shot(&views, &split), 0..3);
    }

    /// Deterministic corner noise, of the size a broadcast codec leaves: about
    /// a third of a pixel, which on a nine-pixel tag is a large fraction of it.
    fn jitter(corners: &mut [Correspondence], seed: &mut u32) {
        let mut noise = || {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((*seed >> 8) as f64 / (1u32 << 24) as f64) - 0.5
        };
        for corner in corners.iter_mut() {
            corner.image.0 += noise() * 0.8;
            corner.image.1 += noise() * 0.8;
        }
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
                time: 0.0,
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
            time: 0.0,
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
                time: 0.0,
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
                time: 0.0,
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
            time: 0.0,
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

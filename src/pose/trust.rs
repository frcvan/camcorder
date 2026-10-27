//! How much a solve can be believed, and when it cannot be believed at all.
//!
//! Three separate questions, which is why they live together rather than in the
//! solver: which detections deserve to be looked at at all, which poses are
//! physically possible, and how much weight a solved frame carries once it is.
//!
//! * [`margin_floor`] and [`keep`] are about the decoder's enthusiasm. A flat
//!   floor throws away the far tags — a tag fifteen pixels across cannot reach
//!   the decision margin a tag filling a quarter of the frame does — so the
//!   floor follows the tag's size, and a marginal read of a small tag has to be
//!   backed by another frame having seen the same id.
//! * [`plausible`] is about the pose. A single square tag has two poses that
//!   reproject its corners identically, and the twin is routinely underneath
//!   the field looking at the roof. It fits the pixels perfectly; it just is
//!   not a camera that could have filmed them. Rejecting it is not a heuristic
//!   tune — the camera cannot be below the carpet, cannot be behind the face of
//!   a tag it read, and cannot be looking at the sky while filming the field.
//! * [`confidence`] is the honest answer for a frame that is possible but
//!   weakly determined, so a single-tag frame that is scaled by the assumed
//!   field of view does not read the same as one pinned by three tags.

use std::collections::HashMap;

use super::{
    layout::TagLayout,
    types::{CameraPose, Mat3, TagObservation, Vec3},
};

/// The tag width, in pixels, at which a detection is judged by the full floor.
/// Below it the floor falls away, down to [`SMALL_TAG_SHARE`] of that at
/// [`ZERO_WIDTH`], because a tag a dozen pixels across has almost no bits left
/// at full contrast.
const FULL_SIZE_PX: f64 = 48.0;
const ZERO_WIDTH_PX: f64 = 8.0;
const SMALL_TAG_SHARE: f32 = 0.35;
/// A marginal read of a small tag is kept only if the same id was read in at
/// least this share of the frames that detected anything, or if it is big
/// enough to deserve the full floor on its own. One frame alone reading a small
/// tag at low margin is the classic false positive; the same id turning up
/// again is the evidence that it is real.
const CORROBORATION: f64 = 0.2;

/// Nobody films a field from underneath its carpet.
const MIN_HEIGHT_M: f64 = 0.0;
/// Above this the pose is a scale failure rather than a camera. Broadcast
/// cameras, gantries and drones all stay under it.
const MAX_HEIGHT_M: f64 = 40.0;
/// How far outside the field's own rectangle a camera may stand. Broadcast
/// cameras sit in the stands a few metres off the carpet; a pose tens of metres
/// out is the one-tag mirror, or a focal length far from the truth.
const FIELD_MARGIN_M: f64 = 30.0;
/// A camera is mounted roughly level about its optical axis. More roll than
/// this turns up only when the solve has balanced itself on a tag edge.
const MAX_ROLL_DEG: f64 = 40.0;
/// A camera filming the field looks down at it. Level is the worst case for a
/// real installation; a pose looking up at this angle is filming the roof — and
/// is what the mirror solution does.
const MAX_PITCH_DEG: f64 = 15.0;
/// A tag is read from its face. A camera a hair inside the plane counts as in
/// front of it; one behind it could not have seen the pattern at all.
const FACE_MARGIN_M: f64 = 0.01;

/// The decision margin a tag this wide in pixels has to reach.
///
/// The floor falls off with size because the margin does: it is the contrast
/// between the best and second-best bit interpretation, and a tag eight pixels
/// across is a couple of samples per bit. Holding small tags to the full floor
/// is how the far tags disappear from a broadcast without any of them being
/// wrong.
pub fn margin_floor(pixel_width: f64, base: f32) -> f32 {
    if !pixel_width.is_finite() {
        return base;
    }
    let span = (pixel_width - ZERO_WIDTH_PX) / (FULL_SIZE_PX - ZERO_WIDTH_PX);
    let span = span.clamp(0.0, 1.0) as f32;
    let share = SMALL_TAG_SHARE + (1.0 - SMALL_TAG_SHARE) * span;
    base * share
}

/// Which share of the frames that detected anything read each tag id.
///
/// A broadcast shows one camera for the whole segment, so a tag that stands in
/// view is seen in every sampled frame where it is not occluded. An id the
/// decoder produced once, and never again, is either moving through the picture
/// or a mis-read; the count here does not decide on its own, but it is what
/// [`keep`] checks the marginal reads against.
pub fn id_support(frames: &[Vec<TagObservation>]) -> HashMap<u16, f64> {
    let mut seen: HashMap<u16, usize> = HashMap::new();
    let mut informative = 0usize;
    for detections in frames {
        if detections.is_empty() {
            continue;
        }
        informative += 1;
        let mut ids: Vec<u16> = detections.iter().map(|detection| detection.id).collect();
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            *seen.entry(id).or_insert(0) += 1;
        }
    }
    if informative == 0 {
        return HashMap::new();
    }
    seen.into_iter()
        .map(|(id, count)| (id, count as f64 / informative as f64))
        .collect()
}

/// Whether one detection earns a place in the frame.
///
/// Two ways in: it cleared the margin its own size deserves, or it is a small
/// tag read marginally *and* another frame read the same id. A small tag seen
/// in one frame at a low margin on its own is not evidence, it is a guess.
pub fn keep(detection: &TagObservation, support: &HashMap<u16, f64>, base: f32) -> bool {
    let width = detection.pixel_width();
    let floor = margin_floor(width, base);
    if detection.margin < floor {
        return false;
    }
    if detection.margin >= base {
        return true;
    }
    support.get(&detection.id).copied().unwrap_or(0.0) >= CORROBORATION
}

/// Why a pose cannot be a camera filming a field, or `Ok` if it can.
///
/// The order matters only for what the report says first: the height is the
/// bluntest failure and the one to lead with.
pub fn plausible(pose: &CameraPose, layout: &TagLayout, tags: &[u16]) -> Result<(), String> {
    let position = pose.position;
    if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() {
        return Err("the pose is not a finite position".to_string());
    }
    if position.z < MIN_HEIGHT_M {
        return Err(format!(
            "the camera is {:.1}m below the carpet",
            -position.z
        ));
    }
    if position.z > MAX_HEIGHT_M {
        return Err(format!(
            "the camera is {:.0}m above the field, which is a scale failure rather than a camera",
            position.z
        ));
    }
    if position.x < -FIELD_MARGIN_M
        || position.x > layout.field_length + FIELD_MARGIN_M
        || position.y < -FIELD_MARGIN_M
        || position.y > layout.field_width + FIELD_MARGIN_M
    {
        return Err(format!(
            "the camera is at ({:.0}, {:.0})m, far outside the {:.0}m x {:.0}m field",
            position.x, position.y, layout.field_length, layout.field_width
        ));
    }
    let roll = pose.roll_deg().abs();
    if !roll.is_finite() || roll > MAX_ROLL_DEG {
        return Err(format!(
            "the camera is rolled {roll:.0}°, which is a solve failure rather than a mount"
        ));
    }
    let pitch = pose.pitch_deg();
    if !pitch.is_finite() || pitch > MAX_PITCH_DEG {
        return Err(format!(
            "the camera is looking {pitch:.0}° above the horizon, away from the field"
        ));
    }
    for id in tags {
        let Some(tag) = layout.tag(*id) else {
            continue;
        };
        let normal = Mat3::from_quaternion(tag.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        let in_front = (position - tag.translation).dot(normal);
        if in_front < FACE_MARGIN_M {
            return Err(format!(
                "the camera is behind the face of tag {id}, which cannot be read from there"
            ));
        }
    }
    Ok(())
}

/// What a solved frame has going for it.
#[derive(Debug, Clone, Copy)]
pub struct Evidence {
    /// Tags the solve was based on.
    pub tags: usize,
    /// Root-mean-square reprojection error, pixels.
    pub rms_px: f64,
    /// The least confident detection the solve used, in the decoder's margin.
    pub weakest_margin: f32,
    /// The weakest [`id_support`] among the tags used, `0..1`.
    pub support: f64,
    /// Whether the focal length was measured from this footage rather than
    /// assumed from `--fov`.
    pub calibrated: bool,
}

/// How much of a result one frame is, `0..1`.
///
/// Four things blunt it, each for a reason that shows in the report:
///
/// * one tag alone: its pose is a homography through four corners, and the
///   scale of that homography is set by the focal length. Assumed, that scale
///   error moves the camera by metres while the corners still fit to a pixel.
///   Measured from this footage it is better, but the square's twin pose is
///   still unresolved, so it stays the weakest evidence there is.
/// * reprojection error, because a fit that needs to bend to explain the corners
///   is a fit that is only approximately right;
/// * the weakest margin, because a barely decoded tag's corners are where its
///   bits were hardest to read;
/// * how often the tags were seen, because a tag read in one frame of ten is
///   more likely to be a mis-read than one read in all ten.
pub fn confidence(evidence: &Evidence, max_rms_px: f64) -> f64 {
    let single_tag = if evidence.calibrated { 0.7 } else { 0.5 };
    let base = if evidence.tags <= 1 { single_tag } else { 1.0 };
    let fit = 1.0 - 0.5 * (evidence.rms_px / max_rms_px.max(1e-6)).clamp(0.0, 1.0);
    let contrast = (evidence.weakest_margin as f64 / 100.0).clamp(0.4, 1.0);
    let agreement = 0.5 + 0.5 * evidence.support.clamp(0.0, 1.0);
    (base * fit * contrast * agreement).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::types::Quat;

    fn observation(id: u16, width: f64, margin: f32) -> TagObservation {
        TagObservation {
            id,
            corners: [[0.0, 0.0], [0.0, width], [width, width], [width, 0.0]],
            margin,
            hamming: 0,
        }
    }

    #[test]
    fn a_far_tag_is_held_to_the_margin_its_size_deserves() {
        // The floor is what decides whether the far tags exist at all: at the
        // old flat 30, a tag twenty pixels across had to match the margin of
        // one filling the frame, and only ever did when it was clean and close
        // enough to be big anyway.
        assert!((margin_floor(48.0, 30.0) - 30.0).abs() < 1e-6);
        assert!((margin_floor(200.0, 30.0) - 30.0).abs() < 1e-6);
        let small = margin_floor(16.0, 30.0);
        assert!(
            small < 22.0 && small > 8.0,
            "a 16px tag should need a little over half the margin, got {small}"
        );
        assert!(margin_floor(4.0, 30.0) >= 30.0 * SMALL_TAG_SHARE);
        assert!(margin_floor(f64::NAN, 30.0) == 30.0);
    }

    #[test]
    fn a_small_marginal_tag_needs_another_frame_to_have_seen_it() {
        let base = 30.0;
        // Big and clean: in on its own.
        let big = observation(13, 120.0, 200.0);
        assert!(keep(&big, &HashMap::new(), base));

        // Small and marginal, seen once: out. It clears the floor its size
        // deserves, so the counting is what decides.
        let lone = observation(13, 16.0, 20.0);
        let mut support = HashMap::new();
        support.insert(13u16, 0.1);
        assert!(!keep(&lone, &support, base));

        // The same read, with the id turning up in other frames: in.
        support.insert(13, 0.8);
        assert!(keep(&lone, &support, base));

        // Small but clear: no corroboration needed, the margin speaks for it.
        let clear = observation(13, 16.0, 60.0);
        assert!(keep(&clear, &HashMap::new(), base));

        // Too faint even for its size: out whatever else saw it.
        let hopeless = observation(13, 16.0, 3.0);
        assert!(!keep(&hopeless, &support, base));
    }

    #[test]
    fn support_counts_the_frames_that_read_each_id() {
        let frames = vec![
            vec![observation(13, 40.0, 90.0), observation(14, 40.0, 90.0)],
            vec![observation(13, 40.0, 90.0)],
            vec![],
            vec![observation(13, 40.0, 90.0), observation(99, 10.0, 12.0)],
        ];
        let support = id_support(&frames);
        assert_eq!(support.get(&13), Some(&1.0));
        assert!((support.get(&14).copied().unwrap_or(0.0) - 1.0 / 3.0).abs() < 1e-9);
        assert!((support.get(&99).copied().unwrap_or(0.0) - 1.0 / 3.0).abs() < 1e-9);
        // The frame with nothing in it is not a vote that its tags are missing.
        assert!(id_support(&[]).is_empty());
    }

    fn layout() -> TagLayout {
        TagLayout::load("2026-rebuilt-welded").expect("bundled layout")
    }

    /// A pose with the given position, looking down the field with no roll.
    fn pose_at(position: Vec3) -> CameraPose {
        // forward = +x, right = -y, down = -z: rows are the camera axes.
        CameraPose {
            position,
            rotation: Mat3::from_rows([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
                .to_quaternion(),
        }
    }

    #[test]
    fn a_camera_under_the_carpet_is_not_a_camera() {
        // The one-tag mirror fits the corners to a fifth of a pixel and puts
        // the camera three metres underground. That is the pose this rejects.
        let layout = layout();
        let error = plausible(&pose_at(Vec3::new(8.0, 8.0, -3.0)), &layout, &[13])
            .expect_err("underground is impossible");
        assert!(error.contains("below the carpet"), "{error}");
    }

    #[test]
    fn a_camera_looking_at_the_roof_is_not_filming_the_field() {
        let layout = layout();
        // Looking up and along +x: forward = (cos, 0, sin) with a positive z.
        let (sin, cos) = 30f64.to_radians().sin_cos();
        let forward = Vec3::new(cos, 0.0, sin);
        let right = forward.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
        let down = forward.cross(right);
        let pose = CameraPose {
            position: Vec3::new(8.0, 8.0, 3.0),
            rotation: Mat3::from_rows([right.to_array(), down.to_array(), forward.to_array()])
                .to_quaternion(),
        };
        let error = plausible(&pose, &layout, &[13]).expect_err("looking up is not a field camera");
        assert!(error.contains("above the horizon"), "{error}");
    }

    #[test]
    fn a_camera_behind_the_tag_face_could_not_have_read_the_tag() {
        // Tag 13 stands on the red wall at x = 16.53 facing into the field
        // (its face normal points at negative x). A camera beyond the wall at
        // x = 20 is behind it.
        let layout = layout();
        let front = plausible(&pose_at(Vec3::new(12.0, 8.0, 5.0)), &layout, &[13]);
        assert!(front.is_ok(), "{front:?}");
        let behind = plausible(&pose_at(Vec3::new(20.0, 8.0, 5.0)), &layout, &[13])
            .expect_err("behind the face");
        assert!(behind.contains("behind the face"), "{behind}");
    }

    #[test]
    fn a_level_camera_in_the_stands_is_accepted() {
        let layout = layout();
        // y = 13m is outside the 8m-wide field: a broadcast camera position.
        assert!(plausible(&pose_at(Vec3::new(8.0, 13.0, 6.0)), &layout, &[13, 14]).is_ok());
        // …and a pose a hundred metres away is not.
        let far = plausible(&pose_at(Vec3::new(8.0, 45.0, 6.0)), &layout, &[13])
            .expect_err("far outside the field");
        assert!(far.contains("far outside"), "{far}");
    }

    #[test]
    fn one_tag_alone_is_the_weakest_evidence_there_is() {
        let strong = Evidence {
            tags: 3,
            rms_px: 0.2,
            weakest_margin: 180.0,
            support: 1.0,
            calibrated: true,
        };
        let weak = Evidence {
            tags: 1,
            rms_px: 0.2,
            weakest_margin: 180.0,
            support: 1.0,
            calibrated: false,
        };
        let strong = confidence(&strong, 8.0);
        let weak = confidence(&weak, 8.0);
        assert!(
            strong > 0.9,
            "three clean tags should be near certain: {strong}"
        );
        assert!(
            weak < strong * 0.6,
            "one tag is scaled by the assumed field of view: {weak} vs {strong}"
        );

        // A measured focal length helps, but does not make one tag strong.
        let measured = confidence(
            &Evidence {
                calibrated: true,
                ..Evidence {
                    tags: 1,
                    rms_px: 0.2,
                    weakest_margin: 180.0,
                    support: 1.0,
                    calibrated: false,
                }
            },
            8.0,
        );
        assert!(measured > weak && measured < 0.8, "measured {measured}");

        // Everything else being equal, a solve that has to bend costs
        // confidence, and so does a barely-read tag.
        let bent = confidence(
            &Evidence {
                tags: 3,
                rms_px: 7.0,
                weakest_margin: 40.0,
                support: 0.2,
                calibrated: true,
            },
            8.0,
        );
        assert!(bent < strong * 0.5, "bent {bent} vs {strong}");
        assert!(
            confidence(
                &Evidence {
                    tags: 0,
                    ..strong_evidence()
                },
                8.0
            ) <= 1.0
        );
    }

    fn strong_evidence() -> Evidence {
        Evidence {
            tags: 2,
            rms_px: 0.2,
            weakest_margin: 150.0,
            support: 1.0,
            calibrated: true,
        }
    }

    #[test]
    fn a_quaternion_round_trip_keeps_the_face_normal_where_it_was() {
        // The gate leans on the tag's own `+x`, so it is worth pinning that the
        // quaternion in the layout really is the face normal.
        let layout = layout();
        let tag = layout.tag(13).expect("tag 13");
        let normal = Mat3::from_quaternion(tag.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            normal.x < -0.9,
            "the red wall's tags should face into the field, got {normal:?}"
        );
        let _ = Quat::IDENTITY;
    }
}

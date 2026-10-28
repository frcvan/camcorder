//! Camera pose from AprilTag corners: planar initialisation, then a joint
//! refinement over every corner of every tag in the frame.
//!
//! One tag gives a planar homography, which is enough to start; two or more
//! tags constrain the pose properly, so the refinement uses all of them at once
//! and the result is the pose that best reprojects everything the camera saw.
//!
//! One tag is where this gets interesting. A single square has *two* poses that
//! reproject its corners to the same pixels, and with real corner noise the
//! wrong one often fits better than the right one. On the fixture that prompted
//! [`restart_seeds`], 13 frames in 40 solved to a camera three metres under the
//! carpet looking at the roof, at a reprojection error of 0.06px against the
//! true pose's 0.39px. Reprojection error cannot separate them, so the answer is
//! chosen the way a person would: by whether it is a camera that could have
//! filmed the picture at all ([`trust::plausible`]), by which end of the field
//! the colours around the tag say the camera is at, and only then by the fit.

use super::{
    layout::TagLayout,
    side::{end_of_x, End},
    trust,
    types::{CameraPose, Intrinsics, Mat3, Quat, TagObservation, Vec3},
};

/// A solved camera pose and how well it explains the detections.
#[derive(Debug, Clone)]
pub struct Solution {
    pub pose: CameraPose,
    /// Root-mean-square reprojection error over the corners that were used.
    pub rms_px: f64,
    /// How many tag corners the solve was based on.
    pub corners: usize,
    /// Tags that survived outlier rejection, in ascending order.
    pub tags: Vec<u16>,
}

/// One frame's solve, together with the rival poses the same corners admit.
///
/// A lone square tag has two poses that reproject its corners to the same
/// pixels, and with real corner noise the fit cannot say which is the camera:
/// on the pane that prompted this, the reported pose and its rival fitted the
/// same corners 0.42px and 0.46px, nine metres apart on the field, and two of
/// its frames fitted both answers to the same 0.29px. The rival is carried out
/// rather than discarded so something that *can* tell them apart — the ink
/// printed inside the tag's border, or the structure the tag is bolted to — has
/// a chance to, and so that a report can say the question was open when nothing
/// could.
#[derive(Debug, Clone)]
pub struct Candidates {
    pub chosen: Solution,
    /// Other plausible cameras that fit the corners about as well, metres from
    /// the winner. Empty when several tags pinned one pose between them.
    pub rivals: Vec<Solution>,
}

/// How far a rival has to be from the winning pose to count as a different
/// place rather than the same fit with different noise in it. A metre: the
/// ambiguity this exists for puts the camera on the far side of the field, and
/// two answers within a metre of each other are one answer.
const RIVAL_MIN_SEPARATION_M: f64 = 1.0;

/// Rivals carried out at most. The ambiguity a square leaves is one pose; more
/// than two near-ties is not a branch question but a broken frame.
const MAX_RIVALS: usize = 2;

/// A 3D point on a tag and where it was seen.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Correspondence {
    pub(crate) field: Vec3,
    pub(crate) image: (f64, f64),
    pub(crate) tag: u16,
}

/// Solve the camera pose from one frame's detections, with no outside hint.
pub fn solve(
    observations: &[TagObservation],
    layout: &TagLayout,
    tag_size: f64,
    camera: &Intrinsics,
) -> Option<Solution> {
    solve_all(observations, layout, tag_size, camera, None)
        .ok()
        .map(|candidates| candidates.chosen)
}

/// Solve the camera pose, preferring a particular end of the field.
///
/// `prefer` is what the colours around the tag say — red surroundings mean the
/// red end — and is used only when two poses fit the corners about equally
/// well. That is exactly the one-tag case, where the fit alone cannot choose.
/// The error says why nothing usable came back, for the report to print.
pub fn solve_preferring(
    observations: &[TagObservation],
    layout: &TagLayout,
    tag_size: f64,
    camera: &Intrinsics,
    prefer: Option<End>,
) -> Result<Solution, String> {
    solve_all(observations, layout, tag_size, camera, prefer).map(|candidates| candidates.chosen)
}

/// Solve the camera pose, and keep the rivals the corners cannot rule out.
///
/// Same ordering and same gates as [`solve_preferring`]; the difference is that
/// the poses that lost by a hair are handed back instead of dropped, because
/// losing the fit is not the same as being wrong — the fit is the one thing a
/// single square cannot use to choose between them.
pub fn solve_all(
    observations: &[TagObservation],
    layout: &TagLayout,
    tag_size: f64,
    camera: &Intrinsics,
    prefer: Option<End>,
) -> Result<Candidates, String> {
    let corners = correspondences(observations, layout, tag_size);
    if corners.len() < 4 {
        return Err("no tag in the frame is in the layout".to_string());
    }

    let mut pool = candidate_poses(observations, &corners, layout, tag_size, camera);

    // Restarting costs a few dozen small refinements, so it is only done when
    // the homography's own answer cannot be believed: it is missing, it is not
    // a camera that could have filmed the picture, or it is at the other end of
    // the field from the one the colours name.
    if pool.is_empty() || !settled(&pool, layout, prefer) {
        // A frame with no candidate at all is one whose homography would not
        // decompose — a handful of pixels of corner noise is enough. There is
        // still a pose in the corners; the seeds find it without needing the
        // decomposition to have worked.
        let (origin, agreed) = match pool.iter().min_by(|a, b| a.0.rms_px.total_cmp(&b.0.rms_px)) {
            Some((closest, agreed)) => (closest.pose.position, agreed.clone()),
            None => {
                let centre = centre_of(&corners);
                (centre + Vec3::new(0.0, 0.0, 10.0), corners.clone())
            }
        };
        let tags: Vec<u16> = {
            let mut ids: Vec<u16> = agreed.iter().map(|corner| corner.tag).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        for seed in restart_seeds(&agreed, layout, &tags, origin) {
            if let Some(solution) = refine(seed, &agreed, camera) {
                pool.push((solution, agreed.clone()));
            }
        }
    }

    choose(pool, layout, prefer)
}

/// Every corner of every known tag in the frame, ready to fit.
pub(crate) fn correspondences(
    observations: &[TagObservation],
    layout: &TagLayout,
    tag_size: f64,
) -> Vec<Correspondence> {
    let mut corners = Vec::new();
    for observation in observations {
        let Some(tag_corners) = layout.corners(observation.id, tag_size) else {
            continue;
        };
        for (index, corner) in tag_corners.iter().enumerate() {
            corners.push(Correspondence {
                field: *corner,
                image: (observation.corners[index][0], observation.corners[index][1]),
                tag: observation.id,
            });
        }
    }
    corners
}

/// The average of a set of corners, which for the tags of one frame is the
/// middle of the patch of tags that were read.
fn centre_of(corners: &[Correspondence]) -> Vec3 {
    let mut centre = Vec3::new(0.0, 0.0, 0.0);
    for corner in corners {
        centre = centre + corner.field;
    }
    centre.scaled(1.0 / corners.len().max(1) as f64)
}

/// One pose per tag, each refined against the tags it agrees with.
///
/// The proposal that explains the most tags is not picked here; every proposal
/// goes into the pool and [`choose`] ranks them, because the ranking needs to
/// weigh the physical constraints that a single tag cannot express.
fn candidate_poses(
    observations: &[TagObservation],
    corners: &[Correspondence],
    layout: &TagLayout,
    tag_size: f64,
    camera: &Intrinsics,
) -> Vec<(Solution, Vec<Correspondence>)> {
    let mut pool = Vec::new();
    for observation in observations {
        let Some(tag) = layout.tag(observation.id) else {
            continue;
        };
        let Some(candidate) =
            pose_from_tag(observation, tag.translation, tag.rotation, tag_size, camera)
        else {
            continue;
        };
        // Two passes. The one-tag pose is inexact — a small tag a long way off
        // gives a homography whose scale is easily a few percent out, which
        // moves the tag itself by more than a tight tolerance allows — so the
        // first pass is deliberately loose, the refinement fixes the pose, and
        // the second pass decides which tags really belong.
        let loose = agreed_corners(observations, corners, candidate, camera, loose_tolerance);
        if loose.len() < corners.len().min(4) {
            continue;
        }
        let Some(coarse) = refine(candidate, &loose, camera) else {
            continue;
        };
        let tight = agreed_corners(observations, corners, coarse.pose, camera, tight_tolerance);
        let (agreed, refined) = if tight.len() >= 4 && tight.len() < loose.len() {
            match refine(coarse.pose, &tight, camera) {
                Some(solution) => (tight, solution),
                None => (loose, coarse),
            }
        } else {
            (loose, coarse)
        };
        pool.push((refined, agreed));
    }
    pool
}

/// Whether the pool already holds an answer worth keeping: a possible camera,
/// at the end of the field the colours named when they named one.
fn settled(
    pool: &[(Solution, Vec<Correspondence>)],
    layout: &TagLayout,
    prefer: Option<End>,
) -> bool {
    pool.iter().any(|(solution, _)| {
        trust::plausible(&solution.pose, layout, &solution.tags).is_ok()
            && matches_preference(&solution.pose, layout, prefer)
    })
}

fn matches_preference(pose: &CameraPose, layout: &TagLayout, prefer: Option<End>) -> bool {
    let Some(prefer) = prefer else {
        return true;
    };
    end_of_x(pose.position.x, layout.field_length) == prefer
}

/// The best pose in the pool, and the near-ties it beat, or the reason there is
/// none.
fn choose(
    pool: Vec<(Solution, Vec<Correspondence>)>,
    layout: &TagLayout,
    prefer: Option<End>,
) -> Result<Candidates, String> {
    // The closest fit is the one whose rejection explains the frame best: if
    // the pose that fits the corners best is impossible, that is the story.
    let mut impossible: Option<(f64, String)> = None;
    let mut candidates: Vec<(Solution, Vec<Correspondence>)> = Vec::new();
    for (solution, agreed) in pool {
        match trust::plausible(&solution.pose, layout, &solution.tags) {
            Ok(()) => candidates.push((solution, agreed)),
            Err(reason) => {
                let closest = impossible
                    .as_ref()
                    .map(|(rms, _)| solution.rms_px < *rms)
                    .unwrap_or(true);
                if closest {
                    impossible = Some((solution.rms_px, reason));
                }
            }
        }
    }
    if candidates.is_empty() {
        return Err(match impossible {
            Some((_, reason)) => reason,
            None => "the tags seen do not determine a pose".to_string(),
        });
    }

    // Only poses whose fit is in the same league as the best one get a vote on
    // the side of the field: a pose that fits ten times worse is not a tie to be
    // broken by a colour reading, it is a worse fit.
    let best_rms = candidates
        .iter()
        .map(|(solution, _)| solution.rms_px)
        .fold(f64::INFINITY, f64::min);
    let mut ranked: Vec<(Solution, Vec<Correspondence>)> = candidates
        .into_iter()
        .filter(|(solution, _)| {
            solution.rms_px.is_finite() && solution.rms_px <= best_rms * 1.5 + 0.5
        })
        .collect();
    // Ordered by the fit before the tie-breaks are applied, so which pose wins
    // a genuine tie is a property of the poses and not of the order the pool
    // happened to be built in.
    ranked.sort_by(|a, b| a.0.rms_px.total_cmp(&b.0.rms_px));
    let mut winner = 0;
    for index in 1..ranked.len() {
        if better(&ranked[index].0, &ranked[winner].0, layout, prefer) {
            winner = index;
        }
    }
    if super::debug_enabled() {
        let listed: Vec<String> = ranked
            .iter()
            .map(|(solution, agreed)| {
                let position = solution.pose.position;
                format!(
                    "({:.2}, {:.2}, {:.2})m rms {:.2}px on {} corner(s)",
                    position.x,
                    position.y,
                    position.z,
                    solution.rms_px,
                    agreed.len()
                )
            })
            .collect();
        eprintln!(
            "  [pool] {} in the tie band: {}",
            ranked.len(),
            listed.join(" | ")
        );
    }
    let chosen = ranked.remove(winner).0;
    let mut rivals: Vec<Solution> = ranked
        .into_iter()
        .map(|(solution, _)| solution)
        .filter(|solution| {
            (solution.pose.position - chosen.pose.position).norm() >= RIVAL_MIN_SEPARATION_M
        })
        .collect();
    rivals.truncate(MAX_RIVALS);
    Ok(Candidates { chosen, rivals })
}

/// Whether `a` should be preferred to `b`: colour agreement first, then how
/// many tags backed it, then the fit.
fn better(a: &Solution, b: &Solution, layout: &TagLayout, prefer: Option<End>) -> bool {
    let a_matches = matches_preference(&a.pose, layout, prefer);
    let b_matches = matches_preference(&b.pose, layout, prefer);
    (a_matches, a.tags.len(), -a.rms_px) > (b_matches, b.tags.len(), -b.rms_px)
}

/// Where a camera could be, as starting points for the fit.
///
/// The twin pose of a single square is not a small perturbation away from the
/// right one — it is metres away, under the field or beyond it — so a seed has
/// to be placed where a camera plausibly is, not nudged from the wrong answer.
/// These ring the tag at broadcast heights (8m and 13m, above the boards and the
/// stand) in several directions, at roughly the distance the homography found
/// and somewhat beyond it, each one looking at the tag. The refinement then
/// settles on the nearest pose that fits the corners, and there is always one
/// that is a real camera.
fn restart_seeds(
    agreed: &[Correspondence],
    layout: &TagLayout,
    tags: &[u16],
    origin: Vec3,
) -> Vec<CameraPose> {
    if agreed.is_empty() {
        return Vec::new();
    }
    let centre = centre_of(agreed);
    let mut seeds = Vec::new();

    // Anchor everything on the answer that is being questioned. Even a mirror
    // pose has the tag's bearing roughly right — it is the distance and the
    // side of the field it gets wrong — so its bearing is the best clue there
    // is about where to look.
    let offset_from_tag = origin - centre;
    let bearing = offset_from_tag.y.atan2(offset_from_tag.x);
    let distance = offset_from_tag.norm();
    let distance = if distance.is_finite() && distance > 0.2 {
        distance
    } else {
        8.0
    };

    // Straight out from the tag's own face, at the distance the fit found and
    // on either side of it. This is what rescues the commonest case in the
    // fixtures and in the field: a tag so large in the frame that the camera is
    // a metre or two away, where every seed ten metres off is useless and the
    // wrong-side answer sits exactly behind the tag.
    for tag in tags {
        let Some(tag_pose) = layout.tag(*tag) else {
            continue;
        };
        let normal = Mat3::from_quaternion(tag_pose.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        for scale in [0.4f64, 0.7, 1.0, 1.6, 3.0] {
            let reach = (distance * scale).clamp(0.2, 45.0);
            seeds.push(look_towards(
                tag_pose.translation + normal.scaled(reach),
                tag_pose.translation,
            ));
        }
    }

    // …and a ring around the tag at the heights a broadcast camera is mounted
    // at, for the case where the camera is not on the tag's axis at all.
    for height in [8.0, 13.0] {
        for offset in [0.0f64, 60.0, 120.0, 180.0, 240.0, 300.0] {
            for scale in [0.8f64, 1.4] {
                let angle = bearing + offset.to_radians();
                let reach = (distance * scale).clamp(1.0, 45.0);
                seeds.push(look_towards(
                    Vec3::new(
                        centre.x + reach * angle.cos(),
                        centre.y + reach * angle.sin(),
                        height,
                    ),
                    centre,
                ));
            }
        }
    }
    seeds
}

/// A pose at `position` looking straight at `target`, with the field's `+z` up.
///
/// Built from the camera axes rather than a rotation about some axis: writing a
/// rotation out by hand gets the axis signs wrong every time.
pub(crate) fn look_towards(position: Vec3, target: Vec3) -> CameraPose {
    let forward = (target - position).normalized();
    let up = Vec3::new(0.0, 0.0, 1.0);
    let right = forward.cross(up);
    let right = if right.norm() < 1e-9 {
        // Looking straight down: any horizontal axis will do.
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        right.normalized()
    };
    let down = forward.cross(right);
    CameraPose {
        position,
        // Rows are the camera axes in field coordinates: field axes to camera
        // axes is the inverse (transpose) of that.
        rotation: Mat3::from_rows([right.to_array(), down.to_array(), forward.to_array()])
            .to_quaternion(),
    }
}

/// How far a corner may miss and still count, for a tag that wide in pixels.
///
/// The tolerance scales with how big the tag looks: a far tag is a few pixels
/// across and its corners cannot be localised to a pixel anyway, while a tag
/// that fills a quarter of the frame is held to a tighter standard.
fn tight_tolerance(pixel_width: f64) -> f64 {
    (0.4 * pixel_width).clamp(5.0, 15.0)
}

/// The first-pass tolerance, generous enough to survive an imperfect one-tag
/// pose but not so generous that a mis-read tag joins in.
fn loose_tolerance(pixel_width: f64) -> f64 {
    (1.5 * pixel_width).clamp(25.0, 120.0)
}

/// Corners of the tags that `pose` already explains, judged one tag at a time.
fn agreed_corners(
    observations: &[TagObservation],
    corners: &[Correspondence],
    pose: CameraPose,
    camera: &Intrinsics,
    tolerance: fn(f64) -> f64,
) -> Vec<Correspondence> {
    let rotation = Mat3::from_quaternion(pose.rotation);
    let mut agreed = Vec::new();
    for observation in observations {
        let tolerance = tolerance(observation.pixel_width());
        let tag_corners: Vec<&Correspondence> = corners
            .iter()
            .filter(|corner| corner.tag == observation.id)
            .collect();
        let mut errors = Vec::with_capacity(tag_corners.len());
        for corner in &tag_corners {
            let point = rotation.mul_vec(corner.field - pose.position);
            let Some((u, v)) = camera.project(point) else {
                // Behind the camera: nothing about this tag is explained.
                errors.push(f64::INFINITY);
                continue;
            };
            errors.push((u - corner.image.0).hypot(v - corner.image.1));
        }
        if errors.iter().all(|error| *error <= tolerance) {
            agreed.extend(tag_corners.into_iter().copied());
        }
    }
    agreed
}

/// The pose one tag implies, from its homography.
fn pose_from_tag(
    observation: &TagObservation,
    tag_translation: Vec3,
    tag_rotation: Quat,
    tag_size: f64,
    camera: &Intrinsics,
) -> Option<CameraPose> {
    // The tag face is the `y`-`z` plane of WPILib's tag frame, so each corner is
    // described by its (y, z) pair, in the same order as [`TagLayout::corners`]:
    // top-right first, running clockwise round the picture.
    let half = tag_size * 0.5;
    let object = [
        Vec3::new(half, half, 0.0),
        Vec3::new(half, -half, 0.0),
        Vec3::new(-half, -half, 0.0),
        Vec3::new(-half, half, 0.0),
    ];
    let homography = homography_from_four(&object, &observation.corners)?;

    // H = K [r1 r2 t] up to scale.
    let inverse_focal = Mat3::from_rows([
        [1.0 / camera.fx, 0.0, -camera.cx / camera.fx],
        [0.0, 1.0 / camera.fy, -camera.cy / camera.fy],
        [0.0, 0.0, 1.0],
    ]);
    let g1 = inverse_focal.mul_vec(homography.column(0));
    let g2 = inverse_focal.mul_vec(homography.column(1));
    let g3 = inverse_focal.mul_vec(homography.column(2));
    let scale = 2.0 / (g1.norm() + g2.norm());
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let mut r1 = g1.scaled(scale);
    let mut r2 = g2.scaled(scale);
    let mut translation = g3.scaled(scale);
    if !translation.z.is_finite() || translation.z.abs() < 1e-9 {
        return None;
    }
    // The homography's scale is arbitrary, so the tag could come out behind the
    // camera. Flip — and only *the in-plane* part, which keeps the rotation
    // proper — rather than throwing the only good tag away.
    if translation.z < 0.0 {
        r1 = r1.scaled(-1.0);
        r2 = r2.scaled(-1.0);
        translation = translation.scaled(-1.0);
    }

    // `r1` is the tag's `+y` in camera axes and `r2` its `+z`; the remaining
    // axis, out of the tag face, is their cross product.
    let y_axis = r1.normalized();
    let z_axis = (r2 - y_axis.scaled(y_axis.dot(r2))).normalized();
    let rotation = Mat3::from_columns(y_axis.cross(z_axis), y_axis, z_axis);
    let tag_frame = Mat3::from_quaternion(tag_rotation);
    let to_camera = rotation.mul_mat(&tag_frame.transpose());
    let centre = tag_translation - to_camera.transpose().mul_vec(translation);

    Some(CameraPose {
        position: centre,
        rotation: to_camera.to_quaternion(),
    })
}

/// Homography mapping tag coordinates (z = 0) to image coordinates.
fn homography_from_four(object: &[Vec3; 4], image: &[[f64; 2]; 4]) -> Option<Mat3> {
    // Unknowns: h11 h12 h13 h21 h22 h23 h31 h32, with h33 = 1.
    let mut matrix = [[0.0f64; 8]; 8];
    let mut rhs = [0.0f64; 8];
    for (index, (point, pixel)) in object.iter().zip(image.iter()).enumerate() {
        let (x, y) = (point.x, point.y);
        let (u, v) = (pixel[0], pixel[1]);
        matrix[2 * index] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
        rhs[2 * index] = u;
        matrix[2 * index + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
        rhs[2 * index + 1] = v;
    }
    let h = solve_dense(matrix, rhs)?;
    Some(Mat3::from_rows([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ]))
}

/// Reprojection error of one pose over a set of corners, in pixels.
///
/// The same measure [`refine`] reports, so a pose fitted to several frames can
/// be compared with the frames it was fitted to.
pub(crate) fn reprojection_rms(
    pose: &CameraPose,
    corners: &[Correspondence],
    camera: &Intrinsics,
) -> f64 {
    let residuals = residuals(
        Mat3::from_quaternion(pose.rotation),
        pose.position,
        corners,
        camera,
    );
    let cost: f64 = residuals.iter().map(|value| value * value).sum();
    (cost / corners.len().max(1) as f64).sqrt()
}

/// Levenberg–Marquardt over the six pose parameters.
pub(crate) fn refine(
    start: CameraPose,
    corners: &[Correspondence],
    camera: &Intrinsics,
) -> Option<Solution> {
    let mut rotation = Mat3::from_quaternion(start.rotation);
    let mut position = start.position;
    let mut lambda = 1e-3;
    let mut cost = residuals(rotation, position, corners, camera)
        .iter()
        .map(|value| value * value)
        .sum::<f64>();

    for _ in 0..80 {
        let base = residuals(rotation, position, corners, camera);
        // Numeric Jacobian: cheap at six parameters and a few dozen corners.
        let mut jacobian = vec![vec![0.0f64; 6]; base.len()];
        // Three rotations, then three translations.
        let steps = [1e-6; 6];
        for (axis, step) in steps.iter().enumerate() {
            let (rotated, moved) = if axis < 3 {
                (delta_rotation(axis, *step).mul_mat(&rotation), position)
            } else {
                (rotation, shift(position, axis - 3, *step))
            };
            let shifted = residuals(rotated, moved, corners, camera);
            for (row, value) in shifted.iter().enumerate() {
                jacobian[row][axis] = (value - base[row]) / step;
            }
        }

        // Normal equations.
        let mut normal = [[0.0f64; 6]; 6];
        let mut gradient = [0.0f64; 6];
        for row in 0..base.len() {
            for i in 0..6 {
                gradient[i] += jacobian[row][i] * base[row];
                for j in 0..6 {
                    normal[i][j] += jacobian[row][i] * jacobian[row][j];
                }
            }
        }

        let mut accepted = false;
        for _ in 0..8 {
            let mut damped = normal;
            for (i, row) in damped.iter_mut().enumerate() {
                row[i] += lambda * normal[i][i].max(1e-12);
            }
            let negative = gradient.map(|value| -value);
            let Some(step) = solve_dense(damped, negative) else {
                lambda *= 10.0;
                continue;
            };
            let candidate_rotation =
                delta_rotation_vector([step[0], step[1], step[2]]).mul_mat(&rotation);
            let candidate_position = Vec3::new(
                position.x + step[3],
                position.y + step[4],
                position.z + step[5],
            );
            let candidate = residuals(candidate_rotation, candidate_position, corners, camera);
            let candidate_cost = candidate.iter().map(|value| value * value).sum::<f64>();
            if candidate_cost < cost {
                rotation = candidate_rotation.orthonormalized();
                position = candidate_position;
                cost = candidate_cost;
                lambda = (lambda * 0.3).max(1e-9);
                accepted = true;
                break;
            }
            lambda *= 10.0;
        }
        if !accepted {
            break;
        }
    }

    let count = corners.len();
    let rms = (cost / count.max(1) as f64).sqrt();
    if !rms.is_finite() {
        return None;
    }
    let mut tags: Vec<u16> = corners.iter().map(|corner| corner.tag).collect();
    tags.sort_unstable();
    tags.dedup();
    Some(Solution {
        pose: CameraPose {
            position,
            rotation: rotation.to_quaternion(),
        },
        rms_px: rms,
        corners: count,
        tags,
    })
}

/// A pixel on a known 3D line, with its position along that line unknown.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LineObservation {
    pub start: Vec3,
    pub end: Vec3,
    pub image: (f64, f64),
}

/// One observation of a fixed camera, with independent orientation and focal length.
#[derive(Clone)]
pub(crate) struct PtzView {
    pub corners: Vec<Correspondence>,
    pub lines: Vec<LineObservation>,
    pub pose: CameraPose,
    pub camera: Intrinsics,
}

/// Jointly refine a shared camera centre and each frame's rotation and zoom.
/// The block normal equations eliminate each frame's four local parameters so
/// the system remains small regardless of the number of sampled frames.
/// A dense solve that gives up on the smallest of the systems here by returning
/// no step at all, so one unconstrained direction cannot abort a fit that is
/// fine in every other direction.
fn solve_or_zero<const N: usize>(matrix: [[f64; N]; N], rhs: [f64; N]) -> [f64; N] {
    solve_dense(matrix, rhs).unwrap_or([0.0; N])
}

/// Damping for one diagonal entry. Levenberg's `lambda` times the entry itself
/// is the usual choice, but it does nothing for an entry that is *zero* — a
/// direction the footage has no opinion about — leaving the block singular. The
/// floor is a fraction of the block's largest entry, so a direction nothing
/// constrains stays unconstrained instead of picking up an arbitrary step.
fn damping(lambda: f64, entry: f64, scale: f64) -> f64 {
    lambda * entry.abs().max(scale * 1e-6).max(1e-12)
}

pub(crate) fn refine_ptz(views: &[PtzView], fixed_position: bool) -> Option<Vec<PtzView>> {
    if views.is_empty() || views.iter().any(|view| view.corners.len() < 4) {
        return None;
    }
    let mut result = views.to_vec();
    let mut position = views[0].pose.position;
    for view in &mut result {
        view.pose.position = position;
    }
    let cost = |views: &[PtzView]| -> f64 {
        views
            .iter()
            .flat_map(|view| {
                ptz_residuals(
                    Mat3::from_quaternion(view.pose.rotation),
                    view.pose.position,
                    view,
                    &view.camera,
                )
            })
            .map(|x| x * x)
            .sum()
    };
    let mut current = cost(&result);
    let initial = current;
    let mut accepted = 0usize;
    let mut lambda = 1e-3;
    for _ in 0..80 {
        let mut global = [[0.0; 3]; 3];
        let mut gradient = [0.0; 3];
        let mut blocks = Vec::with_capacity(result.len());
        for view in &result {
            let rotation = Mat3::from_quaternion(view.pose.rotation);
            let base = ptz_residuals(rotation, position, view, &view.camera);
            let mut jac = vec![[0.0; 7]; base.len()];
            for (axis, step) in [1e-6_f64; 7].into_iter().enumerate() {
                let mut camera = view.camera;
                let mut rot = rotation;
                let mut pos = position;
                if axis < 3 {
                    pos = shift(pos, axis, step);
                } else if axis < 6 {
                    rot = delta_rotation(axis - 3, step).mul_mat(&rot);
                } else {
                    camera.fx *= step.exp();
                    camera.fy *= step.exp();
                }
                let shifted = ptz_residuals(rot, pos, view, &camera);
                for row in 0..base.len() {
                    jac[row][axis] = (shifted[row] - base[row]) / step;
                }
            }
            let mut normal = [[0.0; 7]; 7];
            let mut g = [0.0; 7];
            for (row, value) in jac.iter().zip(&base) {
                for i in 0..7 {
                    g[i] += row[i] * value;
                    for j in 0..7 {
                        normal[i][j] += row[i] * row[j];
                    }
                }
            }
            for i in 0..3 {
                gradient[i] += g[i];
                for j in 0..3 {
                    global[i][j] += normal[i][j];
                }
            }
            let local_scale = (3..7).map(|i| normal[i][i].abs()).fold(0.0f64, f64::max);
            let mut local = [[0.0; 4]; 4];
            let mut cross = [[0.0; 4]; 3];
            let mut local_g = [0.0; 4];
            for i in 0..4 {
                local_g[i] = g[i + 3];
                for j in 0..4 {
                    local[i][j] = normal[i + 3][j + 3];
                }
                local[i][i] += damping(lambda, local[i][i], local_scale);
                for j in 0..3 {
                    cross[j][i] = normal[j][i + 3];
                }
            }
            blocks.push((local, cross, local_g));
        }
        let global_scale = (0..3).map(|i| global[i][i].abs()).fold(0.0f64, f64::max);
        for (i, row) in global.iter_mut().enumerate() {
            row[i] += damping(lambda, row[i], global_scale);
        }
        let mut reduced = global;
        let mut reduced_g = gradient;
        for (local, cross, local_g) in &blocks {
            // A frame whose own block will not solve is dropped from the shared
            // system rather than ending the fit. The cost test below is what
            // decides whether the resulting step was worth taking.
            let inv_g = solve_or_zero(*local, *local_g);
            for i in 0..3 {
                reduced_g[i] -= (0..4).map(|k| cross[i][k] * inv_g[k]).sum::<f64>();
                for j in 0..3 {
                    let inv_cross = solve_or_zero(*local, cross[j]);
                    reduced[i][j] -= (0..4).map(|k| cross[i][k] * inv_cross[k]).sum::<f64>();
                }
            }
        }
        // The shared centre's normal block is singular exactly when the footage
        // does not pin it — which is the case this whole pass exists to report
        // rather than to give up on. The step is dropped and the fixed centre
        // kept when it cannot be solved; the caller's displacement probe is what
        // says how loose that leaves the answer.
        let dp = if fixed_position {
            [0.0; 3]
        } else {
            solve_dense(reduced, reduced_g.map(|v| -v)).unwrap_or([0.0; 3])
        };
        let mut candidate = result.clone();
        let next_position = position + Vec3::from_array(dp);
        let mut step_size = dp.iter().map(|v| v * v).sum::<f64>();
        for (view, (local, cross, local_g)) in candidate.iter_mut().zip(&blocks) {
            let mut rhs = local_g.map(|v| -v);
            for i in 0..4 {
                for j in 0..3 {
                    rhs[i] -= cross[j][i] * dp[j];
                }
            }
            let Some(step) = solve_dense(*local, rhs) else {
                continue;
            };
            // A focal-length step of e^2 would be a different lens, not a
            // refinement: clamp it and let the cost decide, rather than throwing
            // the whole fit away over one frame's wild step.
            let focal_step = step[3].clamp(-0.5, 0.5);
            step_size +=
                step[0] * step[0] + step[1] * step[1] + step[2] * step[2] + focal_step * focal_step;
            let rotation = delta_rotation_vector([step[0], step[1], step[2]])
                .mul_mat(&Mat3::from_quaternion(view.pose.rotation))
                .orthonormalized();
            view.pose = CameraPose {
                position: next_position,
                rotation: rotation.to_quaternion(),
            };
            view.camera.fx *= focal_step.exp();
            view.camera.fy *= focal_step.exp();
        }
        let next = cost(&candidate);
        if next.is_finite() && next < current {
            result = candidate;
            position = next_position;
            current = next;
            accepted += 1;
            lambda = (lambda * 0.3).max(1e-9);
            if step_size < 1e-16 {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
    }
    // A fit that took no acceptable step is the one worth printing: it means the
    // seed was already a local minimum, or the frames do not agree on a camera.
    if super::debug_enabled() && accepted == 0 {
        let _ = (initial, current);
        eprintln!("  [ptz] no step improved on the seed, cost {current:.1}");
    }
    Some(result)
}

/// Model-line samples one frame contributes at most, in equivalent independent
/// pixels. Samples along one edge are strongly correlated, so a long line cannot
/// outvote the tag corners by sheer count — but the cap has to leave a tag's
/// whole mount outline counting for something: at four it was outvoted, and the
/// ambiguity a lone small tag leaves survived it.
pub(crate) const LINE_WEIGHT_CAP: usize = 8;

fn ptz_residuals(rotation: Mat3, position: Vec3, view: &PtzView, camera: &Intrinsics) -> Vec<f64> {
    let mut out = residuals(rotation, position, &view.corners, camera);
    let weight = (LINE_WEIGHT_CAP as f64 / view.lines.len().max(LINE_WEIGHT_CAP) as f64).sqrt();
    for line in &view.lines {
        let a = camera.project(rotation.mul_vec(line.start - position));
        let b = camera.project(rotation.mul_vec(line.end - position));
        let error = if let (Some(a), Some(b)) = (a, b) {
            let dx = b.0 - a.0;
            let dy = b.1 - a.1;
            let length = dx.hypot(dy);
            if length > 1e-9 {
                (dx * (line.image.1 - a.1) - dy * (line.image.0 - a.0)) / length
            } else {
                camera.width as f64 + camera.height as f64
            }
        } else {
            camera.width as f64 + camera.height as f64
        };
        out.push(error * weight);
    }
    out
}

pub(crate) fn ptz_error(view: &PtzView) -> (f64, usize) {
    let values = ptz_residuals(
        Mat3::from_quaternion(view.pose.rotation),
        view.pose.position,
        view,
        &view.camera,
    );
    (
        values.iter().map(|v| v * v).sum(),
        view.corners.len() + view.lines.len().min(LINE_WEIGHT_CAP),
    )
}

fn residuals(
    rotation: Mat3,
    position: Vec3,
    corners: &[Correspondence],
    camera: &Intrinsics,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(corners.len() * 2);
    let penalty = camera.width as f64 + camera.height as f64;
    for corner in corners {
        let point = rotation.mul_vec(corner.field - position);
        match camera.project(point) {
            Some((u, v)) => {
                out.push(u - corner.image.0);
                out.push(v - corner.image.1);
            }
            // Behind the camera. Abandoning the iteration here would give up on
            // perfectly good starting poses, so push back with a large but
            // finite residual instead — the solver can walk away from it.
            None => {
                out.push(penalty);
                out.push(penalty);
            }
        }
    }
    out
}

fn delta_rotation(axis: usize, angle: f64) -> Mat3 {
    let mut vector = [0.0; 3];
    vector[axis] = angle;
    delta_rotation_vector(vector)
}

fn delta_rotation_vector(vector: [f64; 3]) -> Mat3 {
    let axis = Vec3::from_array(vector);
    let angle = axis.norm();
    if angle < 1e-12 {
        Mat3::IDENTITY
    } else {
        Mat3::from_axis_angle(axis, angle)
    }
}

fn shift(position: Vec3, axis: usize, amount: f64) -> Vec3 {
    match axis {
        0 => Vec3::new(position.x + amount, position.y, position.z),
        1 => Vec3::new(position.x, position.y + amount, position.z),
        _ => Vec3::new(position.x, position.y, position.z + amount),
    }
}

/// Gaussian elimination with partial pivoting, for the small dense systems here.
fn solve_dense<const N: usize>(mut matrix: [[f64; N]; N], mut rhs: [f64; N]) -> Option<[f64; N]> {
    for column in 0..N {
        let mut pivot = column;
        for row in column + 1..N {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][column].abs() < 1e-14 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        for row in column + 1..N {
            let factor = matrix[row][column] / matrix[column][column];
            let pivot_row = matrix[column];
            for (entry, pivot_entry) in matrix[row].iter_mut().zip(pivot_row.iter()).skip(column) {
                *entry -= factor * pivot_entry;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = [0.0f64; N];
    for row in (0..N).rev() {
        let mut sum = rhs[row];
        for column in row + 1..N {
            sum -= matrix[row][column] * solution[column];
        }
        solution[row] = sum / matrix[row][row];
    }
    if solution.iter().all(|value| value.is_finite()) {
        Some(solution)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout with tags placed by hand, for solver tests. Written in the
    /// older `AprilTagFieldLayout` schema, which is also worth covering.
    fn test_layout(tags: &[(u16, Vec3, Quat)]) -> TagLayout {
        let entries: Vec<serde_json::Value> = tags
            .iter()
            .map(|(id, translation, rotation)| {
                serde_json::json!({
                    "ID": id,
                    "pose": {
                        "translation": {
                            "x": translation.x,
                            "y": translation.y,
                            "z": translation.z,
                        },
                        "rotation": {
                            "quaternion": {
                                "W": rotation.w,
                                "X": rotation.x,
                                "Y": rotation.y,
                                "Z": rotation.z,
                            }
                        }
                    }
                })
            })
            .collect();
        let json = serde_json::json!({
            "field": { "length": 16.0, "width": 8.0 },
            "tags": entries,
        });
        TagLayout::from_json("test", &json.to_string()).expect("test layout parses")
    }

    /// Tags standing on the +y side of the field, facing -y.
    fn facing_tag(id: u16, at: Vec3) -> (u16, Vec3, Quat) {
        // Tag frame: +x out of the face, +y to the left, +z up. Rotating -90°
        // about the field's z axis turns "out of the face" from +x to -y, so
        // the tag looks back across the field.
        let rotation =
            Mat3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -std::f64::consts::FRAC_PI_2)
                .to_quaternion();
        (id, at, rotation)
    }

    /// A pose that looks from `position` straight at `target` with the field's
    /// `+z` up. This is the honest way to build a test pose: writing the
    /// rotation matrix out by hand gets the axis signs wrong every time.
    fn look_at(position: Vec3, target: Vec3) -> CameraPose {
        let forward = (target - position).normalized();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let right = forward.cross(up).normalized();
        let down = forward.cross(right);
        CameraPose {
            position,
            // Rows are the camera axes in field coordinates: field axes to
            // camera axes is the inverse (transpose) of that.
            rotation: Mat3::from_rows([right.to_array(), down.to_array(), forward.to_array()])
                .to_quaternion(),
        }
    }

    fn project_corners(
        layout: &TagLayout,
        ids: &[u16],
        tag_size: f64,
        pose: CameraPose,
        camera: &Intrinsics,
    ) -> Vec<TagObservation> {
        let rotation = Mat3::from_quaternion(pose.rotation);
        let mut observations = Vec::new();
        for id in ids {
            let corners = layout.corners(*id, tag_size).expect("corners");
            let mut image = [[0.0f64; 2]; 4];
            for (index, corner) in corners.iter().enumerate() {
                let point = rotation.mul_vec(*corner - pose.position);
                let (u, v) = camera.project(point).expect("corner in front");
                image[index] = [u, v];
            }
            observations.push(TagObservation {
                id: *id,
                corners: image,
                margin: 100.0,
                hamming: 0,
            });
        }
        observations
    }

    #[test]
    fn recovers_a_pose_from_one_tag() {
        let layout = test_layout(&[facing_tag(7, Vec3::new(8.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1920, 1080, 55.0);
        let truth = look_at(Vec3::new(3.0, 1.0, 4.0), Vec3::new(8.0, 6.0, 0.5));
        let observations = project_corners(&layout, &[7], 0.1651, truth, &camera);
        let solution = solve(&observations, &layout, 0.1651, &camera).expect("solves");
        assert!(
            (solution.pose.position - truth.position).norm() < 1e-6,
            "position {:?} vs {:?}",
            solution.pose.position,
            truth.position
        );
        assert!(solution.rms_px < 1e-6, "rms {}", solution.rms_px);
    }

    #[test]
    fn two_tags_pin_the_pose() {
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(6.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(10.0, 6.0, 0.5)),
        ]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let truth = look_at(Vec3::new(4.0, -1.0, 3.5), Vec3::new(8.0, 6.0, 0.5));
        let observations = project_corners(&layout, &[3, 4], 0.1651, truth, &camera);
        let solution = solve(&observations, &layout, 0.1651, &camera).expect("solves");
        assert!(
            (solution.pose.position - truth.position).norm() < 1e-5,
            "position {:?} vs {:?}",
            solution.pose.position,
            truth.position
        );
        assert_eq!(solution.tags, vec![3, 4]);
        assert_eq!(solution.corners, 8);
    }

    #[test]
    fn unknown_tags_are_ignored() {
        let layout = test_layout(&[facing_tag(3, Vec3::new(6.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let truth = look_at(Vec3::new(4.0, -1.0, 3.5), Vec3::new(6.0, 6.0, 0.5));
        let mut observations = project_corners(&layout, &[3], 0.1651, truth, &camera);
        let mut stray = observations[0].clone();
        stray.id = 99;
        observations.push(stray);
        let solution = solve(&observations, &layout, 0.1651, &camera).expect("solves");
        assert_eq!(solution.tags, vec![3]);
        assert!(solution.rms_px < 1e-6);
    }

    #[test]
    fn a_tag_outlier_is_dropped_when_the_others_agree() {
        // Three tags, one of them mis-read: the two that agree out-vote it. Two
        // tags could never be arbitrated — whichever was wrong, the pair is all
        // the information there is.
        let layout = test_layout(&[
            facing_tag(3, Vec3::new(4.0, 6.0, 0.5)),
            facing_tag(4, Vec3::new(8.0, 6.0, 0.5)),
            facing_tag(5, Vec3::new(12.0, 6.0, 0.5)),
        ]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let truth = look_at(Vec3::new(8.0, -1.0, 3.5), Vec3::new(8.0, 6.0, 0.5));
        let mut observations = project_corners(&layout, &[3, 4, 5], 0.1651, truth, &camera);
        // Corrupt the last tag's corners, as a mis-decode would.
        for corner in observations[2].corners.iter_mut() {
            corner[0] += 80.0;
            corner[1] -= 35.0;
        }
        let solution = solve(&observations, &layout, 0.1651, &camera).expect("solves");
        assert_eq!(solution.tags, vec![3, 4], "the bad tag should be dropped");
        assert!(
            (solution.pose.position - truth.position).norm() < 1e-4,
            "position {:?}",
            solution.pose.position
        );
    }

    #[test]
    fn a_tag_only_sixteen_pixels_across_still_solves() {
        // Real broadcast footage gives a few dozen pixels at best for a far tag;
        // the solver must not fall over just because the quad is small.
        let layout = test_layout(&[facing_tag(27, Vec3::new(8.0, 6.0, 0.55))]);
        let camera = Intrinsics::from_horizontal_fov(1920, 1080, 60.0);
        let truth = look_at(Vec3::new(8.0, -14.0, 6.0), Vec3::new(8.0, 6.0, 0.55));
        let observations = project_corners(&layout, &[27], 0.1651, truth, &camera);
        let width = observations[0].pixel_width();
        assert!(width < 20.0, "tag should be small, is {width:.1}px");

        let solution = solve(&observations, &layout, 0.1651, &camera).expect("solves");
        assert!(
            (solution.pose.position - truth.position).norm() < 0.5,
            "position {:?} vs {:?}",
            solution.pose.position,
            truth.position
        );
    }

    #[test]
    fn one_tag_never_solves_to_the_camera_behind_it() {
        // The bug this pins, measured on this exact geometry before it was
        // fixed: of forty frames whose corners were moved by less than a pixel,
        // thirteen solved to a camera eleven metres away and three metres under
        // the carpet, looking at the roof — and eight more solved to nothing at
        // all. The mirror fits those corners *better* than the truth did (0.06px
        // against 0.39px), which is why reprojection error can never be the
        // thing that rejects it.
        let layout = test_layout(&[facing_tag(13, Vec3::new(15.5, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let truth = look_at(Vec3::new(11.0, 2.0, 4.0), Vec3::new(15.5, 6.0, 0.5));
        let clean = project_corners(&layout, &[13], 0.1651, truth, &camera);
        let normal = Mat3::from_quaternion(layout.tag(13).expect("tag").rotation)
            .mul_vec(Vec3::new(1.0, 0.0, 0.0));

        let mut seed = 7u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f64 / (1u32 << 24) as f64) - 0.5
        };
        let mut solved = 0;
        for trial in 0..12 {
            let mut noisy = clean[0].clone();
            for corner in noisy.corners.iter_mut() {
                corner[0] += noise() * 1.5;
                corner[1] += noise() * 1.5;
            }
            let solution = solve(&[noisy], &layout, 0.1651, &camera)
                .unwrap_or_else(|| panic!("trial {trial} found no pose at all"));
            solved += 1;
            let position = solution.pose.position;
            assert!(
                position.z > 0.0,
                "trial {trial} put the camera {:.1}m under the carpet",
                -position.z
            );
            assert!(
                (position - layout.tag(13).unwrap().translation).dot(normal) > 0.0,
                "trial {trial} put the camera behind the tag's face"
            );
            assert!(
                (position - truth.position).norm() < 2.0,
                "trial {trial} is {:.1}m from the camera's actual position",
                (position - truth.position).norm()
            );
        }
        assert_eq!(solved, 12);
    }

    #[test]
    fn a_colour_reading_cannot_throw_away_the_only_pose() {
        // The colours around a tag break a tie; they do not overrule the
        // geometry. A red banner hung at the blue end would otherwise lose the
        // frame entirely.
        let layout = test_layout(&[facing_tag(3, Vec3::new(12.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let truth = look_at(Vec3::new(6.0, 1.0, 4.0), Vec3::new(12.0, 6.0, 0.5));
        let observations = project_corners(&layout, &[3], 0.1651, truth, &camera);
        let wrong_end = Some(End::LowX);
        let solution = solve_preferring(&observations, &layout, 0.1651, &camera, wrong_end)
            .expect("the only pose is still a pose");
        assert!(
            (solution.pose.position - truth.position).norm() < 1e-6,
            "position {:?} vs {:?}",
            solution.pose.position,
            truth.position
        );
    }

    #[test]
    fn nothing_usable_means_no_solution() {
        let layout = test_layout(&[facing_tag(3, Vec3::new(6.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        assert!(solve(&[], &layout, 0.1651, &camera).is_none());
    }
}

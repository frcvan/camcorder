//! Camera pose from AprilTag corners: planar initialisation, then a joint
//! refinement over every corner of every tag in the frame.
//!
//! One tag gives a planar homography, which is enough to start; two or more
//! tags constrain the pose properly, so the refinement uses all of them at once
//! and the result is the pose that best reprojects everything the camera saw.

use super::{
    layout::TagLayout,
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

/// A 3D point on a tag and where it was seen.
#[derive(Clone, Copy)]
struct Correspondence {
    field: Vec3,
    image: (f64, f64),
    tag: u16,
}

/// Solve the camera pose from one frame's detections.
///
/// Returns `None` when nothing in the frame can be solved — no known tags, or
/// too few corners.
pub fn solve(
    observations: &[TagObservation],
    layout: &TagLayout,
    tag_size: f64,
    camera: &Intrinsics,
) -> Option<Solution> {
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
    if corners.len() < 4 {
        return None;
    }

    // Every tag proposes a pose; the proposal that explains the most tags wins,
    // and only the tags it explains are used for the final solve. A mis-read
    // tag never gets a vote: it agrees with nothing, so no other tag's proposal
    // includes it. Fitting everything first and rejecting afterwards does not
    // work — one joint fit spreads the error over all the tags, so nothing ever
    // looks like an outlier.
    let mut best: Option<(Solution, usize, bool)> = None;
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
        let loose = agreed_corners(observations, &corners, candidate, camera, loose_tolerance);
        if loose.len() < corners.len().min(4) {
            continue;
        }
        let Some(coarse) = refine(candidate, &loose, camera) else {
            continue;
        };
        let tight = agreed_corners(observations, &corners, coarse.pose, camera, tight_tolerance);
        let (agreed, refined) = if tight.len() >= 4 && tight.len() < loose.len() {
            match refine(coarse.pose, &tight, camera) {
                Some(solution) => (tight, solution),
                None => (loose, coarse),
            }
        } else {
            (loose, coarse)
        };
        let agreed_tags = distinct_tags(&agreed);
        // A near-frontal tag admits a mirror solution with almost the same
        // error, and that one leaves the camera below the carpet. Nobody films
        // a field element from under the field, so it is a safe tie-break.
        let above_ground = refined.pose.position.z > 0.0;
        let better = match &best {
            None => true,
            Some((current, tags, ground)) => {
                (agreed_tags, above_ground, -refined.rms_px) > (*tags, *ground, -current.rms_px)
            }
        };
        if better {
            best = Some((refined, agreed_tags, above_ground));
        }
    }
    best.map(|(solution, _, _)| solution)
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

fn distinct_tags(corners: &[Correspondence]) -> usize {
    let mut tags: Vec<u16> = corners.iter().map(|corner| corner.tag).collect();
    tags.sort_unstable();
    tags.dedup();
    tags.len()
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
    // described by its (y, z) pair. Same order as [`TagLayout::corners`].
    let half = tag_size * 0.5;
    let object = [
        Vec3::new(-half, half, 0.0),
        Vec3::new(-half, -half, 0.0),
        Vec3::new(half, -half, 0.0),
        Vec3::new(half, half, 0.0),
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

/// Levenberg–Marquardt over the six pose parameters.
fn refine(start: CameraPose, corners: &[Correspondence], camera: &Intrinsics) -> Option<Solution> {
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
    fn nothing_usable_means_no_solution() {
        let layout = test_layout(&[facing_tag(3, Vec3::new(6.0, 6.0, 0.5))]);
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        assert!(solve(&[], &layout, 0.1651, &camera).is_none());
    }
}

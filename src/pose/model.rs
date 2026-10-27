//! Image observations of the AdvantageScope 2026 field model.
//!
//! Two kinds, both of which avoid having to *identify* an arbitrary line in the
//! picture — the thing that makes automatic registration hard:
//!
//! * **Alliance tape.** Long red and blue carpet tape, whose colour says which
//!   of the full-width field lines it can be. Only the tape that survives a
//!   straightness check across its whole extent is used.
//! * **Tag structure.** The box outline of whatever each tag is mounted on,
//!   keyed by tag id. The decoded tag already says which structure is in view,
//!   so nothing has to be guessed: the model says where to look, and the picture
//!   only has to have an edge there. This is also the evidence a lone tag cannot
//!   give — see [`structure_lines`].
//!
//! A match constrains the distance from the projected model line, not position
//! along it, which is all a single line can say.
//!
//! Both kinds are looked for near where the current pose projects them, so the
//! run needs a pose before it can use the model at all; [`super::station`]
//! re-matches as its fit improves for exactly that reason.

use serde::Deserialize;

use super::{
    frames::Frame,
    solve::LineObservation,
    types::{CameraPose, Intrinsics, Vec3},
};

#[derive(Deserialize)]
struct Marking {
    name: String,
    start: [f64; 3],
    end: [f64; 3],
}

/// One tag's mounting structure, as a box outline in field coordinates.
#[derive(Deserialize)]
struct StructureTag {
    id: u16,
    segments: Vec<[[f64; 3]; 2]>,
}

/// How far from a projected model edge the image edge is looked for on the first
/// pass, pixels. Wide, because the pose the search starts from may be metres out
/// along the view ray: a tag's own corners cannot tell a near camera from a far
/// one with a longer lens, and that is exactly the error the structure is there
/// to remove. Later passes start from a fitted pose and use a tighter window, so
/// a match is never to an edge that merely happened to be nearby.
pub(crate) const EDGE_SEARCH_WIDE_PX: f64 = 24.0;
pub(crate) const EDGE_SEARCH_TIGHT_PX: f64 = 10.0;
/// Luma difference per pixel an image edge needs before it counts.
const MIN_EDGE_GRADIENT: f64 = 10.0;
/// Points sampled along each model edge before support is measured.
const EDGE_SAMPLES: usize = 24;
/// Supported samples kept per edge. One edge is one constraint, however long.
const EDGE_KEEP: usize = 6;
/// Fraction of a model edge that must have an image edge under it.
const EDGE_SUPPORT_FRACTION: f64 = 0.4;
/// Most model observations taken from one frame, so a single busy picture
/// cannot outweigh the rest of the run.
const MAX_LINES_PER_FRAME: usize = 60;

/// Every image-to-model line observation in one frame: the alliance tape, which
/// the colour identifies on its own, and the outline of whatever structure the
/// tags that were decoded are mounted on, which the tag id identifies.
pub(crate) fn observations(
    frame: &Frame,
    pose: &CameraPose,
    camera: &Intrinsics,
    tags: &[u16],
    search_px: f64,
) -> Vec<LineObservation> {
    let mut found = tape_lines(frame, pose, camera);
    found.extend(structure_lines(frame, pose, camera, tags, search_px));
    found.truncate(MAX_LINES_PER_FRAME);
    found
}

/// Find unambiguous field-colour tape near a projected model line.
fn tape_lines(frame: &Frame, pose: &CameraPose, camera: &Intrinsics) -> Vec<LineObservation> {
    let Ok(markings) = serde_json::from_str::<Vec<Marking>>(include_str!(
        "../../assets/field/2026-rebuilt-advantagescope/markings.json"
    )) else {
        return Vec::new();
    };
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width < 3 || height < 3 || frame.rgb.len() != width * height * 3 {
        return Vec::new();
    }
    let mut result = Vec::new();
    for marking in markings {
        let color = if marking.name.contains("Red,") {
            0
        } else if marking.name.contains("Blue,") {
            2
        } else {
            continue;
        };
        let start = Vec3::from_array(marking.start);
        let end = Vec3::from_array(marking.end);
        let (Some(a), Some(b)) = (
            camera.project(pose.to_camera(start)),
            camera.project(pose.to_camera(end)),
        ) else {
            continue;
        };
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let length = dx.hypot(dy);
        if length < 60.0 {
            continue;
        }
        let (nx, ny) = (-dy / length, dx / length);
        let mut points = Vec::new();
        // Sample along the visible projection and look across a small corridor.
        // Group contiguous coloured pixels into runs; multiple runs are ambiguous.
        for n in 0..40 {
            let t = (n as f64 + 0.5) / 40.0;
            let x = a.0 + t * dx;
            let y = a.1 + t * dy;
            let mut runs: Vec<Vec<(f64, f64)>> = Vec::new();
            let mut run = Vec::new();
            for offset in -12..=12 {
                let px = (x + nx * offset as f64).round() as isize;
                let py = (y + ny * offset as f64).round() as isize;
                let matches = if px >= 0 && py >= 0 && px < width as isize && py < height as isize {
                    let i = (py as usize * width + px as usize) * 3;
                    let rgb = &frame.rgb[i..i + 3];
                    let dominant = rgb[color] as i16;
                    let others = (rgb[(color + 1) % 3] as i16).max(rgb[(color + 2) % 3] as i16);
                    dominant >= 90 && dominant - others >= 45 && dominant * 10 >= others * 14
                } else {
                    false
                };
                if matches {
                    run.push((px as f64, py as f64));
                } else if !run.is_empty() {
                    runs.push(std::mem::take(&mut run));
                }
            }
            if !run.is_empty() {
                runs.push(run);
            }
            if runs.len() != 1 || runs[0].len() < 2 || runs[0].len() > 12 {
                continue;
            }
            let run = &runs[0];
            let px = run.iter().map(|p| p.0).sum::<f64>() / run.len() as f64;
            let py = run.iter().map(|p| p.1).sum::<f64>() / run.len() as f64;
            points.push((px, py));
        }
        if points.len() < 10 {
            continue;
        }
        // A real straight line must agree across its extent, not just happen to
        // contain a few pixels of a similarly coloured robot or graphic.
        let mean = points
            .iter()
            .map(|p| nx * (p.0 - a.0) + ny * (p.1 - a.1))
            .sum::<f64>()
            / points.len() as f64;
        let variance = points
            .iter()
            .map(|p| (nx * (p.0 - a.0) + ny * (p.1 - a.1) - mean).powi(2))
            .sum::<f64>()
            / points.len() as f64;
        if variance.sqrt() > 2.5 {
            continue;
        }
        result.extend(
            points
                .into_iter()
                .map(|image| LineObservation { start, end, image }),
        );
    }
    result
}

/// Box outline of the structure the given tags stand on, where the picture has
/// an edge to match.
///
/// This is what a single planar tag cannot provide. A tag's own corners fix the
/// ratio of its size to its distance, so a pose can slide along the view ray and
/// grow the focal length to compensate, and the corners still land on the same
/// pixels. The top of the structure the tag is bolted to is not on the tag's
/// plane, so that trade no longer works: it moves across the picture when the
/// camera does. Only edges with a real image edge under them become
/// observations — the model says where to look, the picture says whether
/// anything is there.
fn structure_lines(
    frame: &Frame,
    pose: &CameraPose,
    camera: &Intrinsics,
    tags: &[u16],
    search_px: f64,
) -> Vec<LineObservation> {
    if tags.is_empty() {
        return Vec::new();
    }
    let Ok(structure) = serde_json::from_str::<Vec<StructureTag>>(include_str!(
        "../../assets/field/2026-rebuilt-advantagescope/structure.json"
    )) else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for entry in &structure {
        if !tags.contains(&entry.id) {
            continue;
        }
        for segment in &entry.segments {
            if let Some(line) = match_segment(frame, pose, camera, segment, search_px) {
                result.extend(line);
            }
        }
    }
    result
}

/// One model edge's image samples, or nothing when the picture has no edge there.
fn match_segment(
    frame: &Frame,
    pose: &CameraPose,
    camera: &Intrinsics,
    segment: &[[f64; 3]; 2],
    search_px: f64,
) -> Option<Vec<LineObservation>> {
    let start = Vec3::from_array(segment[0]);
    let end = Vec3::from_array(segment[1]);
    let mut projected = Vec::new();
    for index in 0..=EDGE_SAMPLES {
        let t = index as f64 / EDGE_SAMPLES as f64;
        let point = start + (end - start) * t;
        projected.push(
            camera
                .project(pose.to_camera(point))
                .map(|(u, v)| (u, v, t)),
        );
    }
    let mut supported = Vec::new();
    for index in 1..EDGE_SAMPLES {
        let (Some(previous), Some(current), Some(next)) =
            (projected[index - 1], projected[index], projected[index + 1])
        else {
            continue;
        };
        // The edge has to run somewhere: the local tangent of the projection.
        let dx = next.0 - previous.0;
        let dy = next.1 - previous.1;
        let length = dx.hypot(dy);
        if length < 1e-6 {
            continue;
        }
        let (nx, ny) = (-dy / length, dx / length);
        // Seek along the normal for the strongest gradient: a pose slightly off
        // still finds the edge, and the offset is what pulls it back.
        let mut best = None;
        let mut offset = -search_px;
        while offset <= search_px {
            let x = current.0 + nx * offset;
            let y = current.1 + ny * offset;
            if let Some(gradient) = perpendicular_gradient(frame, x, y, nx, ny) {
                if best.is_none_or(|(magnitude, _)| gradient > magnitude) {
                    best = Some((gradient, offset));
                }
            }
            offset += 0.5;
        }
        if let Some((magnitude, offset)) = best {
            if magnitude >= MIN_EDGE_GRADIENT {
                supported.push((current.0 + nx * offset, current.1 + ny * offset, offset));
            }
        }
    }
    if supported.len() * 2 < (EDGE_SAMPLES as f64 * EDGE_SUPPORT_FRACTION) as usize {
        return None;
    }
    let step = supported.len().div_ceil(EDGE_KEEP).max(1);
    Some(
        supported
            .into_iter()
            .step_by(step)
            .take(EDGE_KEEP)
            .map(|(x, y, _)| LineObservation {
                start,
                end,
                image: (x, y),
            })
            .collect(),
    )
}

/// Luma change per pixel along `(nx, ny)` at a point, bilinearly sampled.
fn perpendicular_gradient(frame: &Frame, x: f64, y: f64, nx: f64, ny: f64) -> Option<f64> {
    let before = sample_luma(frame, x - nx, y - ny)?;
    let after = sample_luma(frame, x + nx, y + ny)?;
    Some((after - before).abs() * 0.5)
}

fn sample_luma(frame: &Frame, x: f64, y: f64) -> Option<f64> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if x < 0.0 || y < 0.0 || x >= (width - 1) as f64 || y >= (height - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let at = |cx: usize, cy: usize| frame.pixels[cy * width + cx] as f64;
    Some(
        at(x0, y0) * (1.0 - fx) * (1.0 - fy)
            + at(x0 + 1, y0) * fx * (1.0 - fy)
            + at(x0, y0 + 1) * (1.0 - fx) * fy
            + at(x0 + 1, y0 + 1) * fx * fy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::solve;

    fn bundled_structure() -> Vec<StructureTag> {
        serde_json::from_str(include_str!(
            "../../assets/field/2026-rebuilt-advantagescope/structure.json"
        ))
        .expect("the bundled structure parses")
    }

    /// A frame with `segment` drawn on it as a dark-to-light edge.
    fn frame_with_edge(camera: &Intrinsics, line: &[[f64; 3]; 2]) -> (Frame, CameraPose) {
        let start = Vec3::from_array(line[0]);
        let end = Vec3::from_array(line[1]);
        let middle = start + (end - start) * 0.5;
        let pose = solve::look_towards(Vec3::new(middle.x - 5.0, middle.y - 5.0, 3.5), middle);
        let mut frame = Frame {
            time: 0.0,
            width: camera.width,
            height: camera.height,
            pixels: vec![60; camera.width as usize * camera.height as usize],
            rgb: vec![60; camera.width as usize * camera.height as usize * 3],
        };
        let (Some(a), Some(b)) = (
            camera.project(pose.to_camera(start)),
            camera.project(pose.to_camera(end)),
        ) else {
            return (frame, pose);
        };
        let width = camera.width as usize;
        let height = camera.height as usize;
        let length = (b.0 - a.0).hypot(b.1 - a.1);
        let (nx, ny) = (-(b.1 - a.1) / length, (b.0 - a.0) / length);
        let steps = (length * 4.0) as usize;
        for step in 0..=steps {
            let t = step as f64 / steps.max(1) as f64;
            let x = a.0 + t * (b.0 - a.0);
            let y = a.1 + t * (b.1 - a.1);
            for offset in 0..=2 {
                for sign in [-1.0, 1.0] {
                    let px = x - nx * offset as f64 * sign;
                    let py = y - ny * offset as f64 * sign;
                    if px < 0.0 || py < 0.0 || px >= width as f64 || py >= height as f64 {
                        continue;
                    }
                    frame.pixels[py as usize * width + px as usize] =
                        if sign < 0.0 { 210 } else { 40 };
                }
            }
        }
        (frame, pose)
    }

    #[test]
    fn tag_structure_edges_are_matched_where_the_picture_has_an_edge() {
        let structure = bundled_structure();
        let entry = structure
            .iter()
            .find(|entry| entry.id == 1)
            .expect("tag 1 has structure");
        let segment = entry
            .segments
            .iter()
            .max_by(|a, b| {
                let length = |segment: &[[f64; 3]; 2]| {
                    (Vec3::from_array(segment[0]) - Vec3::from_array(segment[1])).norm()
                };
                length(a).total_cmp(&length(b))
            })
            .expect("a longest edge");
        let camera = Intrinsics::from_horizontal_fov(640, 480, 60.0);
        let (frame, pose) = frame_with_edge(&camera, segment);

        let matched = structure_lines(&frame, &pose, &camera, &[1], EDGE_SEARCH_WIDE_PX);
        assert!(
            matched
                .iter()
                .any(|observation| observation.start == Vec3::from_array(segment[0])),
            "the structure edge drawn on the frame was not matched: {matched:?}"
        );
        // Every match has to lie on the model line it is claimed to be, within
        // the window the search was allowed.
        for observation in &matched {
            let a = camera
                .project(pose.to_camera(observation.start))
                .expect("the model edge is in front of the camera");
            let b = camera
                .project(pose.to_camera(observation.end))
                .expect("the model edge is in front of the camera");
            let dx = b.0 - a.0;
            let dy = b.1 - a.1;
            let length = dx.hypot(dy);
            let offset = (dx * (observation.image.1 - a.1) - dy * (observation.image.0 - a.0))
                .abs()
                / length;
            assert!(
                offset <= EDGE_SEARCH_WIDE_PX + 0.5,
                "a match {offset:.1}px off the model line it was matched to"
            );
        }

        // The same picture, but the run decoded a different tag: nothing is
        // looked for. Each tag's structure is known by its id, so a match can
        // never be to the wrong structure.
        assert!(structure_lines(&frame, &pose, &camera, &[2], EDGE_SEARCH_WIDE_PX).is_empty());
        assert!(structure_lines(&frame, &pose, &camera, &[], EDGE_SEARCH_WIDE_PX).is_empty());

        // The model says where to look; the picture has to have an edge there.
        let blank = Frame {
            pixels: vec![60; 640 * 480],
            ..frame.clone()
        };
        assert!(
            structure_lines(&blank, &pose, &camera, &[1], EDGE_SEARCH_WIDE_PX).is_empty(),
            "a featureless frame cannot report structure edges"
        );
    }

    #[test]
    fn red_field_tape_is_matched_to_its_model_line_not_a_blue_or_blank_image() {
        let camera = Intrinsics::from_horizontal_fov(640, 480, 75.0);
        let pose = solve::look_towards(Vec3::new(8.0, -6.0, 8.0), Vec3::new(12.5, 4.0, 0.0));
        let mut frame = Frame {
            time: 0.0,
            width: 640,
            height: 480,
            pixels: vec![60; 640 * 480],
            rgb: vec![60; 640 * 480 * 3],
        };
        let start = camera
            .project(pose.to_camera(Vec3::new(12.5377, 0.0062, 0.000127)))
            .expect("start");
        let end = camera
            .project(pose.to_camera(Vec3::new(12.5377, 8.0628, 0.000127)))
            .expect("end");
        for n in 0..2000 {
            let t = n as f64 / 1999.0;
            let x = (start.0 + t * (end.0 - start.0)).round() as isize;
            let y = (start.1 + t * (end.1 - start.1)).round() as isize;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let px = x + dx;
                    let py = y + dy;
                    if (0..640).contains(&px) && (0..480).contains(&py) {
                        let i = (py as usize * 640 + px as usize) * 3;
                        frame.rgb[i..i + 3].copy_from_slice(&[220, 30, 30]);
                    }
                }
            }
        }
        let lines = observations(&frame, &pose, &camera, &[], EDGE_SEARCH_WIDE_PX);
        assert!(
            lines.len() >= 10,
            "red tape was not matched: {}",
            lines.len()
        );
        assert!(lines
            .iter()
            .all(|line| (line.start.x - 12.5377).abs() < 0.001));
        frame.rgb.fill(60);
        assert!(observations(&frame, &pose, &camera, &[], EDGE_SEARCH_WIDE_PX).is_empty());
    }
}

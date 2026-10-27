//! Putting the tag's corners where the edges actually cross.
//!
//! The detector hands back a quad that is already sub-pixel — it refines its
//! edges before decoding — but it refines them for the *decoder*, on the
//! thresholded picture, and it stops when the bits are readable. What the solve
//! needs is the intersection of the tag's two border lines on the greyscale
//! picture itself: a tag thirty pixels across has its whole pose riding on
//! corners that are a couple of tenths of a pixel out, and a tag in a broadcast
//! sub-view is thirty pixels.
//!
//! For each of the four edges this finds the border independently at several
//! points along it, fits a line through them, and intersects adjacent lines.
//! Two lines rather than four points is the point: a straight border through
//! five measurements averages the encoder's ringing and the lens's softness
//! instead of trusting any one sample of them.
//!
//! Every step is bounded and falls back to the detector's corner. A corner is
//! only moved when both of its edges were found, both lines fit their points,
//! and the intersection is close to where the decoder said it was — so a
//! picture with nothing to refine (a blank patch, a corner off the frame)
//! leaves the solve exactly as it was.

/// Points measured along each edge, and the middle of the edge they occupy.
/// The ends are left out because there the neighbouring border is in the way.
const EDGE_SAMPLES: usize = 7;
const EDGE_START: f64 = 0.28;
const EDGE_END: f64 = 0.72;
/// How far across the border to look for it, and in what steps. Four pixels is
/// enough to reach a border that the decoder's own quad missed by a pixel or
/// two, and small enough that the *neighbouring* border of the tag — eight
/// pixels away on a small tag — is not found instead.
const SEARCH_PX: f64 = 3.0;
const SEARCH_STEP_PX: f64 = 0.25;
/// The smallest brightness step that counts as a border at all. Below this the
/// picture has no edge there, and a peak would be noise standing in for one.
const MIN_STEP: f64 = 8.0;
/// How far a fitted corner may move from the decoder's. Past this the fit found
/// something other than the tag's corner: a shadow, the field line it stands
/// on, or the neighbouring tag.
const MAX_SHIFT_PX: f64 = 4.0;
/// A line whose measured points scatter further than this from it was not an
/// edge, and its intersection would be a fabricated corner.
const MAX_LINE_SPREAD_PX: f64 = 1.2;

/// The four corners of one tag, refined against the picture.
///
/// `gray` is `width * height` bytes, top row first. The corners are in the
/// detector's order; the result keeps that order, and keeps any corner this
/// could not improve.
pub fn refine(gray: &[u8], width: usize, height: usize, corners: &[[f64; 2]; 4]) -> [[f64; 2]; 4] {
    if width == 0 || height == 0 || gray.len() < width * height {
        return *corners;
    }
    let mut lines: [Option<Line>; 4] = [None; 4];
    for (index, line) in lines.iter_mut().enumerate() {
        *line = border_line(
            gray,
            width,
            height,
            corners[index],
            corners[(index + 1) % 4],
        );
    }

    let mut refined = *corners;
    for index in 0..4 {
        // Corner `index` is where the edge ending at it meets the edge
        // starting at it.
        let before = lines[(index + 3) % 4];
        let after = lines[index];
        let (Some(before), Some(after)) = (before, after) else {
            continue;
        };
        let Some(point) = before.intersect(&after) else {
            continue;
        };
        let shifted = (point[0] - corners[index][0]).hypot(point[1] - corners[index][1]);
        if shifted > MAX_SHIFT_PX || !point[0].is_finite() || !point[1].is_finite() {
            continue;
        }
        let inside = point[0] >= -1.0
            && point[1] >= -1.0
            && point[0] <= width as f64 + 1.0
            && point[1] <= height as f64 + 1.0;
        if inside {
            refined[index] = point;
        }
    }
    refined
}

/// A line in the image: a point on it and a unit direction.
#[derive(Clone, Copy, Debug)]
struct Line {
    point: [f64; 2],
    direction: [f64; 2],
}

impl Line {
    /// Where two lines cross, or `None` when they are close to parallel.
    fn intersect(&self, other: &Line) -> Option<[f64; 2]> {
        let cross = self.direction[0] * other.direction[1] - self.direction[1] * other.direction[0];
        if cross.abs() < 1e-3 {
            return None;
        }
        let delta = [
            other.point[0] - self.point[0],
            other.point[1] - self.point[1],
        ];
        let along = (delta[0] * other.direction[1] - delta[1] * other.direction[0]) / cross;
        Some([
            self.point[0] + along * self.direction[0],
            self.point[1] + along * self.direction[1],
        ])
    }
}

/// Fit a line to the border between two corners.
fn border_line(
    gray: &[u8],
    width: usize,
    height: usize,
    from: [f64; 2],
    to: [f64; 2],
) -> Option<Line> {
    let span = [to[0] - from[0], to[1] - from[1]];
    let length = (span[0] * span[0] + span[1] * span[1]).sqrt();
    if length < 4.0 {
        return None;
    }
    let direction = [span[0] / length, span[1] / length];
    // The normal points across the border, which is the direction the border is
    // searched in.
    let normal = [-direction[1], direction[0]];

    let mut points = Vec::with_capacity(EDGE_SAMPLES);
    for sample in 0..EDGE_SAMPLES {
        let t = EDGE_START + (EDGE_END - EDGE_START) * sample as f64 / (EDGE_SAMPLES - 1) as f64;
        let at = [from[0] + span[0] * t, from[1] + span[1] * t];
        if let Some(offset) = border_offset(gray, width, height, at, normal) {
            points.push([at[0] + normal[0] * offset, at[1] + normal[1] * offset]);
        }
    }
    // Half a fitted line is not a line.
    if points.len() * 2 < EDGE_SAMPLES {
        return None;
    }

    let count = points.len() as f64;
    let centroid = [
        points.iter().map(|point| point[0]).sum::<f64>() / count,
        points.iter().map(|point| point[1]).sum::<f64>() / count,
    ];
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for point in &points {
        let (dx, dy) = (point[0] - centroid[0], point[1] - centroid[1]);
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let fitted = [angle.cos(), angle.sin()];
    // A border the decoder placed is close to the one it walked to; take the
    // principal axis in whichever direction is nearer the original edge, so a
    // line and its reverse are the same line.
    let fitted = if fitted[0] * direction[0] + fitted[1] * direction[1] < 0.0 {
        [-fitted[0], -fitted[1]]
    } else {
        fitted
    };

    // Total least squares residual: the spread across the fitted direction.
    let spread = points
        .iter()
        .map(|point| {
            let dx = point[0] - centroid[0];
            let dy = point[1] - centroid[1];
            (dx * -fitted[1] + dy * fitted[0]).abs()
        })
        .fold(0.0f64, f64::max);
    if !spread.is_finite() || spread > MAX_LINE_SPREAD_PX {
        return None;
    }
    Some(Line {
        point: centroid,
        direction: fitted,
    })
}

/// How far along `normal` from `at` the strongest border lies, in pixels.
///
/// The profile across the border is sampled and differentiated; the peak of the
/// derivative is interpolated with a parabola, which is what makes this
/// sub-pixel rather than the best quarter-pixel.
fn border_offset(
    gray: &[u8],
    width: usize,
    height: usize,
    at: [f64; 2],
    normal: [f64; 2],
) -> Option<f64> {
    let steps = (SEARCH_PX / SEARCH_STEP_PX).round() as isize;
    let mut values = Vec::with_capacity((2 * steps + 1) as usize);
    for step in -steps..=steps {
        let offset = step as f64 * SEARCH_STEP_PX;
        let x = at[0] + normal[0] * offset;
        let y = at[1] + normal[1] * offset;
        values.push(sample(gray, width, height, x, y)?);
    }

    // The border's position is the halfway point between the two levels, not
    // the steepest sample: a border that the lens and the encoder have spread
    // over a pixel has a whole stretch of equally steep samples, and the
    // halfway crossing is the one place in that stretch that stays put as the
    // smear changes. The steepest pair only says *which* crossing is the
    // border, when the search happens to hold more than one.
    let darkest = values.iter().copied().fold(f64::INFINITY, f64::min);
    let brightest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if brightest - darkest < MIN_STEP {
        return None;
    }
    let middle_level = 0.5 * (darkest + brightest);

    // Of the pairs that bracket the halfway level — a smeared border brackets
    // it in several — the one that changes fastest is the border itself.
    let mut crossing: Option<(usize, f64)> = None;
    for index in 0..values.len() - 1 {
        let (a, b) = (values[index], values[index + 1]);
        let brackets = (a - middle_level) * (b - middle_level) <= 0.0;
        if !brackets || (a - b).abs() < 1e-9 {
            continue;
        }
        let step = (b - a).abs();
        if crossing
            .map(|(_, sharpest)| step > sharpest)
            .unwrap_or(true)
        {
            crossing = Some((index, step));
        }
    }
    let (index, _) = crossing?;
    let between = (middle_level - values[index]) / (values[index + 1] - values[index]);
    if !between.is_finite() {
        return None;
    }
    let between = between.clamp(0.0, 1.0);

    // Sample `index` sits at offset `index - steps`; the crossing is `between`
    // of the way to the next one.
    Some((index as f64 + between - steps as f64) * SEARCH_STEP_PX)
}

/// Luma at a fractional position, bilinear, or `None` outside the frame.
fn sample(gray: &[u8], width: usize, height: usize, x: f64, y: f64) -> Option<f64> {
    if x < 0.0 || y < 0.0 {
        return None;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let xi = x0 as usize;
    let yi = y0 as usize;
    if xi + 1 >= width || yi + 1 >= height {
        return None;
    }
    let fx = x - x0;
    let fy = y - y0;
    let at = |dx: usize, dy: usize| gray[(yi + dy) * width + xi + dx] as f64;
    Some(
        at(0, 0) * (1.0 - fx) * (1.0 - fy)
            + at(1, 0) * fx * (1.0 - fy)
            + at(0, 1) * (1.0 - fx) * fy
            + at(1, 1) * fx * fy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: usize = 120;
    /// The square's true corner, at a position no integer pixel grid contains.
    const TRUE_LEFT: f64 = 30.4;
    const TRUE_TOP: f64 = 24.7;
    const TRUE_SIDE: f64 = 40.0;

    /// A dark square on a light field, with anti-aliased edges: each pixel is
    /// the average of 16 samples, so an edge at 30.4 really is 40% into the
    /// pixel. This is the picture a real edge produces once a lens and a codec
    /// have had it, and it is the one whose sub-pixel position is known.
    fn square(blur: bool) -> Vec<u8> {
        let mut pixels = vec![0u8; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                let mut inside = 0;
                for sy in 0..4 {
                    for sx in 0..4 {
                        let px = x as f64 + (sx as f64 + 0.5) / 4.0;
                        let py = y as f64 + (sy as f64 + 0.5) / 4.0;
                        let within = (TRUE_LEFT..=TRUE_LEFT + TRUE_SIDE).contains(&px)
                            && (TRUE_TOP..=TRUE_TOP + TRUE_SIDE).contains(&py);
                        if within {
                            inside += 1;
                        }
                    }
                }
                pixels[y * SIDE + x] = (240.0 - 220.0 * inside as f64 / 16.0).round() as u8;
            }
        }
        if blur {
            pixels = blur3x3(&pixels);
        }
        pixels
    }

    fn blur3x3(gray: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let sy = (y as i32 + dy).clamp(0, SIDE as i32 - 1) as usize;
                        let sx = (x as i32 + dx).clamp(0, SIDE as i32 - 1) as usize;
                        sum += gray[sy * SIDE + sx] as u32;
                        count += 1;
                    }
                }
                out[y * SIDE + x] = (sum / count) as u8;
            }
        }
        out
    }

    /// The detector's quad, a pixel and a bit outside the true one in every
    /// direction — what the solve gets when the decoder's own edge refinement
    /// lands a pixel wide.
    fn detected() -> [[f64; 2]; 4] {
        let truth = truth();
        let push = 1.2;
        [
            [truth[0][0] + push, truth[0][1] - push],
            [truth[1][0] + push, truth[1][1] + push],
            [truth[2][0] - push, truth[2][1] + push],
            [truth[3][0] - push, truth[3][1] - push],
        ]
    }

    /// Where the corners are in the sampler's coordinates: the centre of pixel
    /// `i` is at `i`, so the continuous edge at `TRUE_LEFT` is half a pixel
    /// before it.
    fn truth() -> [[f64; 2]; 4] {
        let (left, top) = (TRUE_LEFT - 0.5, TRUE_TOP - 0.5);
        let right = left + TRUE_SIDE;
        let bottom = top + TRUE_SIDE;
        [[right, top], [right, bottom], [left, bottom], [left, top]]
    }

    fn worst_error(corners: &[[f64; 2]; 4]) -> f64 {
        corners
            .iter()
            .zip(truth().iter())
            .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn a_corner_lands_where_the_two_borders_cross() {
        // The square's edges do not fall on whole pixels, so its corners are
        // only right if the refinement reads where the borders actually are.
        let picture = square(false);
        let refined = refine(&picture, SIDE, SIDE, &detected());
        assert!(
            worst_error(&refined) < 0.3,
            "refined corners are up to {:.2}px out: {refined:?}",
            worst_error(&refined)
        );
        // …and the decoder's own quad was further out than that, or this test
        // would pass against a refinement that did nothing.
        assert!(worst_error(&detected()) > 1.5);
    }

    #[test]
    fn a_soft_edge_is_still_one_line() {
        // Broadcast footage is never sharp: the codec and the lens spread the
        // border over a couple of pixels. The line fit is what makes the same
        // measurement survive that.
        let picture = square(true);
        let refined = refine(&picture, SIDE, SIDE, &detected());
        assert!(
            worst_error(&refined) < 0.5,
            "soft edges left the corners up to {:.2}px out",
            worst_error(&refined)
        );
    }

    #[test]
    fn a_picture_with_no_edges_leaves_the_corners_alone() {
        // Nothing to fit means nothing to change: a flat frame must not have a
        // corner invented for it.
        let flat = vec![128u8; SIDE * SIDE];
        let corners = detected();
        assert_eq!(refine(&flat, SIDE, SIDE, &corners), corners);

        // Nor may a corner be dragged off to something that is not a border: a
        // picture whose only feature is a line through the middle.
        let mut striped = vec![200u8; SIDE * SIDE];
        for y in 55..70 {
            for x in 0..SIDE {
                striped[y * SIDE + x] = 20;
            }
        }
        let moved = refine(&striped, SIDE, SIDE, &corners);
        for (corner, original) in moved.iter().zip(corners.iter()) {
            let shift = (corner[0] - original[0]).hypot(corner[1] - original[1]);
            assert!(shift <= MAX_SHIFT_PX, "corner moved {shift:.2}px");
        }
    }

    #[test]
    fn a_buffer_or_frame_too_small_to_read_is_left_alone() {
        let corners = detected();
        assert_eq!(refine(&[0u8; 4], SIDE, SIDE, &corners), corners);
        assert_eq!(refine(&vec![0u8; SIDE * SIDE], 0, 0, &corners), corners);
        assert_eq!(sample(&vec![0u8; SIDE * SIDE], SIDE, SIDE, -1.0, 5.0), None);
        assert_eq!(
            sample(&vec![0u8; SIDE * SIDE], SIDE, SIDE, SIDE as f64, 5.0),
            None
        );
    }

    #[test]
    fn two_parallel_lines_have_no_corner_between_them() {
        let a = Line {
            point: [0.0, 0.0],
            direction: [1.0, 0.0],
        };
        let b = Line {
            point: [0.0, 5.0],
            direction: [1.0, 0.0],
        };
        assert_eq!(a.intersect(&b), None);
        let cross = Line {
            point: [3.0, 0.0],
            direction: [0.0, 1.0],
        };
        let point = a.intersect(&cross).expect("they cross");
        assert!((point[0] - 3.0).abs() < 1e-9 && point[1].abs() < 1e-9);
    }
}

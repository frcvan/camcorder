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
/// How far from the decoder's own edge the border is looked for first, pixels.
/// The decoder's quad is good to about a pixel, and the tag's next cell is a
/// whole cell away — three pixels on a tag twenty-four pixels across — so
/// proximity is what separates the border from the code printed beside it. A
/// quad further out than this still gets its border, from the whole corridor.
const NEAR_PX: f64 = 1.5;
/// How far apart the parallel profiles are, across the edge direction. The codec
/// and the lens put noise on individual pixels, not on an edge, and a tag thirty
/// pixels across is small enough that one profile's noise is a tenth of a pixel.
const PROFILE_SPACING_PX: f64 = 0.75;
/// What a corner is worth when the fit could not measure it at all: the
/// decoder's own quad is good to somewhere under a pixel, and that is the best
/// anyone knows about a corner this refinement left alone.
const FALLBACK_SIGMA_PX: f64 = 0.75;
/// Uncertainties are clamped rather than trusted: a line fitted to almost no
/// points, or two borders nearly parallel in the picture, can divide a small
/// spread by a small sine and claim a precision no corner has.
const MIN_SIGMA_PX: f64 = 0.05;
const MAX_SIGMA_PX: f64 = 2.0;
/// The smallest sine of the angle between two borders that is used to propagate
/// an edge's uncertainty to the corner they cross at. A tag seen very obliquely
/// has two long, nearly parallel borders, and its far corner really is that
/// badly determined.
const MIN_CROSSING_SIN: f64 = 0.25;
/// What share of a border's ramp width its position is uncertain by.
///
/// This is the measure that matters, and it took a fixture to find: the spread
/// of the sampled points around the fitted line says how *straight* an edge is,
/// and a perfectly straight one has no spread at all however coarsely its
/// position was bracketed. What does set the precision is how wide the profile
/// spreads the crossing over — a sharp border crosses its halfway level in one
/// pixel and a blurred one in three — so the uncertainty scales with that.
const RAMP_SHARE: f64 = 0.25;
/// The smallest an edge's own uncertainty is ever taken to be: the profile is
/// sampled on a quarter-pixel grid, and a crossing between two samples cannot be
/// known better than that grid.
const MIN_EDGE_SIGMA_PX: f64 = 0.06;

/// The four corners of one tag, refined against the picture, and how well the
/// picture pinned each of them.
#[derive(Debug, Clone, Copy)]
pub struct Refined {
    /// Where the borders cross, in the detector's order. Any corner this could
    /// not improve is the detector's own.
    pub corners: [[f64; 2]; 4],
    /// How far each corner could be from where it was put, in pixels, one number
    /// per corner in the same order: the two borders' own scatter propagated to
    /// the corner they cross at.
    ///
    /// This is what the joint fits weight by. A tag thirty pixels across is
    /// pinned several times less tightly than one eighty pixels across, and on
    /// footage that holds both, counting them alike lets the small tag's noise
    /// lean a pose the large tag had already settled — and the small tag is also
    /// the one more likely to have been read at the wrong place.
    pub sigma_px: [f64; 4],
}

/// The four corners of one tag, refined against the picture.
///
/// `gray` is `width * height` bytes, top row first. The corners are in the
/// detector's order; the result keeps that order, and keeps any corner this
/// could not improve.
pub fn refine(gray: &[u8], width: usize, height: usize, corners: &[[f64; 2]; 4]) -> [[f64; 2]; 4] {
    refine_with_quality(gray, width, height, corners).corners
}

/// The same refinement, together with the uncertainty it leaves behind.
pub fn refine_with_quality(
    gray: &[u8],
    width: usize,
    height: usize,
    corners: &[[f64; 2]; 4],
) -> Refined {
    let fallback = Refined {
        corners: *corners,
        sigma_px: [FALLBACK_SIGMA_PX; 4],
    };
    if width == 0 || height == 0 || gray.len() < width * height {
        return fallback;
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

    let mut refined = fallback;
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
            refined.corners[index] = point;
            // Each border's uncertainty is measured across its own normal; where
            // two of them cross, a displacement along one is divided by the sine
            // of their angle, which is what makes a grazing view's corners as
            // poor as they look.
            let cross = (before.direction[0] * after.direction[1]
                - before.direction[1] * after.direction[0])
                .abs();
            let sigma = before.uncertainty.hypot(after.uncertainty) / cross.max(MIN_CROSSING_SIN);
            refined.sigma_px[index] = sigma.clamp(MIN_SIGMA_PX, MAX_SIGMA_PX);
        }
    }
    refined
}

/// A line in the image: a point on it, a unit direction, and how well the
/// picture pinned it.
#[derive(Clone, Copy, Debug)]
struct Line {
    point: [f64; 2],
    direction: [f64; 2],
    /// How far the samples scatter around this line, averaged over them, in
    /// pixels. The line is fitted to several measurements of one border, so its
    /// position across the normal is known to the spread divided by the square
    /// root of their number — which is why four edges of a tag are measured
    /// better than one.
    uncertainty: f64,
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
    let mut ramps = Vec::with_capacity(EDGE_SAMPLES);
    for sample in 0..EDGE_SAMPLES {
        let t = EDGE_START + (EDGE_END - EDGE_START) * sample as f64 / (EDGE_SAMPLES - 1) as f64;
        let at = [from[0] + span[0] * t, from[1] + span[1] * t];
        if let Some((offset, ramp)) = border_offset(gray, width, height, at, normal, direction) {
            points.push([at[0] + normal[0] * offset, at[1] + normal[1] * offset]);
            ramps.push(ramp);
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
    let ramp = ramps.iter().sum::<f64>() / ramps.len().max(1) as f64;
    Some(Line {
        point: centroid,
        direction: fitted,
        uncertainty: (spread / count.sqrt())
            .max(RAMP_SHARE * ramp)
            .max(MIN_EDGE_SIGMA_PX),
    })
}

/// How far along `normal` from `at` the strongest border lies, in pixels, and
/// how wide the profile spreads that border over.
///
/// Three parallel profiles across the border are averaged, sampled, and each
/// adjacent pair of samples is a candidate transition. Two decisions are made
/// from them, and they are the whole precision of this function:
///
/// * **which transition is the border.** The one nearest the decoder's own edge,
///   because that edge is good to about a pixel and the tag's next cell is a
///   whole cell away. Taking the steepest in the corridor instead picks the
///   painted line the tag stands on, or the code cell beside it, whenever either
///   is sharper than the border — and it usually is.
/// * **what the two levels are.** The darkest and brightest samples *around that
///   transition only*. Reading them from the whole corridor lets a dark seam or
///   a light cell three pixels away set the halfway level, which moves the
///   crossing along the border's own ramp by a fraction of a pixel, or moves it
///   off the border entirely onto the thing that set the level.
///
/// The crossing of the halfway level between the two samples that bracket it is
/// interpolated linearly, which is what makes this sub-pixel rather than the
/// best quarter-pixel.
fn border_offset(
    gray: &[u8],
    width: usize,
    height: usize,
    at: [f64; 2],
    normal: [f64; 2],
    along: [f64; 2],
) -> Option<(f64, f64)> {
    let steps = (SEARCH_PX / SEARCH_STEP_PX).round() as isize;
    let mut values = Vec::with_capacity((2 * steps + 1) as usize);
    for step in -steps..=steps {
        let offset = step as f64 * SEARCH_STEP_PX;
        let x = at[0] + normal[0] * offset;
        let y = at[1] + normal[1] * offset;
        // The centre profile has to be on the picture, as it always did; a
        // neighbour that runs off the edge is simply not averaged in.
        sample(gray, width, height, x, y)?;
        let mut total = 0.0;
        let mut seen: f64 = 0.0;
        for shear in [-PROFILE_SPACING_PX, 0.0, PROFILE_SPACING_PX] {
            let px = x + along[0] * shear;
            let py = y + along[1] * shear;
            if let Some(value) = sample(gray, width, height, px, py) {
                total += value;
                seen += 1.0;
            }
        }
        values.push(total / seen.max(1.0));
    }

    let step_at = |index: usize| (values[index + 1] - values[index]).abs();
    let offset_of = |index: usize| (index as f64 + 0.5 - steps as f64) * SEARCH_STEP_PX;

    let mut chosen: Option<usize> = None;
    for index in 0..values.len() - 1 {
        if step_at(index) < 1e-9 || offset_of(index).abs() > NEAR_PX {
            continue;
        }
        if chosen.is_none_or(|best| step_at(index) > step_at(best)) {
            chosen = Some(index);
        }
    }
    // A quad more than a pixel and a half out still has a border in the
    // corridor; there it is the sharpest transition, which is where this started.
    let chosen = chosen.or_else(|| {
        (0..values.len() - 1)
            .filter(|index| step_at(*index) > 1e-9)
            .max_by(|a, b| step_at(*a).total_cmp(&step_at(*b)))
    })?;

    // The two levels are the extrema either side of *this* transition, found by
    // walking down its own ramp until it turns. A fixed window would stop inside
    // the ramp of a blurred border and underestimate the contrast, while a
    // corridor-wide extreme would pick up the next cell, or the painted line the
    // tag stands on, and move the level it is measured against.
    let rising = values[chosen + 1] > values[chosen];
    let mut low = chosen;
    while low > 0
        && if rising {
            values[low - 1] < values[low]
        } else {
            values[low - 1] > values[low]
        }
    {
        low -= 1;
    }
    let mut high = chosen + 1;
    while high + 1 < values.len()
        && if rising {
            values[high + 1] > values[high]
        } else {
            values[high + 1] < values[high]
        }
    {
        high += 1;
    }
    let window = &values[low..=high];
    let darkest = window.iter().copied().fold(f64::INFINITY, f64::min);
    let brightest = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if brightest - darkest < MIN_STEP {
        return None;
    }
    let middle_level = 0.5 * (darkest + brightest);

    // Of the pairs that bracket the halfway level — a smeared border brackets it
    // in several — the steepest is the border. Both are inside the window, so
    // neither can be an edge that is not part of this border.
    let mut crossing: Option<(usize, f64)> = None;
    for index in low..high {
        let (a, b) = (values[index], values[index + 1]);
        if (a - middle_level) * (b - middle_level) > 0.0 || (a - b).abs() < 1e-9 {
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
    // of the way to the next one. The ramp is how much of the corridor this one
    // transition took up, which is the width the crossing is bracketed over.
    Some((
        (index as f64 + between - steps as f64) * SEARCH_STEP_PX,
        (high - low) as f64 * SEARCH_STEP_PX,
    ))
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

    /// The detector's quad, `push` pixels outside the true one in every
    /// direction: what the solve gets when the decoder's own edge refinement
    /// lands that far wide.
    fn nudged(push: f64) -> [[f64; 2]; 4] {
        let truth = truth();
        [
            [truth[0][0] + push, truth[0][1] - push],
            [truth[1][0] + push, truth[1][1] + push],
            [truth[2][0] - push, truth[2][1] + push],
            [truth[3][0] - push, truth[3][1] - push],
        ]
    }

    /// The quad a decoder a pixel and a bit wide hands back.
    fn detected() -> [[f64; 2]; 4] {
        nudged(1.2)
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
    fn a_dark_line_just_outside_the_border_does_not_move_the_corner() {
        // A tag rarely arrives alone: the floor it stands on has seams, paint
        // and the tag's own shadow, and one of those can run two pixels outside
        // a border — inside the corridor the border is searched across. Taking
        // the darkest and brightest samples of the whole corridor as the two
        // levels makes that line the reference the border is measured against,
        // and the corner snaps onto the line instead of the border: a couple of
        // pixels, which is most of a small tag's pose.
        let mut picture = square(false);
        for y in 19..22 {
            for x in 20..80 {
                picture[y * SIDE + x] = 10;
            }
        }
        // A decoder good to four tenths of a pixel, which is what the border fit
        // is supposed to improve on rather than lose to.
        let refined = refine(&picture, SIDE, SIDE, &nudged(0.4));
        let error = worst_error(&refined);
        assert!(
            error < 0.35,
            "a line beside the tag moved the corners {error:.2}px: {refined:?}"
        );
    }

    #[test]
    fn a_soft_border_is_reported_as_less_certain_than_a_sharp_one() {
        // What the joint fits are weighted by. A blurred border brackets its
        // halfway level over a wider stretch and its fitted line scatters
        // further from its own samples, so the corner really is less known — and
        // a frame of small soft tags must count for less than a frame whose
        // corners the picture pinned.
        let sharp = refine_with_quality(&square(false), SIDE, SIDE, &detected());
        let soft = refine_with_quality(&square(true), SIDE, SIDE, &detected());
        let sharp_total: f64 = sharp.sigma_px.iter().sum();
        let soft_total: f64 = soft.sigma_px.iter().sum();
        assert!(
            soft_total > sharp_total,
            "a soft border should be less certain: {soft_total:.3} against {sharp_total:.3}"
        );
        assert!(
            sharp
                .sigma_px
                .iter()
                .all(|sigma| *sigma > 0.0 && sigma.is_finite()),
            "every refined corner should carry a finite uncertainty: {:?}",
            sharp.sigma_px
        );
    }

    #[test]
    fn a_corner_that_could_not_be_refined_is_not_claimed_to_be_precise() {
        // A flat picture leaves the decoder's corner where it was; the solve has
        // to be told that corner was never measured, or it will weight it like one
        // that was.
        let flat = vec![128u8; SIDE * SIDE];
        let refined = refine_with_quality(&flat, SIDE, SIDE, &detected());
        assert_eq!(refined.corners, detected());
        assert!(
            refined
                .sigma_px
                .iter()
                .all(|sigma| (*sigma - FALLBACK_SIGMA_PX).abs() < 1e-12),
            "an unmeasured corner must keep the decoder's own precision: {:?}",
            refined.sigma_px
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
            uncertainty: 0.0,
        };
        let b = Line {
            point: [0.0, 5.0],
            direction: [1.0, 0.0],
            uncertainty: 0.0,
        };
        assert_eq!(a.intersect(&b), None);
        let cross = Line {
            point: [3.0, 0.0],
            direction: [0.0, 1.0],
            uncertainty: 0.0,
        };
        let point = a.intersect(&cross).expect("they cross");
        assert!((point[0] - 3.0).abs() < 1e-9 && point[1].abs() < 1e-9);
    }
}

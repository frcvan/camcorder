//! AprilTag 36h11 detection.
//!
//! This is the only file that knows about the vision library: everything else
//! works in terms of [`TagObservation`]. Swapping detectors means replacing the
//! body of [`TagDetector::detect`].

use anyhow::{Context, Result};
use kornia_apriltag::{
    family::{TagFamily, TagFamilyKind},
    AprilTagDecoder, DecodeTagsConfig,
};
use kornia_image::{Image, ImageSize};

use super::types::TagObservation;

/// Detects FRC's tag family in grayscale frames of a fixed size.
pub struct TagDetector {
    decoder: AprilTagDecoder,
    size: ImageSize,
}

impl TagDetector {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])
            .context("configuring the 36h11 family")?;
        let size = ImageSize {
            width: width as usize,
            height: height as usize,
        };
        let decoder = AprilTagDecoder::new(config, size).context("creating the tag decoder")?;
        Ok(TagDetector { decoder, size })
    }

    /// Every tag the decoder finds in one frame, weak ones included.
    ///
    /// `gray` is `width * height` bytes, top row first. Nothing is filtered
    /// here on purpose: how much margin a read deserves depends on how big the
    /// tag is and on whether other frames saw the same id, which is
    /// [`super::trust`]'s business. Filtering in the detector would throw the
    /// far tags away before anything could notice that they are real.
    pub fn detect(&mut self, gray: &[u8]) -> Result<Vec<TagObservation>> {
        let expected = self.size.width * self.size.height;
        if gray.len() != expected {
            anyhow::bail!(
                "frame is {} bytes, expected {expected} for {}x{}",
                gray.len(),
                self.size.width,
                self.size.height
            );
        }
        let image = Image::<u8, 1>::from_size_slice(self.size, gray)
            .context("wrapping the frame for the decoder")?;
        let detections = self
            .decoder
            .decode(&image)
            .context("decoding AprilTags in the frame")?;
        Ok(observations(detections, 1.0))
    }

    /// Every tag the decoder finds in a frame enlarged `factor` times, with the
    /// corners returned in the coordinates of the original frame.
    ///
    /// The detector's own parameters are in pixels, so a tag fifteen pixels
    /// across is at the edge of what it can find at all: its border is under two
    /// pixels, which is barely more than the window the quad detector smooths
    /// with. Sampling the same picture on a finer grid does not add information,
    /// but it does give that window more samples to work across, and a tag this
    /// small is the case that decides whether a pane solves at all. Nothing here
    /// is cheap — the cost is quadratic in `factor` — and nothing pretends
    /// otherwise: this exists because the pipeline is not realtime.
    ///
    /// The corners are on the doubled grid, so they are quantised twice as
    /// finely and then halved: what comes back is a better *seed* for
    /// [`super::corners::refine`] on the original picture, which is where the
    /// sub-pixel position is finally measured.
    pub fn detect_magnified(&mut self, gray: &[u8], factor: usize) -> Result<Vec<TagObservation>> {
        let expected = self.size.width * self.size.height;
        if gray.len() != expected {
            anyhow::bail!(
                "frame is {} bytes, expected {expected} for {}x{}",
                gray.len(),
                self.size.width,
                self.size.height
            );
        }
        // Past this the frame is already large enough that the detector's scale
        // is not what is losing the tag, and the copy would cost a gigabyte for
        // nothing. A 1080p pane doubled is 8MP and fine; 4K is not.
        const MAX_MAGNIFIED_PIXELS: usize = 16_000_000;
        let wide = self.size.width * factor.max(1);
        let tall = self.size.height * factor.max(1);
        if factor <= 1 || wide.saturating_mul(tall) > MAX_MAGNIFIED_PIXELS {
            return self.detect(gray);
        }

        let picture = magnify(gray, self.size.width, self.size.height, factor);
        let size = ImageSize {
            width: wide,
            height: tall,
        };
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])
            .context("configuring the 36h11 family")?;
        let mut decoder = AprilTagDecoder::new(config, size).context("creating the tag decoder")?;
        let image =
            Image::<u8, 1>::from_size_slice(size, &picture).context("wrapping the frame")?;
        let detections = decoder
            .decode(&image)
            .context("decoding AprilTags in the enlarged frame")?;
        Ok(observations(detections, factor as f64))
    }
}

/// Two looks at the same frame, as one list of detections.
///
/// A tag found by both looks is one tag, and the native detection is the one
/// kept: its corners are on the original pixel grid and it is the reading every
/// other stage was checked against. What the enlarged look is for is the tags
/// the native one did not find at all.
///
/// Where the two looks disagree about *which* tag a quad is, the native answer
/// stands. A quad with two ids is a frame to be suspicious of either way, and
/// the pipeline's existing behaviour is the safer of the two to keep.
pub fn merge(native: Vec<TagObservation>, magnified: Vec<TagObservation>) -> Vec<TagObservation> {
    let mut out = native;
    for detection in magnified {
        let (x, y) = centre(&detection);
        let duplicate = out.iter().any(|existing| {
            let (other_x, other_y) = centre(existing);
            // A tag's two looks put its centre in the same place; two different
            // tags of a cluster are a whole tag width apart, which is twice
            // this. So three quarters of the smaller apparent width separates
            // "the same tag seen twice" from "two tags".
            let reach = 0.75 * existing.pixel_width().min(detection.pixel_width());
            (x - other_x).hypot(y - other_y) <= reach
        });
        if !duplicate {
            out.push(detection);
        }
    }
    out
}

fn centre(detection: &TagObservation) -> (f64, f64) {
    let x = detection
        .corners
        .iter()
        .map(|corner| corner[0])
        .sum::<f64>()
        / 4.0;
    let y = detection
        .corners
        .iter()
        .map(|corner| corner[1])
        .sum::<f64>()
        / 4.0;
    (x, y)
}

/// Turn a decoder's detections into observations, dividing the corners by
/// `scale` so they always refer to the original frame.
fn observations(
    detections: Vec<kornia_apriltag::decoder::Detection>,
    scale: f64,
) -> Vec<TagObservation> {
    let mut out = Vec::with_capacity(detections.len());
    for detection in detections {
        out.push(TagObservation {
            id: detection.id,
            corners: detection
                .quad
                .corners
                .map(|corner| [corner.x as f64 / scale, corner.y as f64 / scale]),
            margin: detection.decision_margin,
            hamming: detection.hamming,
            // The border fit fills these in; a frame that never gets that far
            // weighs its corners equally.
            sigma_px: [1.0; 4],
        });
    }
    out
}

/// A larger greyscale copy of a frame, Lanczos-3 in both directions.
///
/// Separable, and the horizontal pass is kept in floating point so the picture
/// is quantised once rather than twice. Bilinear would be half the work and
/// would also have halved the very edges this exists to preserve: a bilinear
/// upscale of a one-pixel border is a ramp, which is the one thing the quad
/// detector needs not to see.
fn magnify(gray: &[u8], width: usize, height: usize, factor: usize) -> Vec<u8> {
    if factor <= 1 || width == 0 || height == 0 || gray.len() < width * height {
        return gray.to_vec();
    }
    let wide = width * factor;
    let tall = height * factor;
    let mut middle = vec![0.0f64; wide * height];
    for y in 0..height {
        for x in 0..wide {
            // Where this output pixel sits in the source, in source samples.
            let source = (x as f64 + 0.5) / factor as f64 - 0.5;
            let base = source.floor() as isize;
            let (mut total, mut weight) = (0.0, 0.0);
            for tap in -2..=3 {
                let at = base + tap;
                let clamped = at.clamp(0, width as isize - 1) as usize;
                let w = lanczos3(source - at as f64);
                total += gray[y * width + clamped] as f64 * w;
                weight += w;
            }
            middle[y * wide + x] = total / weight.max(1e-9);
        }
    }
    let mut out = vec![0u8; wide * tall];
    for y in 0..tall {
        let source = (y as f64 + 0.5) / factor as f64 - 0.5;
        let base = source.floor() as isize;
        for x in 0..wide {
            let (mut total, mut weight) = (0.0, 0.0);
            for tap in -2..=3 {
                let at = base + tap;
                let clamped = at.clamp(0, height as isize - 1) as usize;
                let w = lanczos3(source - at as f64);
                total += middle[clamped * wide + x] * w;
                weight += w;
            }
            out[y * wide + x] = (total / weight.max(1e-9)).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Lanczos-3: `sinc(x) * sinc(x/3)` inside three samples of the centre.
fn lanczos3(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 {
        return 1.0;
    }
    if x >= 3.0 {
        return 0.0;
    }
    let pi_x = std::f64::consts::PI * x;
    (pi_x.sin() / pi_x) * ((pi_x / 3.0).sin() / (pi_x / 3.0))
}

/// One cell of a tag's printed pattern. 1 is the light ink, 0 the dark.
pub const TAG_CELLS: usize = 8;

/// The ink of every tag in FRC's family, so a pose can be asked whether the
/// picture under it has the pattern the tag is actually printed with.
///
/// A pose is fitted to a tag's four *outer* corners, and those four points are
/// the corners of a square — and a square is the same square after a half turn,
/// and after a mirror. So the corners alone admit a second pose that reprojects
/// them just as well but stands on the other side of the tag's plane, looking at
/// the back of the pattern. On Orlando two such poses fitted the same corners
/// 0.42px and 0.46px, nine metres apart on the field. What separates them is not
/// the border, which is square, but the code inside it, which is not: read one
/// way round the ink is where the tag says it is, and the other way round it is
/// mirrored. The decode has already read that code; this hands it back, so
/// evidence the fit threw away can be used against it.
pub struct TagPatterns {
    ink: Vec<[u8; TAG_CELLS * TAG_CELLS]>,
}

impl TagPatterns {
    pub fn new() -> Result<Self> {
        let family = TagFamily::tag36_h11().context("the 36h11 family")?;
        let mut ink = Vec::with_capacity(family.code_data.len());
        for code in &family.code_data {
            let mut cells = [0u8; TAG_CELLS * TAG_CELLS];
            for row in 0..TAG_CELLS {
                for column in 0..TAG_CELLS {
                    // The outermost ring is the black border, dark whatever the
                    // code says. Only the six-by-six inside carries the code.
                    if row == 0 || column == 0 || row + 1 == TAG_CELLS || column + 1 == TAG_CELLS {
                        continue;
                    }
                    for bit in 0..family.nbits {
                        if family.bit_x[bit] as usize == column && family.bit_y[bit] as usize == row
                        {
                            // Most significant bit first, which is how the
                            // family's code words are stored: read the other way
                            // round and the tag decodes as a different one.
                            let value = (code >> (family.nbits - 1 - bit)) & 1;
                            cells[row * TAG_CELLS + column] = value as u8;
                        }
                    }
                }
            }
            ink.push(cells);
        }
        Ok(TagPatterns { ink })
    }

    /// The 8×8 grid of a tag, row-major, as the printed pattern looks from in
    /// front of it: row 0 the top of the picture, column 0 its left.
    pub fn ink(&self, id: u16) -> Option<&[u8; TAG_CELLS * TAG_CELLS]> {
        self.ink.get(id as usize)
    }

    /// How many tag ids the family has a code for.
    pub fn tag_count(&self) -> usize {
        self.ink.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = 128;

    /// A tag as a camera might deliver it: `side` pixels across, on a light
    /// quiet zone, with its eight cells drawn at that size.
    fn small_tag(id: u16, side: usize) -> Vec<u8> {
        let patterns = TagPatterns::new().expect("the family loads");
        let ink = patterns.ink(id).expect("a pattern");
        let mut pixels = vec![200u8; FRAME * FRAME];
        let left = (FRAME - side) / 2;
        let top = (FRAME - side) / 2;
        for y in 0..side {
            for x in 0..side {
                let column = x * TAG_CELLS / side;
                let row = y * TAG_CELLS / side;
                let value = if ink[row * TAG_CELLS + column] == 1 {
                    240
                } else {
                    20
                };
                pixels[(top + y) * FRAME + left + x] = value;
            }
        }
        pixels
    }

    #[test]
    fn a_second_look_at_a_larger_scale_finds_smaller_tags() {
        // The border of a tag fifteen pixels across is under two pixels, which is
        // about the window the quad detector smooths with. The same picture
        // sampled on a finer grid gives that window more to work across, and a
        // tag this size is the case that decides whether a broadcast pane
        // solves at all.
        let mut detector = TagDetector::new(FRAME as u32, FRAME as u32).expect("a detector");
        let mut native = Vec::new();
        let mut magnified = Vec::new();
        for side in [6usize, 8, 10, 12, 14, 16, 20, 24, 32] {
            let picture = small_tag(1, side);
            let plain = detector.detect(&picture).expect("native look");
            let doubled = detector
                .detect_magnified(&picture, 2)
                .expect("magnified look");
            eprintln!(
                "{side:3}px: native {} magnified {}",
                plain.len(),
                doubled.len()
            );
            if !plain.is_empty() {
                native.push(side);
            }
            if !doubled.is_empty() {
                magnified.push(side);
            }
            // Whatever the larger look gains, it must not report the tag at a
            // size or place the original picture does not have: its corners are
            // divided back, so they land in the same frame.
            for detection in &doubled {
                assert!(
                    detection.corners.iter().all(|corner| corner[0] >= 0.0
                        && corner[1] >= 0.0
                        && corner[0] <= FRAME as f64
                        && corner[1] <= FRAME as f64),
                    "a magnified corner came back outside the frame: {:?}",
                    detection.corners
                );
            }
        }
        assert!(
            magnified.len() >= native.len(),
            "the larger look found fewer sizes ({magnified:?}) than the native one ({native:?})"
        );
    }

    #[test]
    fn the_two_looks_at_one_frame_are_merged_into_one_list() {
        // A tag both looks find is one tag, not two: the report counts frames by
        // tags, and a doubled detection would also put eight corners into a
        // solve that has four.
        let picture = small_tag(7, 40);
        let mut detector = TagDetector::new(FRAME as u32, FRAME as u32).expect("a detector");
        let native = detector.detect(&picture).expect("native look");
        let doubled = detector
            .detect_magnified(&picture, 2)
            .expect("magnified look");
        assert_eq!(native.len(), 1, "the fixture should hold one tag");
        assert_eq!(doubled.len(), 1, "the larger look should see the same tag");
        let merged = merge(native.clone(), doubled.clone());
        assert_eq!(merged.len(), 1, "one tag, one detection: {merged:?}");
        assert_eq!(merged[0].id, 7);
        // The native reading is kept: it is the one on the original grid.
        assert_eq!(merged[0].corners, native[0].corners);

        // …and a tag only one of them found is still carried through.
        let only_one = merge(native, Vec::new());
        assert_eq!(only_one.len(), 1);
        assert_eq!(merge(Vec::new(), doubled).len(), 1);
    }

    #[test]
    fn every_cell_of_the_pattern_inside_the_border_carries_a_bit() {
        // The border is a ring of dark cells; the 6x6 inside is the code, and a
        // cell left at its default would silently read as dark ink. A light
        // cell of any tag is therefore proof the bit lookup reached it, and the
        // count of them has to be the code's own popcount.
        let patterns = TagPatterns::new().expect("the family loads");
        assert!(patterns.tag_count() >= 587, "{} tags", patterns.tag_count());
        let family = TagFamily::tag36_h11().expect("the family loads");
        for id in 0..patterns.tag_count() as u16 {
            let ink = patterns.ink(id).expect("every id has a pattern");
            let light = ink.iter().filter(|cell| **cell == 1).count();
            let expected = family.code_data[id as usize].count_ones() as usize;
            assert_eq!(light, expected, "tag {id} has {light} light cells");
        }
    }

    #[test]
    fn the_border_ring_is_dark_and_the_inside_is_not_all_one_colour() {
        let patterns = TagPatterns::new().expect("the family loads");
        for id in [0u16, 7, 26, 586] {
            let ink = patterns.ink(id).expect("a pattern");
            for row in 0..TAG_CELLS {
                for column in 0..TAG_CELLS {
                    let on_border =
                        row == 0 || column == 0 || row + 1 == TAG_CELLS || column + 1 == TAG_CELLS;
                    if on_border {
                        assert_eq!(ink[row * TAG_CELLS + column], 0, "tag {id} border");
                    }
                }
            }
            let inside: Vec<u8> = (1..TAG_CELLS - 1)
                .flat_map(|row| (1..TAG_CELLS - 1).map(move |column| (row, column)))
                .map(|(row, column)| ink[row * TAG_CELLS + column])
                .collect();
            assert_eq!(inside.len(), 36);
            assert!(
                inside.contains(&1) && inside.contains(&0),
                "a pattern with one colour inside says nothing"
            );
        }
    }
}

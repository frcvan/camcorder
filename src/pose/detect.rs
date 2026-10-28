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

        let mut out = Vec::with_capacity(detections.len());
        for detection in detections {
            out.push(TagObservation {
                id: detection.id,
                corners: [
                    [
                        detection.quad.corners[0].x as f64,
                        detection.quad.corners[0].y as f64,
                    ],
                    [
                        detection.quad.corners[1].x as f64,
                        detection.quad.corners[1].y as f64,
                    ],
                    [
                        detection.quad.corners[2].x as f64,
                        detection.quad.corners[2].y as f64,
                    ],
                    [
                        detection.quad.corners[3].x as f64,
                        detection.quad.corners[3].y as f64,
                    ],
                ],
                margin: detection.decision_margin,
                hamming: detection.hamming,
            });
        }
        Ok(out)
    }
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

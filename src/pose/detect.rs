//! AprilTag 36h11 detection.
//!
//! This is the only file that knows about the vision library: everything else
//! works in terms of [`TagObservation`]. Swapping detectors means replacing the
//! body of [`TagDetector::detect`].

use anyhow::{Context, Result};
use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
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

    /// Tags in one frame. `gray` is `width * height` bytes, top row first.
    ///
    /// Detections whose colour margin is below `min_margin` are dropped: on
    /// broadcast video the far tags are a few pixels across, and a marginal
    /// read of those is worse than no read at all.
    pub fn detect(&mut self, gray: &[u8], min_margin: f32) -> Result<Vec<TagObservation>> {
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
            if detection.decision_margin < min_margin {
                continue;
            }
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

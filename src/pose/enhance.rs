//! Making the picture easier for the decoder to read.
//!
//! Broadcast footage is not a printed target: a sub-view can be dim, hazy or
//! badly exposed, the picture has been through a low bitrate, and a tag can be
//! thirty pixels across. What costs the decoder the tag is not the geometry but
//! the contrast it has to work with — a tag whose border is ten luma levels
//! above its quiet zone stops being a tag and becomes noise.
//!
//! Two steps, both in place on the greyscale plane so the detector still reads
//! exactly the frame that was measured:
//!
//! * [`stretch_contrast`] — a percentile stretch, which lifts a dim or hazy
//!   picture so a faint tag gets its contrast back;
//! * [`sharpen`] — an unsharp mask, which puts back the edge that the encoder
//!   and a soft lens took off the border.
//!
//! The stretch is one map for the whole frame, not one per region, and that is
//! deliberate. Block-wise equalisation looks like the obvious answer to a tag
//! under a shadow, and on a real faded fixture it *lost* a tag the raw frame
//! still decoded: two neighbouring blocks landed on different maps, the
//! background was pinned to black in one of them, and the tag's black border
//! ended up with no light quiet zone to stand against. The decoder thresholds
//! adaptively already, so what it needs is a frame whose contrast ratios are
//! intact and whose range uses the scale — not one whose local statistics have
//! been rewritten.
//!
//! Both steps are bounded. A stretch that amplified a flat, noisy patch would
//! hand the quad detector a field of speckle, which costs more tags than the
//! exposure did, so the gain is capped and the unsharp push is limited.

/// The dark and bright ends of the picture, as percentiles, which the stretch
/// spreads apart. Percentiles rather than the extremes: one bright light or one
/// black corner should not set the scale for the whole frame.
const LOW_PERCENTILE: f64 = 0.01;
const HIGH_PERCENTILE: f64 = 0.99;
/// The most the picture may be stretched. A nearly flat frame holds mostly
/// noise, and amplifying that is how a stretch turns the carpet into a field of
/// false quads.
const MAX_GAIN: f64 = 4.0;
/// Below this spread there is no picture to stretch, only noise. It has to stay
/// under the contrast of a tag worth rescuing — a faint one is about 28 levels —
/// or the stretch would skip exactly the frames it is here for.
const MIN_SPREAD: f64 = 24.0;
/// How much of the unsharp difference is added back, and how far one pixel may
/// be pushed by it — the limit is what keeps a sharp edge from ringing.
const SHARPEN_AMOUNT: f64 = 0.6;
const SHARPEN_LIMIT: f64 = 32.0;

/// Prepare one greyscale frame for the detector, in place.
///
/// `gray` is `width * height` bytes, top row first. A buffer that is too small
/// for the size given is left untouched rather than being read out of bounds.
pub fn enhance(gray: &mut [u8], width: usize, height: usize) {
    if width == 0 || height == 0 || gray.len() < width * height {
        return;
    }
    stretch_contrast(gray, width, height);
    sharpen(gray, width, height);
}

/// Spread the picture over the range it actually uses.
///
/// Centred on the middle of that range rather than pinned to black and white,
/// so a washed-out frame keeps its shape instead of saturating whichever end it
/// happened to be closer to — a tag's quiet zone sitting on white would lose
/// exactly the border it needs.
fn stretch_contrast(gray: &mut [u8], width: usize, height: usize) {
    let mut histogram = [0u32; 256];
    for value in gray.iter().take(width * height) {
        histogram[*value as usize] += 1;
    }
    let total: u32 = histogram.iter().sum();
    if total == 0 {
        return;
    }
    let low = percentile(&histogram, total, LOW_PERCENTILE) as f64;
    let high = percentile(&histogram, total, HIGH_PERCENTILE) as f64;
    let spread = high - low;
    if spread < MIN_SPREAD {
        return;
    }
    let gain = (255.0 / spread).min(MAX_GAIN);
    if gain <= 1.0 {
        return;
    }
    let middle = (low + high) * 0.5;
    for value in gray.iter_mut().take(width * height) {
        *value = clamp_u8(128.0 + gain * (*value as f64 - middle));
    }
}

/// The value at a cumulative fraction of a 256-bin histogram.
fn percentile(histogram: &[u32; 256], total: u32, fraction: f64) -> u8 {
    let target = (total as f64 * fraction) as u32;
    let mut seen = 0u32;
    for (value, count) in histogram.iter().enumerate() {
        seen += count;
        if seen > target {
            return value as u8;
        }
    }
    255
}

/// Unsharp mask: add back a bounded share of what a 3x3 blur removed.
fn sharpen(gray: &mut [u8], width: usize, height: usize) {
    let blurred = blur3x3(gray, width, height);
    for (pixel, soft) in gray.iter_mut().zip(blurred.iter()) {
        let difference = (*pixel as f64 - *soft as f64).clamp(-SHARPEN_LIMIT, SHARPEN_LIMIT);
        *pixel = clamp_u8(*pixel as f64 + SHARPEN_AMOUNT * difference);
    }
}

/// 3x3 binomial blur with the edges clamped, which leaves the picture the same
/// size and does not darken the border the way a zero-padded blur would.
fn blur3x3(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    let weights = [[1u32, 2, 1], [2, 4, 2], [1, 2, 1]];
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u32;
            for (dy, row) in weights.iter().enumerate() {
                let sy = (y + dy).saturating_sub(1).min(height - 1);
                for (dx, weight) in row.iter().enumerate() {
                    let sx = (x + dx).saturating_sub(1).min(width - 1);
                    sum += gray[sy * width + sx] as u32 * weight;
                }
            }
            out[y * width + x] = (sum / 16) as u8;
        }
    }
    out
}

fn clamp_u8(value: f64) -> u8 {
    value.clamp(0.0, 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: usize = 128;

    fn clamp(value: f64) -> u8 {
        value.clamp(0.0, 255.0).round() as u8
    }

    /// A tag as the decoder sees it: a light quiet zone, a dark border, and a
    /// six-by-six code of dark and light cells inside it. `ink` is the
    /// (dark, bright) pair, so a faint tag is the same geometry with no
    /// contrast.
    fn tag_picture(background: u8, ink: (u8, u8)) -> Vec<u8> {
        const UNIT: usize = 8;
        const QUIET: usize = 2;
        const BORDER: usize = 8;
        let mut pixels = vec![background; SIDE * SIDE];
        let tag = (BORDER + 2 * QUIET) * UNIT;
        let left = (SIDE - tag) / 2;
        let top = (SIDE - tag) / 2;
        for y in 0..tag {
            for x in 0..tag {
                pixels[(top + y) * SIDE + left + x] = ink.1;
            }
        }
        for y in 0..BORDER {
            for x in 0..BORDER {
                let edge = x == 0 || y == 0 || x == BORDER - 1 || y == BORDER - 1;
                // A checker stands in for the code: what matters is that dark
                // and light cells alternate at the same contrast as the border.
                let value = if edge || (x + y) % 2 == 0 {
                    ink.0
                } else {
                    ink.1
                };
                for py in 0..UNIT {
                    for px in 0..UNIT {
                        let ix = left + (QUIET + x) * UNIT + px;
                        let iy = top + (QUIET + y) * UNIT + py;
                        pixels[iy * SIDE + ix] = value;
                    }
                }
            }
        }
        pixels
    }

    /// Spread between the darkest and brightest pixel of a square region.
    fn spread(pixels: &[u8], x0: usize, x1: usize, y0: usize, y1: usize) -> f64 {
        let mut low = 255u8;
        let mut high = 0u8;
        for y in y0..y1 {
            for x in x0..x1 {
                let value = pixels[y * SIDE + x];
                low = low.min(value);
                high = high.max(value);
            }
        }
        (high - low) as f64
    }

    #[test]
    fn a_washed_out_picture_gets_its_contrast_back() {
        let mut pixels = tag_picture(110, (100, 128));
        let darkest_before = *pixels.iter().min().expect("darkest");
        let brightest_before = *pixels.iter().max().expect("brightest");
        assert!(
            (brightest_before - darkest_before) < 40,
            "the fixture should be washed out, {darkest_before}..{brightest_before}"
        );

        enhance(&mut pixels, SIDE, SIDE);

        let darkest = *pixels.iter().min().expect("darkest");
        let brightest = *pixels.iter().max().expect("brightest");
        assert!(
            (brightest - darkest) as f64 > (brightest_before - darkest_before) as f64 * 3.0,
            "contrast only went {darkest_before}..{brightest_before} -> {darkest}..{brightest}"
        );
    }

    #[test]
    fn a_faint_tags_border_still_stands_against_its_quiet_zone() {
        // The failure this pins down: equalising per region drove the
        // background down to the border's own level, so the border stopped
        // being a closed dark loop and the quad detector lost the tag. A tag is
        // only a tag while its dark cells are clearly darker than everything
        // around them.
        let mut pixels = tag_picture(110, (100, 138));
        enhance(&mut pixels, SIDE, SIDE);

        // The code's six-by-six cells, split by their parity: the mean of the
        // dark half against the mean of the light half is the contrast the
        // decoder reads the bits from.
        let code_start = 40; // (quiet 2 + border 1) cells of 8px, inset 16px
        let (mut dark, mut light) = ((0.0, 0usize), (0.0, 0usize));
        for cell_y in 0..6 {
            for cell_x in 0..6 {
                let value = pixels
                    [(code_start + cell_y * 8 + 4) * SIDE + code_start + cell_x * 8 + 4]
                    as f64;
                if (cell_x + cell_y) % 2 == 0 {
                    dark = (dark.0 + value, dark.1 + 1);
                } else {
                    light = (light.0 + value, light.1 + 1);
                }
            }
        }
        let dark = dark.0 / dark.1 as f64;
        let light = light.0 / light.1 as f64;
        let background = pixels[8 * SIDE + 8] as f64;
        assert!(
            light - dark > 80.0,
            "a code cell pair is only {dark}..{light} apart"
        );
        assert!(
            background - dark > 20.0,
            "the background ({background}) has been pulled down to the border ({dark})"
        );
    }

    #[test]
    fn a_flat_patch_is_not_turned_into_speckle() {
        // A nearly flat patch holds noise, not picture. Stretching it would
        // hand the quad detector a field of spots at its own contrast.
        let mut noise = vec![0u8; SIDE * SIDE];
        let mut seed = 12345u32;
        for pixel in noise.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *pixel = 128u8.wrapping_add((seed >> 24) as u8 % 7).wrapping_sub(3);
        }
        let before = spread(&noise, 0, SIDE, 0, SIDE);
        enhance(&mut noise, SIDE, SIDE);
        let after = spread(&noise, 0, SIDE, 0, SIDE);
        assert!(
            after <= before + 8.0,
            "noise spread grew from {before} to {after}; a flat patch was stretched"
        );
    }

    #[test]
    fn a_clean_high_contrast_frame_keeps_black_black_and_white_white() {
        let mut pixels = vec![0u8; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                let checker = ((x / 8) + (y / 8)) % 2 == 0;
                pixels[y * SIDE + x] = if checker { 255 } else { 0 };
            }
        }
        let mut enhanced = pixels.clone();
        enhance(&mut enhanced, SIDE, SIDE);

        assert!(
            enhanced.iter().copied().min().expect("darkest") < 40,
            "black was lifted: darkest {}",
            enhanced.iter().copied().min().expect("darkest")
        );
        assert!(
            enhanced.iter().copied().max().expect("brightest") > 215,
            "white was pulled down: brightest {}",
            enhanced.iter().copied().max().expect("brightest")
        );
        // The polarity of the picture must survive: the darkest pixel of the
        // original is still darker than the brightest one afterwards.
        let darkest_before = (0..SIDE * SIDE)
            .min_by_key(|i| pixels[*i])
            .expect("darkest");
        let brightest_before = (0..SIDE * SIDE)
            .max_by_key(|i| pixels[*i])
            .expect("brightest");
        assert!(enhanced[darkest_before] < enhanced[brightest_before]);
        // A picture that already uses the whole range is not stretched at all,
        // so away from the edges it must come through untouched.
        assert_eq!(enhanced[0], pixels[0], "a full-range frame was moved");
    }

    #[test]
    fn sizes_that_cannot_be_enhanced_are_left_alone() {
        let mut tiny = vec![7u8];
        enhance(&mut tiny, 1, 1);
        assert_eq!(tiny, vec![7]);

        let mut odd = vec![10u8, 20, 30, 40, 50, 60];
        enhance(&mut odd, 3, 2);
        assert_eq!(odd.len(), 6);

        let mut short = vec![1u8, 2, 3];
        enhance(&mut short, 8, 8);
        assert_eq!(
            short,
            vec![1, 2, 3],
            "a short buffer must not be written past its end"
        );
    }

    #[test]
    fn values_stay_inside_a_byte() {
        assert_eq!(clamp(-30.0), 0);
        assert_eq!(clamp(300.0), 255);
        assert_eq!(clamp(127.4), 127);
    }
}

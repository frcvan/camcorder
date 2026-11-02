//! A scoreboard for the tag reader: recall and corner error across a ladder of
//! sizes and degradations, so a change to the detector, the enlargement or the
//! enhancement can be judged by numbers rather than by argument.
//!
//! The pictures are synthetic and their truth is known exactly: a tag drawn at a
//! stated pixel size with 4×4 supersampled edges, so its corners are where the
//! fixture says and a corner error is a real distance rather than a comparison
//! with another detector. That is also the limit of what this can say — it holds
//! a *regression* still, it does not stand in for footage. What the footage does
//! is in `pose_pipeline.rs`, which builds real h264 clips with `ffmpeg`.
//!
//! Every row prints before anything is asserted, so a failure shows the whole
//! table rather than stopping at the first size. The floors below were read off
//! this table, not chosen: each one sits just under what the reader measured, and
//! the doc comment on each row says what that was.

use camcorder::pose::{
    corners,
    detect::{merge, TagDetector, TagPatterns},
    enhance,
};

/// Big enough that the quiet zone and the floor around the tag are both in the
/// picture at the largest size measured.
const FRAME: usize = 192;
/// A tag id that is in the family and in both the 2025 and 2026 layouts.
const TAG: u16 = 27;
/// The light quiet zone and the darker floor around it.
const QUIET: u8 = 200;
const FLOOR: u8 = 140;
const INK_LIGHT: u8 = 235;
const INK_DARK: u8 = 25;

/// What one size under one degradation came out as.
struct Row {
    label: &'static str,
    side: f64,
    found: usize,
    error: f64,
    /// The corner error this row may have, pixels.
    allowed: f64,
}

/// A tag `side` pixels across, centred, with anti-aliased edges, and the truth
/// its four corners stand for in the detector's order: top-right, bottom-right,
/// bottom-left, top-left.
fn render(side: f64) -> (Vec<u8>, [[f64; 2]; 4]) {
    let patterns = TagPatterns::new().expect("the family loads");
    let ink = patterns.ink(TAG).expect("a pattern");
    let cell = side / 8.0;
    let left = (FRAME as f64 - side) * 0.5;
    let top = (FRAME as f64 - side) * 0.5;
    let quiet = (side / 8.0).ceil().max(6.0);

    let mut pixels = vec![FLOOR; FRAME * FRAME];
    for y in 0..FRAME {
        for x in 0..FRAME {
            // Four samples per pixel each way: a border at 30.4 really is 40%
            // into the pixel, and the truth below is worth something.
            let mut total = 0.0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let px = x as f64 + (sx as f64 + 0.5) / 4.0;
                    let py = y as f64 + (sy as f64 + 0.5) / 4.0;
                    let gx = (px - left) / cell;
                    let gy = (py - top) / cell;
                    let value = if (0.0..8.0).contains(&gx) && (0.0..8.0).contains(&gy) {
                        let column = gx.floor() as usize;
                        let row = gy.floor() as usize;
                        if ink[row * 8 + column] == 1 {
                            INK_LIGHT
                        } else {
                            INK_DARK
                        }
                    } else if px >= left - quiet
                        && px <= left + side + quiet
                        && py >= top - quiet
                        && py <= top + side + quiet
                    {
                        QUIET
                    } else {
                        FLOOR
                    };
                    total += value as f64;
                }
            }
            pixels[y * FRAME + x] = (total / 16.0).round() as u8;
        }
    }

    // The printed square runs from `left` to `left + side` in continuous
    // coordinates, and a pixel index is centred half a pixel into its pixel.
    let (l, t) = (left - 0.5, top - 0.5);
    let (r, b) = (left + side - 0.5, top + side - 0.5);
    (pixels, [[r, t], [r, b], [l, b], [l, t]])
}

/// A tag softened the way a lens and a codec leave it: one binomial blur per
/// pass, which spreads a border over a pixel.
fn blur(pixels: &mut [u8], passes: usize) {
    const WEIGHTS: [[u32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
    for _ in 0..passes {
        let source = pixels.to_vec();
        for y in 0..FRAME {
            for x in 0..FRAME {
                let mut sum = 0u32;
                for (dy, row) in WEIGHTS.iter().enumerate() {
                    let sy = (y + dy).saturating_sub(1).min(FRAME - 1);
                    for (dx, weight) in row.iter().enumerate() {
                        let sx = (x + dx).saturating_sub(1).min(FRAME - 1);
                        sum += source[sy * FRAME + sx] as u32 * weight;
                    }
                }
                pixels[y * FRAME + x] = (sum / 16) as u8;
            }
        }
    }
}

/// Deterministic per-pixel noise, which is what a low-bitrate dim sub-view has
/// and what makes a low-contrast tag hard: the levels are there, the samples are
/// not reliable.
fn noise(pixels: &mut [u8], sigma: f64) {
    let mut seed = 0x9E37_79B9u32;
    for value in pixels.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let unit = (seed >> 8) as f64 / (1u32 << 24) as f64 - 0.5;
        // Uniform on ±sqrt(3)σ has standard deviation σ.
        *value = (*value as f64 + unit * 2.0 * sigma * 3f64.sqrt())
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

/// A dim, hazy sub-view: whatever contrast the picture had, squeezed into
/// `levels` luma levels around grey.
fn faint(pixels: &mut [u8], levels: f64, original: f64) {
    let share = levels / original;
    for value in pixels.iter_mut() {
        *value = (128.0 + (*value as f64 - 128.0) * share)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

/// Read one picture exactly as the pipeline does — enhance, both looks, merge,
/// border fit — and report how many times the right id came back and how far the
/// worst corner was from where the fixture put it.
fn read(picture: &mut [u8], truth: &[[f64; 2]; 4]) -> (usize, f64) {
    enhance::enhance(picture, FRAME, FRAME);
    let mut detector = TagDetector::new(FRAME as u32, FRAME as u32).expect("a detector");
    let native = detector.detect(picture).expect("the native look");
    let magnified = detector
        .detect_magnified(picture, 2)
        .expect("the enlarged look");
    let merged = merge(native, magnified);

    let mut right = 0usize;
    let mut worst: f64 = 0.0;
    for detection in &merged {
        if detection.id != TAG {
            continue;
        }
        right += 1;
        let refined = corners::refine_with_quality(picture, FRAME, FRAME, &detection.corners);
        for corner in refined.corners {
            // Matched to the nearest true corner rather than by index: a detector
            // is free to start its quad anywhere, and this measures placement.
            let nearest = truth
                .iter()
                .map(|true_corner| (corner[0] - true_corner[0]).hypot(corner[1] - true_corner[1]))
                .fold(f64::INFINITY, f64::min);
            worst = worst.max(nearest);
        }
    }
    (right, worst)
}

fn measure(
    rows: &mut Vec<Row>,
    label: &'static str,
    side: f64,
    allowed: f64,
    picture: Vec<u8>,
    truth: &[[f64; 2]; 4],
) {
    let mut picture = picture;
    let (found, error) = read(&mut picture, truth);
    rows.push(Row {
        label,
        side,
        found,
        error,
        allowed,
    });
}

/// Print the whole table, then fail on the rows that are out of bounds. Printing
/// first is the point: a regression at one size should show what the reader did
/// at the others.
fn report(rows: &[Row]) {
    for row in rows {
        eprintln!(
            "  {:<8} {:>4.0}px: {} detection(s), corner error {:.2}px (allowed {:.2})",
            row.label, row.side, row.found, row.error, row.allowed
        );
    }
    let wrong = rows
        .iter()
        .filter(|row| row.found != 1 || row.error > row.allowed)
        .map(|row| {
            format!(
                "{} at {:.0}px: {} detection(s), {:.2}px out",
                row.label, row.side, row.found, row.error
            )
        })
        .collect::<Vec<_>>();
    assert!(
        wrong.is_empty(),
        "the reader fell short:\n  {}",
        wrong.join("\n  ")
    );
}

#[test]
fn the_reader_finds_a_tag_at_every_size_a_pane_holds() {
    // Clean synthetic tags are the easy end, and the point of the row is the
    // corner, not the recall: measured here at 0.01px across 16-48px, so a
    // quarter of a pixel is a regression rather than a tolerance.
    let mut rows = Vec::new();
    for side in [16.0, 20.0, 24.0, 32.0, 48.0] {
        let (picture, truth) = render(side);
        measure(&mut rows, "clean", side, 0.25, picture, &truth);
    }
    report(&rows);
}

#[test]
fn a_softened_tag_is_still_read_where_the_border_is_a_smear() {
    // Broadcast footage is never sharp, and the blur is what costs the small
    // tags: measured here, a 16px tag under one pass is *not* read, and the
    // ladder starts where it is — 20px for one pass, 24px for two. That limit is
    // what a pane holding only fifteen-pixel tags runs into, and it is why the
    // report says a pane with one small tag is not pinned.
    // Each allowance is the error this row measures plus about a quarter, so the
    // row holds a regression still rather than describing an ideal nobody reaches:
    // a 20px tag blurred once lands 1.21px out, a 24px tag blurred twice 0.62px,
    // and everything from 32px up is within 0.13px.
    let mut rows = Vec::new();
    for (passes, sizes) in [
        (1usize, [(20.0, 1.5), (32.0, 0.25), (48.0, 0.25)]),
        (2, [(24.0, 1.0), (32.0, 0.35), (48.0, 0.25)]),
    ] {
        for (side, allowed) in sizes {
            let (mut picture, truth) = render(side);
            blur(&mut picture, passes);
            measure(
                &mut rows,
                if passes == 1 { "blur1" } else { "blur2" },
                side,
                allowed,
                picture,
                &truth,
            );
        }
    }
    report(&rows);
}

#[test]
fn a_dark_low_contrast_sub_view_still_reads() {
    // A tag 30 luma levels above the floor, dim and noisy — a sub-view at the end
    // of a long lens. Measured here, the read survives down to 16px but the
    // *corners* are coarser: a border quantised into 30 levels and carrying a
    // couple of levels of noise is bracketed over a wider stretch at every
    // crossing, so this row's allowance is looser than the sharp rows' and is
    // stated as such.
    //
    // Whether this picture *needs* the enhancement is not claimed here. That the
    // enhancement reads a tag the raw frame does not is pinned on real h264
    // footage by the faint fixture in `pose_pipeline.rs`.
    let mut rows = Vec::new();
    for side in [16.0, 24.0, 32.0] {
        let (mut picture, truth) = render(side);
        let original = (INK_LIGHT - INK_DARK) as f64;
        faint(&mut picture, 30.0, original);
        noise(&mut picture, 2.5);

        // Measured at 1.82px on the 16px tag and better from 24px up: a border
        // quantised into thirty levels is coarser than a clean one at every
        // crossing, and the allowance says so rather than pretending otherwise.
        measure(&mut rows, "dim", side, 2.5, picture, &truth);
    }
    report(&rows);
}

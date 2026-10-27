//! Which side of the field a frame is looking at.
//!
//! Two readings, kept apart so a run can say when they disagree instead of
//! quietly averaging them:
//!
//! * **Colour.** FRC's field elements, bumpers and alliance wall graphics are
//!   red at one end of the field and blue at the other, so a picture full of red
//!   was taken at the red alliance's end. This reads the surroundings, so it
//!   still answers for a frame whose tags could not be solved — but it is a
//!   reading of whatever is in shot, and a red robot parked at the blue end
//!   fools it.
//! * **Geometry.** WPILib's field frame has its origin at the blue alliance
//!   driver station with `+x` along the field toward the red one ("+X is toward
//!   the red alliance station", 2026 field drawings), so a tag at a small `x`
//!   stands at the blue end and a tag at a large `x` at the red end. The tags
//!   read in the frame, and the solved camera position, each give an end.
//!
//! Neither is authoritative, which is the point: colour says which end the
//! *picture* is from, geometry says which end the *solve* is at, and a frame
//! where those two differ is a frame to look at rather than trust. The overview
//! pane of a broadcast is skipped entirely — one wide shot from the end of the
//! field holds both alliances, so neither reading means anything there.

use std::path::Path;

use serde::Serialize;

use super::{layout::TagLayout, types::CameraPose};

/// Sampled pixels the smaller colour count needs before a frame is called.
const MIN_COLOURED: usize = 200;
/// How far one colour has to lead the other before the frame is called for it.
const DOMINANCE: f64 = 1.3;
/// Rough number of pixels to look at, whatever the frame size.
const COLOUR_SAMPLES: usize = 40_000;
/// Ignore dark and washed-out pixels: only a real chromatic pixel votes.
const MIN_VALUE: f32 = 45.0;
const MIN_CHROMA: f32 = 45.0;

/// The alliance whose colours surround the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Alliance {
    Red,
    Blue,
    /// Both colours are present in about equal measure.
    Mixed,
    /// Not enough coloured picture to say.
    Unknown,
}

/// Which half of the field's length something stands in. The low-`x` half is
/// the blue alliance's and the high-`x` half the red alliance's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum End {
    LowX,
    HighX,
    /// Tags from both ends, or the whole field, in one frame.
    Both,
    Unknown,
}

/// What was read about the side of the field in one frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SideVerdict {
    /// The alliance whose colours dominate the frame.
    pub alliance: Alliance,
    /// How many sampled pixels counted as red and as blue, out of the frame.
    pub red_pixels: usize,
    pub blue_pixels: usize,
    /// Which end of the field the tags that survived the solve stand at.
    pub tags_end: End,
    /// Which end the solved camera position falls in, when the frame solved.
    pub camera_end: Option<End>,
    /// Set when colour and geometry claim different ends, saying what each one
    /// read. A frame that disagrees is not necessarily wrong — a red banner
    /// over the blue end, or a pose scaled by the assumed `--fov` — but it is
    /// the frame to check first.
    pub disagreement: Option<String>,
}

/// Which pane of the broadcast frame a file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pane {
    /// The wide full-width pane at the top of the frame, or the whole frame
    /// when nothing was tiled. It holds both alliances, so the side of the
    /// field is not a question it can answer.
    Overview,
    /// One of the sub-views: a camera at one end of the field.
    View,
}

impl Pane {
    /// Read the pane out of the name the splitter gave the file.
    ///
    /// `split_views` writes the top pane of each segment as `segNN_view1.mp4`
    /// and the rest as `segNN_viewN.mp4`, and in the broadcast layouts the top
    /// pane is the wide overview. A hand-cut clip, or a differently named file,
    /// is a view. Rename a `view1` that is really a sub-view and the side of the
    /// field is read from it again.
    pub fn of(video: &Path) -> Pane {
        let stem = video
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default();
        let Some((segment, view)) = stem.rsplit_once('_') else {
            return Pane::View;
        };
        let numbered = segment
            .strip_prefix("seg")
            .is_some_and(|number| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()));
        if numbered && view.eq_ignore_ascii_case("view1") {
            Pane::Overview
        } else {
            Pane::View
        }
    }
}

/// The end of the field a colour reading names, when it names one.
///
/// This is what the solver is given before it starts: with red around the tag,
/// the camera should come out at the red end. `None` means the picture does not
/// say — both alliances in shot, or too little colour to count — and the solve
/// is left to its own devices.
pub fn expected_end(alliance: Alliance) -> Option<End> {
    match alliance {
        Alliance::Red => Some(End::HighX),
        Alliance::Blue => Some(End::LowX),
        Alliance::Mixed | Alliance::Unknown => None,
    }
}

/// Read both sides of the field from one frame and say whether they agree.
///
/// `rgb` is `width * height * 3` bytes, top row first. `tag_ids` are the tags
/// that survived the solve, so a mis-read tag does not get a vote on the end.
/// `camera` is the solved pose when there is one.
pub fn analyse(
    rgb: &[u8],
    width: u32,
    height: u32,
    tag_ids: &[u16],
    layout: &TagLayout,
    camera: Option<&CameraPose>,
) -> SideVerdict {
    let (red, blue) = count_colours(rgb, width, height);
    verdict(alliance_from(red, blue), red, blue, tag_ids, layout, camera)
}

/// The verdict, from colour counts that have already been taken.
///
/// Split from [`analyse`] because the counts are needed *before* the solve — the
/// colours are what the solver is told to prefer — and recounting the frame
/// afterwards would be the same answer computed twice.
pub fn verdict(
    alliance: Alliance,
    red_pixels: usize,
    blue_pixels: usize,
    tag_ids: &[u16],
    layout: &TagLayout,
    camera: Option<&CameraPose>,
) -> SideVerdict {
    let tags_end = end_of_tags(tag_ids, layout);
    let camera_end = camera.map(|pose| end_of_x(pose.position.x, layout.field_length));
    let disagreement = cross_check(alliance, tags_end, camera_end);
    SideVerdict {
        alliance,
        red_pixels,
        blue_pixels,
        tags_end,
        camera_end,
        disagreement,
    }
}

/// Count the clearly red and clearly blue pixels of a frame.
///
/// Only pixels with real colour and real brightness vote: the grey of the
/// carpet, the white of the field lines and the black of a robot are all
/// colourless, and letting them into the count would let a big neutral field
/// outvote the alliance elements around it.
pub fn count_colours(rgb: &[u8], width: u32, height: u32) -> (usize, usize) {
    let pixels = (width as usize) * (height as usize);
    if pixels == 0 || rgb.len() < pixels * 3 {
        return (0, 0);
    }
    let stride = (pixels / COLOUR_SAMPLES).max(1);
    let mut red = 0usize;
    let mut blue = 0usize;
    for index in (0..pixels).step_by(stride) {
        let base = index * 3;
        match classify(rgb[base], rgb[base + 1], rgb[base + 2]) {
            Pixel::Red => red += 1,
            Pixel::Blue => blue += 1,
            Pixel::Neither => {}
        }
    }
    (red, blue)
}

/// Which alliance the colour counts describe.
pub fn alliance_from(red: usize, blue: usize) -> Alliance {
    if red + blue < MIN_COLOURED {
        return Alliance::Unknown;
    }
    let (red, blue) = (red as f64, blue as f64);
    if red > blue * DOMINANCE {
        Alliance::Red
    } else if blue > red * DOMINANCE {
        Alliance::Blue
    } else {
        Alliance::Mixed
    }
}

/// Which end of the field a tag stands at, from the layout.
pub fn end_of_x(x: f64, field_length: f64) -> End {
    if x < field_length * 0.5 {
        End::LowX
    } else {
        End::HighX
    }
}

/// Which end of the field the given tags stand at.
///
/// Tags that are not in the layout cannot place anything, so they are ignored;
/// no usable tag at all is [`End::Unknown`] rather than a guess.
pub fn end_of_tags(ids: &[u16], layout: &TagLayout) -> End {
    let mut low = false;
    let mut high = false;
    for id in ids {
        let Some(tag) = layout.tag(*id) else {
            continue;
        };
        match end_of_x(tag.translation.x, layout.field_length) {
            End::LowX => low = true,
            End::HighX => high = true,
            _ => {}
        }
    }
    match (low, high) {
        (true, true) => End::Both,
        (true, false) => End::LowX,
        (false, true) => End::HighX,
        (false, false) => End::Unknown,
    }
}

/// What colour and geometry each claim, when they claim different ends.
fn cross_check(alliance: Alliance, tags_end: End, camera_end: Option<End>) -> Option<String> {
    let expected = match alliance {
        Alliance::Red => End::HighX,
        Alliance::Blue => End::LowX,
        Alliance::Mixed | Alliance::Unknown => return None,
    };
    let mut problems = Vec::new();
    let decisive = |end: End| matches!(end, End::LowX | End::HighX);
    if decisive(tags_end) && tags_end != expected {
        problems.push(format!(
            "the tags read stand at the {} end",
            describe(tags_end)
        ));
    } else if let Some(end) = camera_end {
        if decisive(end) && end != expected {
            problems.push(format!(
                "the solved pose puts the camera at the {} end",
                describe(end)
            ));
        }
    }
    if problems.is_empty() {
        return None;
    }
    Some(format!(
        "the picture is {} but {}",
        describe_alliance(alliance),
        problems.join(", and ")
    ))
}

fn describe(end: End) -> &'static str {
    match end {
        End::LowX => "blue (low-x)",
        End::HighX => "red (high-x)",
        End::Both => "both",
        End::Unknown => "unknown",
    }
}

fn describe_alliance(alliance: Alliance) -> &'static str {
    match alliance {
        Alliance::Red => "red",
        Alliance::Blue => "blue",
        Alliance::Mixed => "both red and blue",
        Alliance::Unknown => "unreadable",
    }
}

enum Pixel {
    Red,
    Blue,
    Neither,
}

/// Red, blue or neither, by hue. The hue bands are wide enough to hold FRC's
/// own red and blue through a video encoder and a colour conversion, and the
/// chroma floor keeps anything near grey out of the count.
fn classify(r: u8, g: u8, b: u8) -> Pixel {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let chroma = max - min;
    if max < MIN_VALUE || chroma < MIN_CHROMA {
        return Pixel::Neither;
    }
    let hue = if max == r {
        60.0 * (((g - b) / chroma) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / chroma + 2.0)
    } else {
        60.0 * ((r - g) / chroma + 4.0)
    };
    let hue = (hue + 360.0) % 360.0;
    if hue <= 25.0 || hue >= 335.0 {
        Pixel::Red
    } else if (200.0..=260.0).contains(&hue) {
        Pixel::Blue
    } else {
        Pixel::Neither
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::types::Vec3;

    fn solid(width: u32, height: u32, colour: (u8, u8, u8)) -> Vec<u8> {
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..width * height {
            rgb.extend_from_slice(&[colour.0, colour.1, colour.2]);
        }
        rgb
    }

    /// Half the frame one colour, half the other, split down the middle.
    fn split(width: u32, height: u32, left: (u8, u8, u8), right: (u8, u8, u8)) -> Vec<u8> {
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..height {
            for x in 0..width {
                let colour = if x < width / 2 { left } else { right };
                rgb.extend_from_slice(&[colour.0, colour.1, colour.2]);
            }
        }
        rgb
    }

    fn layout() -> TagLayout {
        TagLayout::load("2026-rebuilt-welded").expect("bundled layout")
    }

    #[test]
    fn red_elements_around_the_camera_means_the_red_end() {
        // FRC's own red and blue, as they come out of a broadcast.
        let red = count_colours(&solid(64, 64, (220, 30, 40)), 64, 64);
        let blue = count_colours(&solid(64, 64, (20, 90, 170)), 64, 64);
        assert!(red.0 > 3000 && red.1 == 0, "red frame: {red:?}");
        assert_eq!(alliance_from(red.0, red.1), Alliance::Red);
        assert_eq!(alliance_from(blue.0, blue.1), Alliance::Blue);
    }

    #[test]
    fn the_grey_carpet_and_the_white_lines_are_not_alliances() {
        // A field with no alliance elements in shot must not be called red or
        // blue on the strength of its flooring.
        assert_eq!(
            count_colours(&solid(64, 64, (110, 112, 120)), 64, 64),
            (0, 0)
        );
        assert_eq!(
            count_colours(&solid(64, 64, (240, 240, 235)), 64, 64),
            (0, 0)
        );
        assert_eq!(count_colours(&solid(64, 64, (20, 20, 22)), 64, 64), (0, 0));
        // …and green field elements are not an alliance either.
        assert_eq!(count_colours(&solid(64, 64, (30, 180, 60)), 64, 64), (0, 0));
    }

    #[test]
    fn a_frame_holding_both_alliances_is_called_mixed_not_a_coin_toss() {
        let both = split(64, 64, (220, 30, 40), (20, 90, 170));
        let (red, blue) = count_colours(&both, 64, 64);
        assert!(
            red > MIN_COLOURED && blue > MIN_COLOURED,
            "{red} red, {blue} blue"
        );
        assert_eq!(alliance_from(red, blue), Alliance::Mixed);
        // One end in shot ten times the other is still that end.
        assert_eq!(alliance_from(10_000, 100), Alliance::Red);
        assert_eq!(alliance_from(0, 0), Alliance::Unknown);
    }

    #[test]
    fn tags_name_the_end_they_stand_at() {
        // 2026 tags: 13-16 are on the red alliance wall at x = 16.53, 29-32 on
        // the blue one at x = 0.01.
        let layout = layout();
        assert_eq!(end_of_tags(&[13, 14], &layout), End::HighX);
        assert_eq!(end_of_tags(&[29, 30], &layout), End::LowX);
        assert_eq!(end_of_tags(&[13, 29], &layout), End::Both);
        // A tag the layout does not know, or no tag at all, places nothing.
        assert_eq!(end_of_tags(&[999], &layout), End::Unknown);
        assert_eq!(end_of_tags(&[], &layout), End::Unknown);
    }

    #[test]
    fn a_red_picture_of_the_blue_end_is_flagged() {
        let layout = layout();
        let red = solid(64, 64, (220, 30, 40));
        let verdict = analyse(&red, 64, 64, &[29], &layout, None);
        assert_eq!(verdict.alliance, Alliance::Red);
        assert_eq!(verdict.tags_end, End::LowX);
        let reason = verdict.disagreement.expect("should be flagged");
        assert!(reason.contains("red"), "{reason}");
        assert!(reason.contains("blue (low-x)"), "{reason}");

        // The other way round: blue picture, tags on the red wall.
        let blue = solid(64, 64, (20, 90, 170));
        let verdict = analyse(&blue, 64, 64, &[13], &layout, None);
        assert!(verdict.disagreement.is_some(), "{verdict:?}");
    }

    #[test]
    fn a_solve_that_lands_at_the_other_end_is_flagged_too() {
        let layout = layout();
        let red = solid(64, 64, (220, 30, 40));
        // A camera the picture says is at the red end, placed at the blue one
        // by the solve — what an assumed field of view does to a single tag.
        let camera = CameraPose {
            position: Vec3::new(1.0, 8.0, 4.0),
            rotation: crate::pose::types::Quat::IDENTITY,
        };
        let verdict = analyse(&red, 64, 64, &[13], &layout, Some(&camera));
        assert_eq!(verdict.camera_end, Some(End::LowX));
        assert_eq!(
            verdict.tags_end,
            End::HighX,
            "the tags themselves agree with red"
        );
        let reason = verdict.disagreement.expect("should be flagged");
        assert!(reason.contains("solved pose"), "{reason}");
    }

    #[test]
    fn a_frame_that_agrees_is_not_flagged() {
        let layout = layout();
        let red = solid(64, 64, (220, 30, 40));
        let camera = CameraPose {
            position: Vec3::new(15.0, 8.0, 4.0),
            rotation: crate::pose::types::Quat::IDENTITY,
        };
        let verdict = analyse(&red, 64, 64, &[13, 14], &layout, Some(&camera));
        assert_eq!(verdict.alliance, Alliance::Red);
        assert_eq!(verdict.tags_end, End::HighX);
        assert_eq!(verdict.camera_end, Some(End::HighX));
        assert_eq!(verdict.disagreement, None);

        // Both colours in shot, or nothing readable, is not a disagreement
        // either — the frame simply does not pick an end.
        let mixed = split(64, 64, (220, 30, 40), (20, 90, 170));
        let verdict = analyse(&mixed, 64, 64, &[13], &layout, None);
        assert_eq!(verdict.alliance, Alliance::Mixed);
        assert_eq!(verdict.disagreement, None);
    }

    #[test]
    fn the_splitter_names_the_overview_and_pose_reads_it() {
        assert_eq!(
            Pane::of(Path::new("views/2025casj_qm1/seg01_view1.mp4")),
            Pane::Overview
        );
        assert_eq!(Pane::of(Path::new("seg12_view1.mkv")), Pane::Overview);
        // The sub-views, a whole-frame clip, and a name that only looks the part.
        assert_eq!(Pane::of(Path::new("seg01_view2.mp4")), Pane::View);
        assert_eq!(Pane::of(Path::new("seg01_view10.mp4")), Pane::View);
        assert_eq!(Pane::of(Path::new("2025casj_qm1.mp4")), Pane::View);
        assert_eq!(Pane::of(Path::new("segment_view1.mp4")), Pane::View);
        assert_eq!(Pane::of(Path::new("seg01_view1_trimmed.mp4")), Pane::View);
    }
}

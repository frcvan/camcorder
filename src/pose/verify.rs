//! Whether the picture backs a pose up, beyond the corners it was fitted to.
//!
//! A pose is fitted to a tag's four outer corners, and those four points are the
//! corners of a *square*. A square is the same square mirrored, and the same
//! square turned half round, so the corners alone admit a second pose that
//! reprojects them to the same pixels and stands metres away on the field. On
//! the pane that prompted this, the solver's pool held two cameras 9 m apart that
//! fitted the same corners at 0.42px and 0.46px, and on two frames of that pane
//! the two answers fitted to the *same* 0.29px. No corner precision separates
//! those two, and neither can any amount of fitting: the fit is the one thing
//! that cannot choose, because both choices fit.
//!
//! Two things in the picture are not symmetric, and each rules out a different
//! half of the ambiguity:
//!
//! * **the code printed inside the tag's border**, which the decode already
//!   read. It says which side of the tag's plane the camera is on, because from
//!   the wrong one the ink reads mirrored. Like the corners, it is flat, so it
//!   cannot see the other half — how far away the camera is at what focal
//!   length, a trade that leaves the pattern on the same pixels.
//! * **the structure the tag is bolted to**, from the field model's own
//!   geometry, which does not lie in the tag's plane and so *does* see that
//!   trade. A pose that puts the mount where the carpet is scores near zero
//!   however well it reprojects the tag itself.
//!
//! Neither check is ever asked to produce a pose. They are asked which of two
//! poses the picture supports, which is a comparison — and a comparison needs no
//! identification of anything: the tag id was decoded, so the model already
//! knows what to project and where, and the picture is only asked whether
//! something is there.
//!
//! Nothing here rejects a pose on its own. A pose the picture does not back is
//! reported as unbacked, with the numbers, and left in the report: the mount may
//! simply be behind a robot, and a subsystem that quietly drops frames is worse
//! than one that says what it saw. A check also has to *read something* before
//! it may decide — see the floors below — because two readings near chance
//! differ by chance, and a check that picks the larger of two nothings is worse
//! than no check at all.

use super::{
    detect::{TagPatterns, TAG_CELLS},
    frames::Frame,
    layout::TagLayout,
    model,
    report::{FrameCheck, RivalPose},
    solve::{Candidates, Solution},
    types::{CameraPose, Intrinsics, Mat3, TagObservation, Vec3},
};

/// How wide a tag has to look before its printed pattern is read at all,
/// pixels, so its six cells of code are four pixels each.
///
/// Measured, not guessed, and the measurement is what set the floor. On the
/// Orlando pane the tags are 15-18px across; with the floor taken out of the way
/// the two answers of the 28 frames that hold a rival read 69-100% of the cells
/// each, and the reading backed the pose the mount backs in only 20 of them. The
/// own-agreement floor below cannot help there, because 69% clears it either way:
/// a reading that backs the wrong answer in 8 frames of 28 is not a decision, and
/// both numbers are high whatever happens, so a tag this small gets none, and the
/// mount decides instead. Nor is there anything to be gained by reading harder at
/// this size: the ink that separates two answers is a couple of pixels wide
/// either way. The report says the pattern could not be read. At 36px in the
/// fixture below the same reading is worth having — the answer the camera really
/// was in reads 89% of the cells and its twin 56%.
const MIN_PATTERN_TAG_PX: f64 = 24.0;

/// How much better one pose's pattern has to read than the other's before the
/// fit's own choice is overridden.
///
/// The ceiling is not the half a coin toss would suggest. A half turn leaves a
/// tag's own cells where they were in 16 of the 36, so even a perfect reading of
/// the wrong answer agrees with the right labels on those: tag 27 differs from
/// its own half turn in 20 cells, tag 18 in 22, tags 13 and 14 in 24 — the most
/// a correct reading can beat a turned one by is 0.56, and the fixture below gets
/// 0.33 of it (89% of the cells against its twin's 56%). A fifth is comfortably
/// under that and well above the noise of a reading taken at four pixels a cell.
const PATTERN_DECIDE: f64 = 0.2;

/// How much more of the tags' mount has to have an image edge under it. A tenth
/// of the model's own edges — enough that a pose which merely happens to look at
/// textured carpet is not preferred to one that looks at its own mount.
const STRUCTURE_DECIDE: f64 = 0.10;

/// What a reading has to reach before it is allowed to decide anything.
///
/// Below these floors the picture is not saying which pose is right, it is
/// saying nothing and the difference between two of its nothings is noise. Both
/// floors were put in because a test went red without them: given a nearly blank
/// picture, two poses that should have been indistinguishable came out 22%
/// against 10% of the mount edges and the check picked one — on nothing. A mount
/// that is really in frame reads 76-83%, so a floor of a third costs nothing in
/// the cases the check is for, and keeps it quiet in the ones it is not.
const MIN_PATTERN_AGREEMENT: f64 = 0.65;
const MIN_STRUCTURE_SUPPORT: f64 = 0.3;

/// How far from a projected model edge the image edge is looked for when scoring
/// a pose.
///
/// Tighter than [`model::EDGE_SEARCH_WIDE_PX`], deliberately. The wide window
/// exists to *find* the error in a pose that is metres out; here the only
/// question is whether the structure is where this pose says it is, and a
/// generous window would find something to match wherever it looked — scoring
/// the wrong pose as highly as the right one.
const STRUCTURE_SEARCH_PX: f64 = 8.0;

/// One tag's printed pattern, as the picture reads it.
#[derive(Debug, Clone, Copy)]
struct PatternRead {
    /// Fraction of the printed cells whose ink lands under the reading this pose
    /// predicts, `0..1`. Chance is 0.5.
    agreement: f64,
    /// Luma levels between the cells this pose predicts are light and the ones it
    /// predicts are dark.
    contrast: f64,
    /// Cells the reading was made from.
    cells: usize,
}

/// What the picture makes of one pose.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Scores {
    pattern: Option<PatternRead>,
    structure: Option<f64>,
}

/// The pose the picture backs, and the check that goes with it.
pub(crate) struct Checked {
    pub chosen: Solution,
    pub check: FrameCheck,
    /// What the picture made of the pose that lost, kept so the sentence in
    /// [`FrameCheck::rival`] can be made again if the run later moves this
    /// frame's pose.
    pub rival_scores: Option<Scores>,
}

/// The still picture a pose is judged against, and everything about it that does
/// not depend on the pose.
///
/// Held together because every part of the check needs all of them — where the
/// tags are, what they are printed with, what they are mounted on, and the
/// frame — and threading six arguments through four functions is how the wrong
/// one gets passed.
pub(crate) struct Picture<'a> {
    pub frame: &'a Frame,
    pub camera: &'a Intrinsics,
    pub layout: &'a TagLayout,
    pub tag_size: f64,
    pub patterns: &'a TagPatterns,
    /// Whether the bundled field model describes this layout at all. It is
    /// converted from the 2026 field's mesh, so on any other layout there is no
    /// structure to match and the check says so rather than guessing.
    pub structure: bool,
}

impl Picture<'_> {
    /// What the picture makes of a pose whose solve used `tags`.
    ///
    /// `observations` are the frame's own detections: the pattern is read where
    /// a tag was actually seen, so it cannot be read from a tag the frame does
    /// not hold.
    pub(crate) fn scores(
        &self,
        pose: &CameraPose,
        tags: &[u16],
        observations: &[TagObservation],
    ) -> Scores {
        let structure = if self.structure {
            model::structure_support(
                self.frame,
                pose,
                self.camera,
                tags,
                self.layout,
                STRUCTURE_SEARCH_PX,
            )
            .map(|(supported, considered)| supported as f64 / considered as f64)
        } else {
            None
        };
        Scores {
            pattern: read_pattern(
                self.frame,
                pose,
                self.camera,
                self.layout,
                self.tag_size,
                observations,
                self.patterns,
            ),
            structure,
        }
    }

    /// The pose the picture backs, among the poses the fit admits.
    ///
    /// The fit's own answer is kept unless the picture argues against it, and
    /// the argument has to be one the picture can actually make: a tag too small
    /// to read a cell, or a layout with no modelled structure, decides nothing.
    pub(crate) fn choose(
        &self,
        candidates: Candidates,
        observations: &[TagObservation],
    ) -> Checked {
        let Candidates { chosen, rivals } = candidates;
        let mut winner = chosen;
        let mut winner_scores = self.scores(&winner.pose, &winner.tags, observations);
        let mut losers: Vec<(Solution, Scores)> = Vec::new();
        for rival in rivals {
            let rival_scores = self.scores(&rival.pose, &rival.tags, observations);
            if let Some((true, _)) = decide(&rival_scores, &winner_scores) {
                let beaten = winner_scores;
                losers.push((std::mem::replace(&mut winner, rival), beaten));
                winner_scores = rival_scores;
            } else {
                losers.push((rival, rival_scores));
            }
        }

        // The rival carried into the report is the one that fits best of the
        // ones that lost: a frame with two of them is a frame about which the
        // reader needs to know the least uncertain alternative.
        let reported = losers
            .iter()
            .min_by(|a, b| a.0.rms_px.total_cmp(&b.0.rms_px));
        let rival = reported.map(|(solution, scores)| RivalPose {
            position: [
                solution.pose.position.x,
                solution.pose.position.y,
                solution.pose.position.z,
            ],
            rotation: [
                solution.pose.rotation.w,
                solution.pose.rotation.x,
                solution.pose.rotation.y,
                solution.pose.rotation.z,
            ],
            rms_px: solution.rms_px,
            pattern_agreement: scores.pattern.map(|read| read.agreement),
            structure_support: scores.structure,
            rejected: rejection(scores, &winner_scores),
        });

        if super::debug_enabled() {
            if let Some((solution, scores)) = reported {
                eprintln!(
                    "  [branch] two poses fit these corners: ({:.2}, {:.2}, {:.2})m rms {:.2}px ({}) \
                     · the other ({:.2}, {:.2}, {:.2})m rms {:.2}px ({})",
                    winner.pose.position.x,
                    winner.pose.position.y,
                    winner.pose.position.z,
                    winner.rms_px,
                    describe(&winner_scores),
                    solution.pose.position.x,
                    solution.pose.position.y,
                    solution.pose.position.z,
                    solution.rms_px,
                    describe(scores),
                );
            }
        }

        let mut check = FrameCheck::uninspected();
        apply(&mut check, winner_scores);
        check.rival = rival;
        Checked {
            chosen: winner,
            check,
            rival_scores: reported.map(|(_, scores)| *scores),
        }
    }
}

/// Put a set of scores into a frame's check.
///
/// Separate from scoring because the run re-scores a frame after the joint
/// fixed-camera fit moves it: the pose that gets reported is the fitted one, and
/// a check left over from the per-frame pose would be a check of a pose the
/// report no longer holds.
pub(crate) fn apply(check: &mut FrameCheck, scores: Scores) {
    check.pattern_agreement = scores.pattern.map(|read| read.agreement);
    check.pattern_contrast = scores.pattern.map(|read| read.contrast);
    check.pattern_cells = scores.pattern.map_or(0, |read| read.cells);
    check.structure_support = scores.structure;
}

/// What decided between two poses.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Basis {
    Pattern,
    Structure,
}

/// Which of two poses the picture backs, and on what evidence.
///
/// `None` when it does not separate them: nothing could be read, or what was
/// read differs by less than the noise. Two readings near chance differ by
/// chance, and choosing on that difference is how a check starts inventing
/// answers.
fn decide(a: &Scores, b: &Scores) -> Option<(bool, Basis)> {
    if let (Some(x), Some(y)) = (a.pattern, b.pattern) {
        let gap = (x.agreement - y.agreement).abs();
        if x.agreement.max(y.agreement) >= MIN_PATTERN_AGREEMENT && gap >= PATTERN_DECIDE {
            return Some((x.agreement > y.agreement, Basis::Pattern));
        }
    }
    if let (Some(x), Some(y)) = (a.structure, b.structure) {
        let gap = (x - y).abs();
        if x.max(y) >= MIN_STRUCTURE_SUPPORT && gap >= STRUCTURE_DECIDE {
            return Some((x > y, Basis::Structure));
        }
    }
    None
}

/// What the picture made of a pose, for the debug line.
fn describe(scores: &Scores) -> String {
    let pattern = match scores.pattern {
        Some(read) => format!(
            "pattern {:.0}% of {} cell(s), contrast {:.0}",
            read.agreement * 100.0,
            read.cells,
            read.contrast
        ),
        None => "pattern unreadable".to_string(),
    };
    let structure = match scores.structure {
        Some(support) => format!("mount {:.0}%", support * 100.0),
        None => "mount unmatched".to_string(),
    };
    format!("{pattern}, {structure}")
}

/// Why `loser` is not the pose that was reported, in words a reader can check.
///
/// Phrased from the loser's side, so it reads correctly whether the loser was
/// the fit's own answer or a rival that the picture turned down. When the
/// picture did not decide, the same numbers are given as numbers rather than as
/// a reason: an unresolved answer is worth reporting, not worth spinning.
///
/// Public to the subsystem because the sentence is made again when the joint
/// fit moves a frame's pose: the numbers in a sentence like this are a
/// comparison, and half a comparison left over from a pose the report no longer
/// holds is worse than no sentence at all.
pub(crate) fn rejection(loser: &Scores, winner: &Scores) -> String {
    let readings = readings(loser, winner);
    match decide(loser, winner) {
        Some((false, Basis::Pattern)) => format!(
            "the tags' own printed pattern reads {:.0}% of its cells here against {:.0}% from the \
             pose that was reported",
            loser.pattern.map_or(0.0, |read| read.agreement) * 100.0,
            winner.pattern.map_or(0.0, |read| read.agreement) * 100.0
        ),
        Some((false, Basis::Structure)) => format!(
            "{:.0}% of the mount the tags stand on has an edge under it here against {:.0}% from \
             the pose that was reported",
            loser.structure.unwrap_or(0.0) * 100.0,
            winner.structure.unwrap_or(0.0) * 100.0
        ),
        // The comparison runs both ways, and the joint fit can move a frame to a
        // pose that reads worse than the answer the report turned down. That is
        // worth saying outright rather than calling a tie.
        Some((true, _)) => format!(
            "the pose that was reported reads worse than this one: {}",
            readings.join("; ")
        ),
        None if readings.is_empty() => "the picture did not pick one: the tags were too small to \
                                         read their printed pattern and the layout has no modelled \
                                         structure to match"
            .to_string(),
        None => format!("the picture did not pick one — {}", readings.join("; ")),
    }
}

/// Both readings, as sentences, from the pose being described to the one that
/// was reported. Empty when the picture had nothing to read.
fn readings(loser: &Scores, winner: &Scores) -> Vec<String> {
    let mut parts = Vec::new();
    if let (Some(l), Some(w)) = (loser.pattern, winner.pattern) {
        parts.push(format!(
            "the tags' own printed pattern reads {:.0}% of its cells here against {:.0}% from the \
             pose that was reported",
            l.agreement * 100.0,
            w.agreement * 100.0
        ));
    }
    if let (Some(l), Some(w)) = (loser.structure, winner.structure) {
        parts.push(format!(
            "{:.0}% of the mount the tags stand on has an edge under it here against {:.0}% from \
             the pose that was reported",
            l * 100.0,
            w * 100.0
        ));
    }
    parts
}

/// Read the tags' printed patterns from where this pose says they are.
///
/// The cells are sampled at their centres in the frame the detector saw, and the
/// reading's own threshold is the midpoint between the light cells this pose
/// predicts and the dark ones — so what comes out is not how bright the picture
/// is but how well the pose explains it, with 0.5 for a pose the pattern cannot
/// tell from its mirror.
///
/// Only the six-by-six code is read, not the black border around it. The border
/// is the same ring on every tag and on both answers to the same corners, so
/// sampling it adds a fixed 44% of the cells to every reading — enough, on a
/// tag a few pixels wide, to drown out the cells that carry the answer. What is
/// left is the only part of the print that can tell two poses apart.
fn read_pattern(
    frame: &Frame,
    pose: &CameraPose,
    camera: &Intrinsics,
    layout: &TagLayout,
    tag_size: f64,
    observations: &[TagObservation],
    patterns: &TagPatterns,
) -> Option<PatternRead> {
    let cell = tag_size / TAG_CELLS as f64;
    let half = TAG_CELLS as f64 * 0.5;
    let rotation = Mat3::from_quaternion(pose.rotation);
    let mut samples: Vec<(f64, bool)> = Vec::new();
    for observation in observations {
        if observation.pixel_width() < MIN_PATTERN_TAG_PX {
            continue;
        }
        let Some(tag) = layout.tag(observation.id) else {
            continue;
        };
        let Some(ink) = patterns.ink(observation.id) else {
            continue;
        };
        let tag_rotation = Mat3::from_quaternion(tag.rotation);
        for row in 1..TAG_CELLS - 1 {
            for column in 1..TAG_CELLS - 1 {
                // The pattern's own frame: the picture's right is the tag's `+y`
                // and its down is the tag's `-z`, which is the corner order
                // `TagLayout::corners` is written in and the one the detector's
                // quad arrives in.
                let across = column as f64 + 0.5 - half;
                let up = row as f64 + 0.5 - half;
                let local = Vec3::new(0.0, across * cell, -up * cell);
                let field = tag_rotation.mul_vec(local) + tag.translation;
                let Some((u, v)) = camera.project(rotation.mul_vec(field - pose.position)) else {
                    continue;
                };
                let Some(luma) = model::sample_luma(frame, u, v) else {
                    continue;
                };
                samples.push((luma, ink[row * TAG_CELLS + column] == 1));
            }
        }
    }
    let light: Vec<f64> = samples
        .iter()
        .filter(|(_, light)| *light)
        .map(|(luma, _)| *luma)
        .collect();
    let dark: Vec<f64> = samples
        .iter()
        .filter(|(_, light)| !*light)
        .map(|(luma, _)| *luma)
        .collect();
    if light.is_empty() || dark.is_empty() {
        return None;
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (mean_light, mean_dark) = (mean(&light), mean(&dark));
    let level = 0.5 * (mean_light + mean_dark);
    let agreed = samples
        .iter()
        .filter(|(luma, light)| (*luma > level) == *light)
        .count();
    Some(PatternRead {
        agreement: agreed as f64 / samples.len() as f64,
        contrast: mean_light - mean_dark,
        cells: samples.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{
        layout::TagLayout,
        solve::look_towards,
        types::{TagObservation, Vec3},
    };

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    const TAG: u16 = 27;
    const SIZE: f64 = 0.1651;
    /// What the printed tag is drawn in, and what the field behind it is drawn
    /// in: enough contrast for a reading, as a broadcast delivers after
    /// enhancement rather than as a fresh PNG.
    const INK_LIGHT: u8 = 220;
    const INK_DARK: u8 = 30;
    const FIELD_GREY: u8 = 120;
    const EDGE_DARK: u8 = 40;

    fn layout() -> TagLayout {
        TagLayout::load("2026-rebuilt-welded").expect("bundled layout")
    }

    /// Narrow enough that the tag is 32px across at the distance below, so its
    /// code is a reading at four pixels a cell. A broadcast pane is 16px and
    /// gives no readable cells at all — that is what the gate refuses.
    fn camera() -> Intrinsics {
        Intrinsics::from_horizontal_fov(WIDTH, HEIGHT, 30.0)
    }

    /// Where the camera really stood for these tests: six metres out in front of
    /// tag 27's face, a little above the tag, which is what a broadcast view of
    /// the trench looks like.
    /// The camera's real position: `distance` metres from tag 27, 42° round from
    /// its face and a little above it.
    ///
    /// The obliquity is the whole point, and it is measured rather than chosen:
    /// a square tag seen off its axis is the case that leaves the corner fit two
    /// answers to choose between, and on the Orlando pane the tags are seen at
    /// 43°. Seen square on, the fit has one answer and none of this arises.
    fn truth_at(distance: f64) -> CameraPose {
        let layout = layout();
        let tag = layout.tag(TAG).expect("tag 27");
        let normal = Mat3::from_quaternion(tag.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        let direction =
            normal.scaled(0.74) + Vec3::new(1.0, 0.0, 0.0).scaled(0.65) + Vec3::new(0.0, 0.0, 0.13);
        look_towards(
            tag.translation + direction.scaled(distance),
            tag.translation,
        )
    }

    fn truth() -> CameraPose {
        truth_at(6.1)
    }

    /// The other camera that reprojects tag 27's corners onto the same pixels:
    /// the true one turned half round about the axis through the tag's centre
    /// along its own face normal.
    ///
    /// This is not a hypothetical. On the Orlando pane the solver's pool held
    /// exactly this pair — cameras 9 m apart, one at each end of the trench, both
    /// fitting the corners to half a pixel — and picking between them is what
    /// this module exists for.
    fn twin(pose: CameraPose) -> CameraPose {
        let layout = layout();
        let tag = layout.tag(TAG).expect("tag 27");
        let normal = Mat3::from_quaternion(tag.rotation).mul_vec(Vec3::new(1.0, 0.0, 0.0));
        let turn = Mat3::from_axis_angle(normal, std::f64::consts::PI);
        // The camera moves to the turned position and the world is turned under
        // it, so what it sees is what the real camera saw of the turned world:
        // for the tag's own square — which the half turn maps to itself — the
        // same pixels.
        CameraPose {
            position: tag.translation + turn.mul_vec(pose.position - tag.translation),
            rotation: Mat3::from_quaternion(pose.rotation)
                .mul_mat(&turn)
                .to_quaternion(),
        }
    }

    fn project(pose: &CameraPose, camera: &Intrinsics, point: Vec3) -> Option<(f64, f64)> {
        camera.project(pose.to_camera(point))
    }

    fn blank() -> Frame {
        Frame {
            time: 0.0,
            width: WIDTH,
            height: HEIGHT,
            pixels: vec![FIELD_GREY; (WIDTH * HEIGHT) as usize],
            rgb: vec![FIELD_GREY; (WIDTH * HEIGHT * 3) as usize],
        }
    }

    fn paint(frame: &mut Frame, x: f64, y: f64, value: u8) {
        if x < 0.0 || y < 0.0 || x >= WIDTH as f64 || y >= HEIGHT as f64 {
            return;
        }
        frame.pixels[y as usize * WIDTH as usize + x as usize] = value;
    }

    /// The tag as a camera at `pose` would photograph it: every cell of the
    /// printed pattern, in its place, in its own ink.
    fn draw_tag(frame: &mut Frame, pose: &CameraPose, camera: &Intrinsics, patterns: &TagPatterns) {
        let layout = layout();
        let tag = layout.tag(TAG).expect("tag 27");
        let tag_rotation = Mat3::from_quaternion(tag.rotation);
        let ink = patterns.ink(TAG).expect("a pattern");
        let cell = SIZE / TAG_CELLS as f64;
        let half = TAG_CELLS as f64 * 0.5;
        for row in 0..TAG_CELLS {
            for column in 0..TAG_CELLS {
                let value = if ink[row * TAG_CELLS + column] == 1 {
                    INK_LIGHT
                } else {
                    INK_DARK
                };
                let across = column as f64 - half;
                let up = row as f64 - half;
                // Nine points across the cell, so cells a few pixels wide are
                // filled rather than dotted: a dotted tag is not one any camera
                // and codec could deliver, and the reading below is about what a
                // real picture gives.
                for dy in [0.17, 0.5, 0.83] {
                    for dx in [0.17, 0.5, 0.83] {
                        let local = Vec3::new(0.0, (across + dx) * cell, -(up + dy) * cell);
                        let point = tag_rotation.mul_vec(local) + tag.translation;
                        if let Some((u, v)) = project(pose, camera, point) {
                            paint(frame, u.round(), v.round(), value);
                        }
                    }
                }
            }
        }
    }

    /// The mount the tag stands on, drawn from where `pose` says it is: the
    /// off-plane part of it only, since that is the part that says anything
    /// about which pose the camera is in.
    fn draw_structure(frame: &mut Frame, pose: &CameraPose, camera: &Intrinsics) {
        for entry in model::bundled_structure() {
            if entry.id != TAG {
                continue;
            }
            for segment in &entry.segments {
                let start = Vec3::from_array(segment[0]);
                let end = Vec3::from_array(segment[1]);
                for step in 0..=120 {
                    let t = step as f64 / 120.0;
                    if let Some((u, v)) = project(pose, camera, start + (end - start) * t) {
                        paint(frame, u.round(), v.round(), EDGE_DARK);
                    }
                }
            }
        }
    }

    fn observation(pose: &CameraPose, camera: &Intrinsics) -> TagObservation {
        let corners = layout().corners(TAG, SIZE).expect("corners");
        let mut image = [[0.0f64; 2]; 4];
        for (index, corner) in corners.iter().enumerate() {
            let (u, v) = project(pose, camera, *corner).expect("in front of the camera");
            image[index] = [u, v];
        }
        TagObservation {
            id: TAG,
            corners: image,
            margin: 200.0,
            hamming: 0,
        }
    }

    /// The picture both poses see: the tag's own print and the mount it stands
    /// on, drawn from the camera's real position.
    fn scene() -> (Frame, TagObservation, TagPatterns) {
        let patterns = TagPatterns::new().expect("the family loads");
        let camera = camera();
        let truth = truth();
        let mut frame = blank();
        draw_tag(&mut frame, &truth, &camera, &patterns);
        draw_structure(&mut frame, &truth, &camera);
        let observation = observation(&truth, &camera);
        (frame, observation, patterns)
    }

    /// The same camera, on a picture of the mount alone: no tag drawn at all, so
    /// the pattern has nothing to read and only the model can answer.
    fn mount_only_scene() -> (Frame, TagObservation, TagPatterns) {
        let patterns = TagPatterns::new().expect("the family loads");
        let camera = camera();
        let truth = truth();
        let mut frame = blank();
        draw_structure(&mut frame, &truth, &camera);
        let observation = observation(&truth, &camera);
        (frame, observation, patterns)
    }

    fn picture<'a>(
        frame: &'a Frame,
        camera: &'a Intrinsics,
        layout: &'a TagLayout,
        patterns: &'a TagPatterns,
    ) -> Picture<'a> {
        Picture {
            frame,
            camera,
            layout,
            tag_size: SIZE,
            patterns,
            structure: true,
        }
    }

    #[test]
    fn the_picture_backs_the_pose_the_camera_was_really_in() {
        let (frame, observation, patterns) = scene();
        let layout = layout();
        let camera = camera();
        let picture = picture(&frame, &camera, &layout, &patterns);
        let scores = picture.scores(&truth(), &[TAG], std::slice::from_ref(&observation));

        let pattern = scores.pattern.expect("the tag is big enough to read");
        assert_eq!(pattern.cells, 36, "one tag carries a six-by-six code");
        assert!(
            pattern.agreement > 0.85,
            "the pose the picture was drawn from should read its own pattern, got {:.0}%",
            pattern.agreement * 100.0
        );
        assert!(
            pattern.contrast > 60.0,
            "a printed tag should read light against dark, got {:.0} levels",
            pattern.contrast
        );
        assert!(
            scores.structure.is_some_and(|support| support > 0.8),
            "the mount was drawn where this pose puts it, got {:?}",
            scores.structure
        );
    }

    #[test]
    fn the_twin_pose_lands_on_the_same_corners_but_does_not_read_the_tag() {
        // Why the corners cannot be the thing that decides, and what can.
        //
        // The half turn maps the tag's *square* onto itself — so the four
        // corners land on the same four pixels and the fit is a tie — but it does
        // not map the *code* onto itself, so the ink under the cells no longer
        // matches what the tag is printed with, and the mount moves outright
        // because it is not on the tag's plane at all. Both facts are measured
        // here rather than assumed.
        let (frame, observation, patterns) = scene();
        let layout = layout();
        let camera = camera();
        let picture = picture(&frame, &camera, &layout, &patterns);
        let observations = std::slice::from_ref(&observation);
        let truth_scores = picture.scores(&truth(), &[TAG], observations);
        let twin_scores = picture.scores(&twin(truth()), &[TAG], observations);

        // Both answers put the tag's four corners on the same pixels: as a set.
        let truth = truth();
        let twin = twin(truth);
        let corners = layout.corners(TAG, SIZE).expect("corners");
        let seen = |pose: &CameraPose| {
            corners
                .iter()
                .map(|corner| project(pose, &camera, *corner).expect("in front"))
                .collect::<Vec<_>>()
        };
        let (a, b) = (seen(&truth), seen(&twin));
        for corner in &b {
            let nearest = a
                .iter()
                .map(|other| (other.0 - corner.0).hypot(other.1 - corner.1))
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest < 1.5,
                "the twin should put the corners on the same pixels, {nearest:.1}px off"
            );
        }

        let (truth_pattern, twin_pattern) = (
            truth_scores.pattern.expect("readable"),
            twin_scores.pattern.expect("readable"),
        );
        assert!(
            truth_pattern.agreement - twin_pattern.agreement >= PATTERN_DECIDE,
            "the tag's own code should tell the two answers apart: {:.0}% against {:.0}%",
            truth_pattern.agreement * 100.0,
            twin_pattern.agreement * 100.0
        );
        let (truth_structure, twin_structure) = (
            truth_scores.structure.expect("the mount is modelled"),
            twin_scores.structure.expect("the mount is modelled"),
        );
        assert!(
            truth_structure - twin_structure >= STRUCTURE_DECIDE,
            "the mount is off the tag's plane, so it should separate the pair: \
             {:.0}% against {:.0}%",
            truth_structure * 100.0,
            twin_structure * 100.0
        );
    }

    fn answer(pose: CameraPose, rms_px: f64) -> Solution {
        Solution {
            pose,
            rms_px,
            corners: 4,
            tags: vec![TAG],
        }
    }

    #[test]
    fn the_model_alone_decides_between_the_two_answers() {
        // The failure this module exists for: the fit's own answer is the wrong
        // one and it fits *better* than the right one. On the Orlando pane the
        // run reported a camera at the far end of the trench that was never
        // there, at 0.40px against the right answer's 0.46px, and no amount of
        // corner precision would have caught it. The picture here holds the
        // mount and no tag at all: the pattern has nothing to read, so the field
        // model is the only thing that can answer — and it has to win against a
        // better-fitting wrong pose.
        let (frame, observation, patterns) = mount_only_scene();
        let layout = layout();
        let camera = camera();
        let picture = picture(&frame, &camera, &layout, &patterns);
        let observations = std::slice::from_ref(&observation);

        let checked = picture.choose(
            Candidates {
                chosen: answer(twin(truth()), 0.42),
                rivals: vec![answer(truth(), 0.55)],
            },
            observations,
        );
        assert!(
            (checked.chosen.pose.position - truth().position).norm() < 0.05,
            "the pose the mount stands under should be reported, got {:?}",
            checked.chosen.pose.position
        );
        let rival = checked
            .check
            .rival
            .as_ref()
            .expect("the pose that was not chosen is reported");
        assert!(
            rival.rejected.contains("mount"),
            "the reason has to say what decided it, got {:?}",
            rival.rejected
        );
        assert!(
            (Vec3::from_array(rival.position) - checked.chosen.pose.position).norm() > 1.0,
            "a rival is somewhere else, not a nudge"
        );
    }

    #[test]
    fn a_picture_that_cannot_choose_says_so_rather_than_guessing() {
        // A tag far enough off that its cells cannot be read, on a picture with
        // no mount in it either: neither check has anything to say. Nothing here
        // can decide, so the fit's own answer stands and the report says the
        // question was open. Inventing a preference out of noise is how a check
        // starts lying, and a reading from a tag this small is pure noise.
        let patterns = TagPatterns::new().expect("the family loads");
        let camera = camera();
        let layout = layout();
        let far = truth_at(25.0);
        let mut frame = blank();
        draw_tag(&mut frame, &far, &camera, &patterns);
        let observation = observation(&far, &camera);
        let picture = picture(&frame, &camera, &layout, &patterns);
        assert!(
            picture
                .scores(&far, &[TAG], std::slice::from_ref(&observation))
                .pattern
                .is_none(),
            "a tag this small must not produce a pattern reading at all"
        );

        let checked = picture.choose(
            Candidates {
                chosen: answer(twin(far), 0.42),
                rivals: vec![answer(far, 0.48)],
            },
            std::slice::from_ref(&observation),
        );
        let rival = checked
            .check
            .rival
            .as_ref()
            .expect("the other answer is still reported");
        assert!(
            rival.rejected.contains("did not pick one"),
            "nothing separated them, so the report must say so, got {:?}",
            rival.rejected
        );
        assert!(
            (checked.chosen.pose.position - twin(far).position).norm() < 0.05,
            "with no evidence either way the fit's own answer stands"
        );
    }

    #[test]
    fn a_reason_quotes_the_pose_the_report_holds() {
        // The reason a rival was turned down is a comparison, and the floor of
        // it is the reported pose's own reading. The joint fixed-camera fit
        // moves a frame's pose after the per-frame check was made, so the
        // sentence is made again at that point: left as it was, it would quote a
        // percentage from a camera the report no longer holds, and that number
        // is the one thing in the report a reader can check by hand.
        let read = |agreement: f64| Scores {
            pattern: Some(PatternRead {
                agreement,
                contrast: 60.0,
                cells: 36,
            }),
            structure: Some(agreement),
        };
        let fitted = rejection(&read(0.63), &read(0.74));
        assert!(
            fitted.contains("74%") && !fitted.contains("83%"),
            "the reason has to quote the pose that is reported, got {fitted:?}"
        );

        // And when the fit moves the pose somewhere the picture likes less than
        // the answer that was turned down, saying so is better than pretending
        // they tied: both readings are given, with which way round they go.
        let overtaken = rejection(&read(0.74), &read(0.63));
        assert!(
            overtaken.contains("reads worse than this one"),
            "a pose moved onto weaker ground has to say so, got {overtaken:?}"
        );
        assert!(
            overtaken.contains("74%") && overtaken.contains("63%"),
            "both readings belong in it, got {overtaken:?}"
        );
    }

    #[test]
    fn a_solved_frame_carries_the_other_answer_a_lone_tag_leaves() {
        use crate::pose::solve;

        // The colours around a mid-field tag can name the far end of the field,
        // which is what sends the solver looking for another answer; the pool
        // then holds both, and both have to come out.
        //
        // The corners are moved by half a pixel because that is what makes the
        // two answers competitive: with exact corners the fit has a unique
        // answer and the question does not arise. Real corners are never exact.
        let (frame, mut observation, patterns) = scene();
        let mut seed = 11u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f64 / (1u32 << 24) as f64) - 0.5
        };
        for corner in observation.corners.iter_mut() {
            corner[0] += noise();
            corner[1] += noise();
        }
        let layout = layout();
        let camera = camera();
        let picture = picture(&frame, &camera, &layout, &patterns);
        let candidates = solve::solve_all(
            std::slice::from_ref(&observation),
            &layout,
            SIZE,
            &camera,
            Some(crate::pose::End::HighX),
        )
        .expect("one tag solves");
        eprintln!(
            "pool: chosen at {:?} rms {:.3}, {} rival(s)",
            candidates.chosen.pose.position,
            candidates.chosen.rms_px,
            candidates.rivals.len()
        );
        assert!(
            !candidates.rivals.is_empty(),
            "a single square tag admits a second answer metres away"
        );
        let checked = picture.choose(candidates, std::slice::from_ref(&observation));
        let rival = checked.check.rival.as_ref().expect("a rival is reported");
        assert!(
            (Vec3::from_array(rival.position) - checked.chosen.pose.position).norm() > 1.0,
            "the rival should be somewhere else"
        );
    }
}

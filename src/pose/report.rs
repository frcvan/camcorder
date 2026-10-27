//! What the pose subsystem produces, and how it is written out.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use super::{
    side::{Pane, SideVerdict},
    types::CameraPose,
};

/// What one solved frame contributes, before it becomes a row of the report.
///
/// Gathered into a struct because the alternative is eight positional arguments
/// whose order nothing checks, and every one of them is a plausible-looking
/// number.
pub struct FrameOutcome<'a> {
    pub time: f64,
    pub pose: &'a CameraPose,
    /// Tag ids used for the solve, ascending.
    pub tags: Vec<u16>,
    pub corners: usize,
    pub rms_px: f64,
    /// `0..1`, see [`super::trust::confidence`].
    pub confidence: f64,
    /// The least confident detection the solve used, in the decoder's margin.
    pub weakest_margin: f32,
    pub side: Option<SideVerdict>,
}

/// The camera's pose at one sampled frame.
#[derive(Debug, Clone, Serialize)]
pub struct FramePose {
    /// Seconds into the video.
    pub time: f64,
    /// Tag ids used for the solve, ascending.
    pub tags: Vec<u16>,
    /// Corners the solve was based on.
    pub corners: usize,
    /// Root-mean-square reprojection error of the solution, in pixels.
    pub rms_px: f64,
    /// How much of a result this frame is, `0..1`: one tag alone is half a
    /// result, a bent fit is less, a barely-read tag less again. Frames are
    /// ranked by this, not by the fit, because a single tag can fit the corners
    /// to a tenth of a pixel and still be metres out.
    pub confidence: f64,
    /// The least confident detection the solve used, in the decoder's margin.
    pub weakest_margin: f32,
    /// Camera centre in field coordinates, metres, `+z` up.
    pub position: [f64; 3],
    /// Camera orientation: field axes to camera axes, as a quaternion.
    pub rotation: [f64; 4],
    /// Heading of the optical axis in the field plane, degrees from `+x`.
    pub yaw_deg: f64,
    /// Elevation of the optical axis above the field plane, degrees.
    pub pitch_deg: f64,
    /// Roll about the optical axis, degrees.
    pub roll_deg: f64,
    /// Which side of the field the frame was taken from: the alliance its
    /// colours show, the end its tags stand at, and where the pose landed.
    /// `None` when the file is the overview pane, which holds both alliances
    /// and so cannot answer the question.
    pub side: Option<SideVerdict>,
}

impl FramePose {
    pub fn new(outcome: FrameOutcome<'_>) -> Self {
        let pose = outcome.pose;
        let rotation = pose.rotation;
        FramePose {
            time: outcome.time,
            tags: outcome.tags,
            corners: outcome.corners,
            rms_px: outcome.rms_px,
            confidence: outcome.confidence,
            weakest_margin: outcome.weakest_margin,
            position: [pose.position.x, pose.position.y, pose.position.z],
            rotation: [rotation.w, rotation.x, rotation.y, rotation.z],
            yaw_deg: pose.yaw_deg(),
            pitch_deg: pose.pitch_deg(),
            roll_deg: pose.roll_deg(),
            side: outcome.side,
        }
    }

    /// Replace this row's pose with the one the fixed-camera fit gives it.
    ///
    /// A single tag is solved two equally well-fitting ways, so a per-frame
    /// solve can land on the far side of the wall from the camera that filmed
    /// it. Once the run knows where the camera stood, the row should say that,
    /// and keep its own orientation — the camera panned and zoomed, it did not
    /// move. `rms_px` is recomputed by the caller against the new pose.
    pub(crate) fn adopt(&mut self, pose: &CameraPose) {
        self.position = [pose.position.x, pose.position.y, pose.position.z];
        self.rotation = [
            pose.rotation.w,
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
        ];
        self.yaw_deg = pose.yaw_deg();
        self.pitch_deg = pose.pitch_deg();
        self.roll_deg = pose.roll_deg();
    }

    /// The alliance the frame's colours show, as one word for the table.
    fn side_label(&self) -> &'static str {
        match self.side.as_ref().map(|side| side.alliance) {
            Some(super::side::Alliance::Red) => "red",
            Some(super::side::Alliance::Blue) => "blue",
            Some(super::side::Alliance::Mixed) => "both",
            Some(super::side::Alliance::Unknown) | None => "--",
        }
    }
}

/// A frame in which no pose could be solved.
#[derive(Debug, Clone, Serialize)]
pub struct MissedFrame {
    pub time: f64,
    pub tags: Vec<u16>,
    pub reason: String,
}

/// One pose for the whole file's camera, fitted to every frame at once.
///
/// This is what a broadcast camera actually did: it stood still. The per-frame
/// poses scatter around this by their corner noise, and a frame far from it is
/// a pan or a bad solve — `frames_rejected` counts those.
#[derive(Debug, Clone, Serialize)]
pub struct StationReport {
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub rms_px: f64,
    pub frames_used: usize,
    pub frames_rejected: usize,
}

/// A sampled orientation and focal length fitted at the shared camera position.
#[derive(Debug, Clone, Serialize)]
pub struct PtzFrameReport {
    pub time: f64,
    pub rotation: [f64; 4],
    pub focal_px: f64,
}

/// A shared camera centre only reported when the observations constrain it.
#[derive(Debug, Clone, Serialize)]
pub struct PtzStationReport {
    pub position: [f64; 3],
    pub rms_px: f64,
    /// Number of frames containing matched 3D field-model structure.
    pub model_frames: usize,
    /// Tag corner observations the fixation was based on.
    pub corner_samples: usize,
    /// Image-to-model line observations the fixation was based on.
    pub line_samples: usize,
    /// Largest displacement of the centre the footage could not rule out, metres.
    /// Censored: `2.0` means "more than two metres, the probe's reach".
    pub sigma_m: f64,
    /// Frames whose own solve agreed with this centre, out of `frames.len()`:
    /// the evidence the position rests on. A lone tag solves about as well from
    /// the far side of the wall, so on a single-tag pane the frames split into
    /// two basins and this counts only the one the answer came from.
    pub frames_agreeing: usize,
    /// Frames that had to be re-aimed at this centre, having been solved from
    /// somewhere else by their own frame.
    pub frames_stray: usize,
    /// Frames the fitted camera does not explain even after re-aiming.
    pub frames_unexplained: usize,
    /// Whether the centre is too loosely determined to be read as a position.
    pub ambiguous: bool,
    pub frames: Vec<PtzFrameReport>,
}

impl PtzStationReport {
    /// One line for the table: the centre, and how tightly it is held.
    pub fn describe(&self) -> String {
        let bound = if self.ambiguous {
            format!(
                "not pinned: a camera up to {:.1}m away explains these frames too",
                self.sigma_m
            )
        } else {
            format!("within {:.2}m of this", self.sigma_m)
        };
        let mut evidence = format!(
            "{} corner(s) in {} frame(s), {} agreeing on this centre",
            self.corner_samples,
            self.frames.len(),
            self.frames_agreeing
        );
        if self.frames_stray > 0 {
            evidence.push_str(&format!(", {} re-aimed from elsewhere", self.frames_stray));
        }
        if self.frames_unexplained > 0 {
            evidence.push_str(&format!(
                ", {} not explained by it",
                self.frames_unexplained
            ));
        }
        let structure = match self.model_frames {
            0 => "no model structure matched".to_string(),
            frames => format!(
                "{} model line match(es) in {frames} frame(s)",
                self.line_samples
            ),
        };
        format!(
            "fixed position ({evidence}, {structure}, rms {:.1}px, {bound}): ({:.2}, {:.2}, {:.2})m",
            self.rms_px, self.position[0], self.position[1], self.position[2]
        )
    }
}

/// Everything one run found.
#[derive(Debug, Clone, Serialize)]
pub struct PoseReport {
    pub video: PathBuf,
    pub tag_layout: String,
    pub frame_width: u32,
    pub frame_height: u32,
    /// The horizontal field of view the poses were solved with, degrees. This is
    /// the `--fov` guess unless frames with two or more tags measured it.
    pub horizontal_fov_deg: f64,
    /// The focal length the poses were solved with, pixels.
    pub focal_px: f64,
    /// Where that focal length came from, in words, for the report to print.
    pub focal_source: String,
    /// Physical edge length of one tag, metres.
    pub tag_size_m: f64,
    pub sampled: usize,
    /// Which pane of the broadcast frame this file holds; the overview pane is
    /// not asked about the side of the field.
    pub pane: Pane,
    pub solved: Vec<FramePose>,
    pub missed: Vec<MissedFrame>,
    /// Every tag id that was read in some frame, ascending.
    pub tags_seen: Vec<u16>,
    /// Frames below this confidence were dropped from `solved` and moved to
    /// `missed`, if the run asked for that.
    pub min_confidence: f64,
    /// The one pose that fits every frame of the station, when there was one.
    pub station: Option<StationReport>,
    /// Fixed position with independent pan, tilt, roll and zoom when constrained.
    pub ptz_station: Option<PtzStationReport>,
}

impl PoseReport {
    /// Median camera height above the field, over the frames that solved.
    ///
    /// A cheap sanity number: a broadcast camera is metres above the carpet,
    /// never below it.
    pub fn median_height(&self) -> Option<f64> {
        if self.solved.is_empty() {
            return None;
        }
        let mut heights: Vec<f64> = self.solved.iter().map(|frame| frame.position[2]).collect();
        heights.sort_by(f64::total_cmp);
        Some(heights[heights.len() / 2])
    }

    /// The frame worth quoting: the most confident one, not the best-fitting
    /// one. A single tag can fit to a tenth of a pixel and be metres out.
    pub fn best(&self) -> Option<&FramePose> {
        self.solved
            .iter()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
    }

    /// One line per solved frame, for the terminal.
    pub fn render_table(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        if self.solved.is_empty() {
            out.push_str("no pose could be solved from this video\n");
        } else {
            let _ = writeln!(
                out,
                "   time  tags  corners   rms  conf  camera x      y      z    yaw   pitch   roll  side"
            );
            for frame in &self.solved {
                let _ = writeln!(
                    out,
                    "  {:5.1}s  {:4}  {:7}  {:4.1}  {:4.2}  {:6.2} {:6.2} {:6.2}  {:5.1} {:6.1} {:6.1}  {:>4}",
                    frame.time,
                    frame.tags.len(),
                    frame.corners,
                    frame.rms_px,
                    frame.confidence,
                    frame.position[0],
                    frame.position[1],
                    frame.position[2],
                    frame.yaw_deg,
                    frame.pitch_deg,
                    frame.roll_deg,
                    frame.side_label(),
                );
            }
            if let Some(station) = &self.station {
                let _ = writeln!(
                    out,
                    "  station pose ({}, rms {:.1}px): ({:.2}, {:.2}, {:.2})m · yaw {:.0}° pitch {:.0}°",
                    match station.frames_rejected {
                        0 => format!("all {} frame(s)", station.frames_used),
                        1 => format!("{} frame(s), 1 rejected", station.frames_used),
                        rejected => format!(
                            "{} frame(s), {rejected} rejected",
                            station.frames_used
                        ),
                    },
                    station.rms_px,
                    station.position[0],
                    station.position[1],
                    station.position[2],
                    heading_deg(&station.rotation),
                    pitch_deg(&station.rotation),
                );
            }
            for frame in &self.solved {
                if let Some(disagreement) = frame
                    .side
                    .as_ref()
                    .and_then(|side| side.disagreement.as_ref())
                {
                    let _ = writeln!(out, "  {:5.1}s: {disagreement}", frame.time);
                }
            }
        }
        if let Some(ptz) = &self.ptz_station {
            let _ = writeln!(out, "  {}", ptz.describe());
            if ptz.ambiguous {
                let _ = writeln!(
                    out,
                    "    the centre is the best fit, not a measurement: with this much footage a camera elsewhere fits nearly as well"
                );
            }
        }
        if self.pane == Pane::Overview {
            let _ = writeln!(
                out,
                "overview pane: the side of the field was not read from it (one wide shot holds both alliances)"
            );
        }
        if !self.missed.is_empty() {
            let _ = writeln!(out, "{} frame(s) had no usable pose", self.missed.len());
        }
        out
    }

    pub fn write_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self).context("encoding the pose report")?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

/// Heading of the optical axis in the field plane, from a field-to-camera
/// quaternion.
fn heading_deg(rotation: &[f64; 4]) -> f64 {
    let pose = CameraPose {
        position: super::types::Vec3::new(0.0, 0.0, 0.0),
        rotation: super::types::Quat::new(rotation[0], rotation[1], rotation[2], rotation[3]),
    };
    pose.yaw_deg()
}

fn pitch_deg(rotation: &[f64; 4]) -> f64 {
    let pose = CameraPose {
        position: super::types::Vec3::new(0.0, 0.0, 0.0),
        rotation: super::types::Quat::new(rotation[0], rotation[1], rotation[2], rotation[3]),
    };
    pose.pitch_deg()
}

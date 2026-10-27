//! What the pose subsystem produces, and how it is written out.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use super::types::CameraPose;

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
}

impl FramePose {
    pub fn new(time: f64, pose: &CameraPose, tags: Vec<u16>, corners: usize, rms: f64) -> Self {
        let rotation = pose.rotation;
        FramePose {
            time,
            tags,
            corners,
            rms_px: rms,
            position: [pose.position.x, pose.position.y, pose.position.z],
            rotation: [rotation.w, rotation.x, rotation.y, rotation.z],
            yaw_deg: pose.yaw_deg(),
            pitch_deg: pose.pitch_deg(),
            roll_deg: pose.roll_deg(),
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

/// Everything one run found.
#[derive(Debug, Clone, Serialize)]
pub struct PoseReport {
    pub video: PathBuf,
    pub tag_layout: String,
    pub frame_width: u32,
    pub frame_height: u32,
    /// The assumed horizontal field of view, degrees.
    pub horizontal_fov_deg: f64,
    /// Physical edge length of one tag, metres.
    pub tag_size_m: f64,
    pub sampled: usize,
    pub solved: Vec<FramePose>,
    pub missed: Vec<MissedFrame>,
    /// Every tag id that was read at some point, ascending.
    pub tags_seen: Vec<u16>,
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

    pub fn best(&self) -> Option<&FramePose> {
        self.solved
            .iter()
            .min_by(|a, b| a.rms_px.total_cmp(&b.rms_px))
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
                "   time  tags  corners   rms  camera x      y      z    yaw   pitch   roll"
            );
            for frame in &self.solved {
                let _ = writeln!(
                    out,
                    "  {:5.1}s  {:4}  {:7}  {:4.1}  {:6.2} {:6.2} {:6.2}  {:5.1} {:6.1} {:6.1}",
                    frame.time,
                    frame.tags.len(),
                    frame.corners,
                    frame.rms_px,
                    frame.position[0],
                    frame.position[1],
                    frame.position[2],
                    frame.yaw_deg,
                    frame.pitch_deg,
                    frame.roll_deg,
                );
            }
        }
        if !self.missed.is_empty() {
            let _ = writeln!(
                out,
                "{} frame(s) had no solvable pose ({} tag(s) read at most)",
                self.missed.len(),
                self.missed
                    .iter()
                    .map(|missed| missed.tags.len())
                    .max()
                    .unwrap_or(0)
            );
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

//! Pulling single greyscale frames out of a video.
//!
//! Everything in this subsystem that talks to `ffmpeg` lives here, so the rest
//! of it only ever sees a [`GrayFrame`].

use std::{
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{bail, Context, Result};

use crate::video::{VideoInfo, BIN_FFMPEG};

/// One frame at full resolution, one byte of luma per pixel, top row first.
#[derive(Debug, Clone)]
pub struct GrayFrame {
    pub time: f64,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Times to sample for a video: `count` frames spread evenly across the run,
/// or across `window` when one is given.
pub fn sample_times(duration: f64, count: usize, window: Option<(f64, f64)>) -> Vec<f64> {
    let (start, end) = match window {
        // An explicit window that ends before it starts is a mistake in the
        // request, not an invitation to sample the whole video.
        Some((start, end)) => {
            if end.is_nan() || start.is_nan() || end <= start {
                return Vec::new();
            }
            (start.max(0.0), end)
        }
        None => (0.0, duration.max(0.0)),
    };
    if count == 0 || end.is_nan() || end <= start {
        return Vec::new();
    }
    (0..count)
        .map(|index| start + (end - start) * (index as f64 + 0.5) / count as f64)
        .collect()
}

/// Decode one frame at each of `times`.
///
/// A frame that cannot be decoded is reported but does not stop the run: live
/// broadcasts have the odd glitched splice, and losing one sample out of ten is
/// better than losing the estimate.
pub fn grab(video: &Path, info: &VideoInfo, times: &[f64]) -> Result<Vec<GrayFrame>> {
    let mut frames = Vec::with_capacity(times.len());
    for &time in times {
        match grab_one(video, time, info.width, info.height) {
            Ok(frame) => frames.push(frame),
            Err(error) if !frames.is_empty() => {
                eprintln!("  note: could not sample {time:.2}s ({error:#})");
            }
            Err(error) => return Err(error),
        }
    }
    if frames.is_empty() {
        bail!("no frames could be decoded from {}", video.display());
    }
    Ok(frames)
}

fn grab_one(video: &Path, time: f64, width: u32, height: u32) -> Result<GrayFrame> {
    let mut cmd = Command::new(BIN_FFMPEG);
    cmd.args([
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        &format!("{time:.3}"),
        "-i",
    ])
    .arg(video)
    .args([
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-vf",
        "format=gray",
        "-f",
        "rawvideo",
        "-",
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    let output = cmd
        .output()
        .with_context(|| format!("running {BIN_FFMPEG} to sample {time:.3}s"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "{BIN_FFMPEG} failed at {time:.3}s: {}",
            stderr.trim().lines().next().unwrap_or("no message")
        );
    }

    let expected = width as usize * height as usize;
    if output.stdout.len() != expected {
        bail!(
            "expected {expected} bytes of greyscale for {width}x{height}, got {} (frame at {time:.3}s)",
            output.stdout.len()
        );
    }
    Ok(GrayFrame {
        time,
        width,
        height,
        pixels: output.stdout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_evenly_spread_and_inside_the_run() {
        let times = sample_times(100.0, 4, None);
        assert_eq!(times.len(), 4);
        assert!((times[0] - 12.5).abs() < 1e-9);
        assert!((times[3] - 87.5).abs() < 1e-9);
        assert!(times.iter().all(|time| *time > 0.0 && *time < 100.0));
    }

    #[test]
    fn a_window_restricts_the_samples() {
        let times = sample_times(100.0, 2, Some((10.0, 20.0)));
        assert!((times[0] - 12.5).abs() < 1e-9);
        assert!((times[1] - 17.5).abs() < 1e-9);
    }

    #[test]
    fn a_degenerate_window_or_count_yields_nothing() {
        assert!(sample_times(0.0, 4, None).is_empty());
        assert!(sample_times(30.0, 0, None).is_empty());
        assert!(sample_times(30.0, 4, Some((20.0, 20.0))).is_empty());
    }
}

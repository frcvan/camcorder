//! Glue between the TBA client, the output layout on disk and the video splitter.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};

use crate::{
    config::CamcorderConfig,
    tba::{self, Match, TbaClient, VideoRef},
    video::{self, SplitOptions, SplitReport},
};

/// Where everything for one team lands.
#[derive(Debug, Clone)]
pub struct Paths {
    pub root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn team_root(&self, team: u32) -> PathBuf {
        self.root.join(tba::team_key(team))
    }

    pub fn matches_json(&self, team: u32, year: u16) -> PathBuf {
        self.team_root(team).join(format!("matches-{year}.json"))
    }

    pub fn videos_dir(&self, team: u32) -> PathBuf {
        self.team_root(team).join("videos")
    }

    pub fn video_file(&self, team: u32, match_key: &str) -> PathBuf {
        self.videos_dir(team).join(format!("{match_key}.mp4"))
    }

    pub fn views_dir(&self, team: u32, match_key: &str) -> PathBuf {
        self.team_root(team).join("views").join(match_key)
    }
}

/// A season of matches, plus where the raw response was written.
pub struct Fetched {
    pub matches: Vec<Match>,
    pub json_path: PathBuf,
}

/// Pull the team's matches for the configured season and save the raw response.
pub fn fetch_matches(
    config: &CamcorderConfig,
    client: &TbaClient,
    paths: &Paths,
) -> Result<Fetched> {
    let season = client.team_matches(config.team, config.year)?;
    let total = season.matches.len();

    let mut matches = season.matches;
    if let Some(event) = &config.event {
        matches.retain(|m| &m.event_key == event);
        if matches.is_empty() {
            bail!(
                "frc{} played {total} matches in {}, none at {event}",
                config.team,
                config.year
            );
        }
    }
    if matches.is_empty() {
        bail!(
            "no matches for frc{} in {} — check the team number and year",
            config.team,
            config.year
        );
    }
    tba::sort_matches(&mut matches);

    let json_path = paths.matches_json(config.team, config.year);
    if let Some(parent) = json_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(
        &json_path,
        serde_json::to_vec_pretty(&season.raw).context("encoding match JSON")?,
    )
    .with_context(|| format!("writing {}", json_path.display()))?;

    Ok(Fetched { matches, json_path })
}

/// One match video on disk.
#[derive(Debug, Clone)]
pub struct Download {
    pub match_key: String,
    pub path: PathBuf,
    pub source: String,
}

/// Work items that succeeded, plus the ones that did not and why.
#[derive(Debug)]
pub struct Outcome<T> {
    pub ok: Vec<T>,
    pub failed: Vec<(String, String)>,
}

impl<T> Default for Outcome<T> {
    fn default() -> Self {
        Self {
            ok: Vec::new(),
            failed: Vec::new(),
        }
    }
}

/// Download one video per match, skipping files that are already there.
pub fn download_match_videos(
    config: &CamcorderConfig,
    paths: &Paths,
    matches: &[Match],
    force: bool,
) -> Result<Outcome<Download>> {
    let mut outcome = Outcome::default();

    for m in matches {
        let Some(video) = m.preferred_video() else {
            outcome
                .failed
                .push((m.key.clone(), "no video recorded on TBA".to_string()));
            continue;
        };
        let dest = paths.video_file(config.team, &m.key);
        if dest.is_file() && !force && video::probe(&dest).is_ok() {
            outcome.ok.push(Download {
                match_key: m.key.clone(),
                path: dest,
                source: format!("cached ({})", video.describe()),
            });
            continue;
        }

        match fetch_one(&video, &dest) {
            Ok(path) => outcome.ok.push(Download {
                match_key: m.key.clone(),
                path,
                source: video.describe(),
            }),
            Err(error) => outcome.failed.push((m.key.clone(), format!("{error:#}"))),
        }
    }

    Ok(outcome)
}

fn fetch_one(video: &VideoRef, dest: &Path) -> Result<PathBuf> {
    match video {
        VideoRef::YouTube(id) => video::download_youtube(id, dest),
        VideoRef::Direct(url) => video::download_direct(url, dest),
    }
}

/// Split one downloaded match video into its individual views.
pub fn split_video(
    input: &Path,
    dest: &Path,
    config: &CamcorderConfig,
    force: bool,
) -> Result<SplitReport> {
    video::split_views(input, dest, &split_options(config, force))
}

pub fn split_options(config: &CamcorderConfig, force: bool) -> SplitOptions {
    SplitOptions {
        mode: config.video.mode,
        layout: config.video.layout,
        scene_threshold: config.video.scene_threshold,
        scene_min_len: config.video.scene_min_len,
        crf: config.video.crf,
        preset: config.video.preset.clone(),
        force,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_paths_are_grouped_by_team() {
        let paths = Paths::new("output");
        assert_eq!(
            paths.matches_json(254, 2025),
            PathBuf::from("output/frc254/matches-2025.json")
        );
        assert_eq!(
            paths.video_file(254, "2025casj_qm1"),
            PathBuf::from("output/frc254/videos/2025casj_qm1.mp4")
        );
        assert_eq!(
            paths.views_dir(254, "2025casj_qm1"),
            PathBuf::from("output/frc254/views/2025casj_qm1")
        );
    }

    #[test]
    fn split_options_come_from_the_config() {
        let config: CamcorderConfig = toml::from_str(
            "team = 254\nyear = 2025\n[video]\nmode = \"tiled\"\nlayout = \"2x2\"\ncrf = 20\n",
        )
        .expect("parses");
        let options = split_options(&config, true);

        assert_eq!(options.crf, 20);
        assert!(options.force);
        assert!(!options.mode.splits_scenes());
        assert_eq!(
            options.layout,
            crate::config::Layout::Grid(video::Grid::new(2, 2).unwrap())
        );
    }
}

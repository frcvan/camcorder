//! Configuration: team/event/video settings from `camcorder.toml`, credentials from `.env`.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::video::Grid;

/// Default config file name, looked up in the working directory.
pub const DEFAULT_CONFIG_FILE: &str = "camcorder.toml";
/// Environment variable holding the The Blue Alliance read API key.
pub const API_KEY_ENV: &str = "TBA_API_KEY";
/// Default output directory: a temp directory (`$TMPDIR/camcorder`), so a run does
/// not leave a season of match video behind. Override it in `camcorder.toml` or
/// with `--output-dir` to keep the files somewhere permanent.
pub fn default_output_dir() -> PathBuf {
    std::env::temp_dir().join("camcorder")
}

/// Everything the program needs to know before it talks to The Blue Alliance.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CamcorderConfig {
    /// FRC team number, e.g. `254`.
    pub team: u32,
    /// Season to pull matches from.
    pub year: u16,
    /// Optional event key (`2025casj`); narrows the run to one event.
    #[serde(default)]
    pub event: Option<String>,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default)]
    pub video: VideoConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Root directory for match JSON, downloaded video and split views.
    pub dir: PathBuf,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: default_output_dir(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoConfig {
    /// Whether to crop tiled views, split on scene changes, or both.
    pub mode: SplitMode,
    /// Camera layout, or `auto` to detect it from the frame.
    pub layout: Layout,
    /// ffmpeg scene score above which a frame starts a new segment.
    pub scene_threshold: f32,
    /// Cuts that would make a shorter segment than this are ignored.
    pub scene_min_len: f64,
    /// x264 CRF for split outputs.
    pub crf: u8,
    /// x264 preset for split outputs.
    pub preset: String,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            mode: SplitMode::Both,
            layout: Layout::Auto,
            scene_threshold: 0.4,
            scene_min_len: 2.0,
            crf: 18,
            preset: "veryfast".to_string(),
        }
    }
}

/// What the video pipeline should do with a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitMode {
    /// Crop each tiled view out of the frame, then cut each view into scenes.
    #[default]
    Both,
    /// Crop each tiled view out of the frame only.
    Tiled,
    /// Split on scene changes only, never cropping.
    Scenes,
}

impl SplitMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SplitMode::Both => "both",
            SplitMode::Tiled => "tiled",
            SplitMode::Scenes => "scenes",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "both" => Ok(SplitMode::Both),
            "tiled" => Ok(SplitMode::Tiled),
            "scenes" => Ok(SplitMode::Scenes),
            other => bail!("unknown video.mode {other:?}; expected \"both\", \"tiled\" or \"scenes\""),
        }
    }

    /// Does this mode crop camera views out of the frame?
    pub fn crops_views(self) -> bool {
        !matches!(self, SplitMode::Scenes)
    }

    /// Does this mode cut the stream at scene changes?
    pub fn splits_scenes(self) -> bool {
        !matches!(self, SplitMode::Tiled)
    }
}

impl fmt::Display for SplitMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SplitMode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        SplitMode::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// Camera layout of a single frame: detected, or pinned in the config file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Scan the frame for separators between views.
    #[default]
    Auto,
    /// A fixed rows x cols grid.
    Grid(Grid),
}

impl Layout {
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim().to_ascii_lowercase();
        if value == "auto" {
            return Ok(Layout::Auto);
        }
        Grid::parse(&value)
            .map(Layout::Grid)
            .with_context(|| format!("unknown video.layout {value:?}; expected \"auto\" or e.g. \"2x2\""))
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Layout::Auto => f.write_str("auto"),
            Layout::Grid(grid) => write!(f, "{grid}"),
        }
    }
}

impl<'de> Deserialize<'de> for Layout {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Layout::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// Command-line values that win over the config file.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub team: Option<u32>,
    pub year: Option<u16>,
    pub event: Option<String>,
    pub output_dir: Option<PathBuf>,
}

impl CamcorderConfig {
    /// Read and check a config file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| {
            format!(
                "reading config {} — copy the sample {} from the repository, or pass --config",
                path.display(),
                DEFAULT_CONFIG_FILE
            )
        })?;
        let config: CamcorderConfig = toml::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn apply_overrides(&mut self, overrides: &Overrides) {
        if let Some(team) = overrides.team {
            self.team = team;
        }
        if let Some(year) = overrides.year {
            self.year = year;
        }
        if let Some(event) = &overrides.event {
            self.event = Some(event.clone());
        }
        if let Some(dir) = &overrides.output_dir {
            self.output.dir = dir.clone();
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.team == 0 || self.team > 99_999 {
            bail!("team must be a real FRC team number, got {}", self.team);
        }
        if !(1992..=2100).contains(&self.year) {
            bail!("year must be between 1992 and 2100, got {}", self.year);
        }
        if let Some(event) = &self.event {
            if event.trim().is_empty() {
                bail!("event must not be empty (remove the key to pull the whole season)");
            }
        }
        if !(self.video.scene_threshold > 0.0 && self.video.scene_threshold <= 1.0) {
            bail!(
                "video.scene_threshold must be in (0.0, 1.0], got {}",
                self.video.scene_threshold
            );
        }
        if self.video.scene_min_len < 0.0 {
            bail!(
                "video.scene_min_len must not be negative, got {}",
                self.video.scene_min_len
            );
        }
        if self.video.crf > 51 {
            bail!("video.crf must be between 0 and 51, got {}", self.video.crf);
        }
        if self.video.preset.trim().is_empty() {
            bail!("video.preset must not be empty");
        }
        Ok(())
    }
}

/// Read `TBA_API_KEY`, loading `.env` next to the config file first, then `./.env`.
pub fn load_api_key(config_path: &Path) -> Result<String> {
    if let Some(dir) = config_path.parent() {
        let candidate = dir.join(".env");
        if candidate.is_file() {
            let _ = dotenvy::from_path(&candidate);
        }
    }
    let _ = dotenvy::dotenv();

    match std::env::var(API_KEY_ENV) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        _ => bail!(
            "{API_KEY_ENV} is not set — add it to a .env file (see .env.example); \
             keys are free at https://www.thebluealliance.com/account"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
team = 254
year = 2025
event = "2025casj"

[output]
dir = "out"

[video]
mode = "scenes"
layout = "2x3"
scene_threshold = 0.5
scene_min_len = 3.0
crf = 21
preset = "fast"
"#;

    #[test]
    fn parses_a_full_config() {
        let config: CamcorderConfig = toml::from_str(SAMPLE).expect("config parses");
        config.validate().expect("config is valid");

        assert_eq!(config.team, 254);
        assert_eq!(config.year, 2025);
        assert_eq!(config.event.as_deref(), Some("2025casj"));
        assert_eq!(config.output.dir, PathBuf::from("out"));
        assert_eq!(config.video.mode, SplitMode::Scenes);
        assert_eq!(config.video.layout, Layout::Grid(Grid::new(2, 3).unwrap()));
        assert_eq!(config.video.crf, 21);
        assert!(!config.video.mode.crops_views());
        assert!(config.video.mode.splits_scenes());
    }

    #[test]
    fn default_output_dir_is_a_temp_dir() {
        let dir = default_output_dir();
        assert!(dir.is_absolute(), "got {}", dir.display());
        assert!(dir.starts_with(std::env::temp_dir()), "got {}", dir.display());
    }

    #[test]
    fn only_team_and_year_are_required() {
        let config: CamcorderConfig = toml::from_str("team = 1678\nyear = 2024\n").expect("parses");
        config.validate().expect("valid");

        assert_eq!(config.event, None);
        assert_eq!(config.output.dir, default_output_dir());
        assert_eq!(config.video.mode, SplitMode::Both);
        assert_eq!(config.video.layout, Layout::Auto);
        assert_eq!(config.video.scene_threshold, 0.4);
        assert_eq!(config.video.preset, "veryfast");
    }

    #[test]
    fn rejects_typos_instead_of_ignoring_them() {
        let err = toml::from_str::<CamcorderConfig>("team = 254\nyear = 2025\nteam_number = 1\n")
            .expect_err("unknown key is rejected");
        assert!(err.to_string().contains("team_number"), "got: {err}");
    }

    #[test]
    fn rejects_bad_values() {
        let config: CamcorderConfig =
            toml::from_str("team = 254\nyear = 1990\n").expect("parses as toml");
        assert!(config.validate().is_err());

        let error = toml::from_str::<CamcorderConfig>(
            "team = 254\nyear = 2025\n[video]\nmode = \"split\"\n",
        )
        .expect_err("bad mode is rejected");
        assert!(error.to_string().contains("unknown video.mode"), "got: {error}");
    }

    #[test]
    fn layout_accepts_auto_and_grids() {
        assert_eq!(Layout::parse("auto").unwrap(), Layout::Auto);
        assert_eq!(Layout::parse("1x1").unwrap(), Layout::Grid(Grid::SINGLE));
        assert_eq!(Layout::parse(" 2X2 ").unwrap(), Layout::Grid(Grid::new(2, 2).unwrap()));
        assert!(Layout::parse("two").is_err());
    }

    #[test]
    fn overrides_win_over_the_file() {
        let mut config: CamcorderConfig = toml::from_str(SAMPLE).expect("parses");
        config.apply_overrides(&Overrides {
            team: Some(1678),
            year: Some(2024),
            event: None,
            output_dir: Some(PathBuf::from("elsewhere")),
        });
        assert_eq!(config.team, 1678);
        assert_eq!(config.year, 2024);
        assert_eq!(config.event.as_deref(), Some("2025casj"));
        assert_eq!(config.output.dir, PathBuf::from("elsewhere"));
    }
}

//! The Blue Alliance API client and match models.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{fmt::Write as _, time::Duration};

pub const DEFAULT_BASE_URL: &str = "https://www.thebluealliance.com/api/v3";
/// Override the API host, e.g. to point at a mirror or a stub server.
pub const BASE_URL_ENV: &str = "TBA_BASE_URL";

/// Read-only client for the TBA v3 API.
pub struct TbaClient {
    http: reqwest::blocking::Client,
    base: String,
    key: String,
}

impl TbaClient {
    pub fn new(key: impl Into<String>) -> Result<Self> {
        let base = std::env::var(BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Self::with_base(key, base)
    }

    pub fn with_base(key: impl Into<String>, base: impl Into<String>) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("camcorder/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("building the HTTP client")?;
        Ok(Self {
            http,
            base: base.into().trim_end_matches('/').to_string(),
            key: key.into(),
        })
    }

    /// Every match a team played in a season, with full score breakdowns and videos.
    pub fn team_matches(&self, team: u32, year: u16) -> Result<TeamMatches> {
        let path = format!("/team/{}/matches/{year}", team_key(team));
        let body = self.get(&path)?;
        let raw: serde_json::Value =
            serde_json::from_str(&body).with_context(|| format!("decoding JSON from {path}"))?;
        let matches: Vec<Match> = serde_json::from_value(raw.clone())
            .with_context(|| format!("decoding matches from {path}"))?;
        Ok(TeamMatches { matches, raw })
    }

    fn get(&self, path: &str) -> Result<String> {
        let url = format!("{}{}", self.base, path);
        let response = self
            .http
            .get(&url)
            .header("X-TBA-Auth-Key", &self.key)
            .header("Accept", "application/json")
            .send()
            .with_context(|| format!("GET {url}"))?;

        let status = response.status();
        if !status.is_success() {
            let hint = match status.as_u16() {
                401 => " — check TBA_API_KEY in your .env",
                404 => " — unknown team or season",
                _ => "",
            };
            bail!("The Blue Alliance returned {status} for {url}{hint}");
        }
        response
            .text()
            .with_context(|| format!("reading body of {url}"))
    }
}

/// A season of matches plus the untouched API response, so nothing is lost when it is saved.
pub struct TeamMatches {
    pub matches: Vec<Match>,
    pub raw: serde_json::Value,
}

pub fn team_key(team: u32) -> String {
    format!("frc{team}")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Match {
    pub key: String,
    #[serde(default)]
    pub comp_level: String,
    #[serde(default)]
    pub set_number: u32,
    #[serde(default)]
    pub match_number: u32,
    #[serde(default)]
    pub event_key: String,
    #[serde(default)]
    pub time: Option<i64>,
    #[serde(default)]
    pub actual_time: Option<i64>,
    #[serde(default)]
    pub predicted_time: Option<i64>,
    #[serde(default)]
    pub post_result_time: Option<i64>,
    #[serde(default)]
    pub winning_alliance: String,
    #[serde(default)]
    pub alliances: Alliances,
    #[serde(default)]
    pub videos: Vec<Video>,
    /// Year-specific detail. Kept as raw JSON because it differs every season
    /// (and is an array for matches with more than one breakdown).
    #[serde(default)]
    pub score_breakdown: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Alliances {
    #[serde(default)]
    pub red: Alliance,
    #[serde(default)]
    pub blue: Alliance,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Alliance {
    /// -1 when the match has not been played yet.
    #[serde(default)]
    pub score: i64,
    #[serde(default)]
    pub team_keys: Vec<String>,
    #[serde(default)]
    pub surrogate_team_keys: Vec<String>,
    #[serde(default)]
    pub dq_team_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Video {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub key: Option<String>,
}

/// A video we know how to fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoRef {
    /// A YouTube video id, fetched with yt-dlp.
    YouTube(String),
    /// A direct file URL, fetched over HTTP.
    Direct(String),
}

impl VideoRef {
    pub fn url(&self) -> String {
        match self {
            VideoRef::YouTube(id) => format!("https://www.youtube.com/watch?v={id}"),
            VideoRef::Direct(url) => url.clone(),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            VideoRef::YouTube(id) => format!("youtube/{id}"),
            VideoRef::Direct(url) => format!("direct/{url}"),
        }
    }
}

impl Video {
    pub fn resolve(&self) -> Option<VideoRef> {
        let key = self.key.as_deref()?.trim();
        if key.is_empty() {
            return None;
        }
        match self.kind.as_str() {
            "youtube" => Some(VideoRef::YouTube(key.to_string())),
            // TBA-hosted videos point straight at a file.
            "tba" => key
                .starts_with("http")
                .then(|| VideoRef::Direct(key.to_string())),
            _ if key.starts_with("http") => Some(VideoRef::Direct(key.to_string())),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllianceColor {
    Red,
    Blue,
}

impl AllianceColor {
    pub fn as_str(self) -> &'static str {
        match self {
            AllianceColor::Red => "red",
            AllianceColor::Blue => "blue",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Loss,
    Tie,
    /// Match has not been played yet.
    Unknown,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Win => "W",
            Outcome::Loss => "L",
            Outcome::Tie => "T",
            Outcome::Unknown => "-",
        }
    }
}

impl Match {
    pub fn alliance(&self, color: AllianceColor) -> &Alliance {
        match color {
            AllianceColor::Red => &self.alliances.red,
            AllianceColor::Blue => &self.alliances.blue,
        }
    }

    pub fn score(&self, color: AllianceColor) -> i64 {
        self.alliance(color).score
    }

    /// Both alliance scores are recorded (TBA uses -1 for unplayed matches).
    pub fn is_played(&self) -> bool {
        self.alliances.red.score >= 0 && self.alliances.blue.score >= 0
    }

    pub fn color_of(&self, team: u32) -> Option<AllianceColor> {
        let key = team_key(team);
        if self.alliances.red.team_keys.contains(&key) {
            Some(AllianceColor::Red)
        } else if self.alliances.blue.team_keys.contains(&key) {
            Some(AllianceColor::Blue)
        } else {
            None
        }
    }

    pub fn opponent(&self, team: u32) -> Option<AllianceColor> {
        self.color_of(team).map(|color| match color {
            AllianceColor::Red => AllianceColor::Blue,
            AllianceColor::Blue => AllianceColor::Red,
        })
    }

    pub fn our_score(&self, team: u32) -> Option<i64> {
        self.color_of(team).map(|color| self.score(color))
    }

    pub fn opponent_score(&self, team: u32) -> Option<i64> {
        self.opponent(team).map(|color| self.score(color))
    }

    pub fn outcome_for(&self, team: u32) -> Option<Outcome> {
        let us = self.color_of(team)?;
        if !self.is_played() {
            return Some(Outcome::Unknown);
        }
        let opponent = match us {
            AllianceColor::Red => AllianceColor::Blue,
            AllianceColor::Blue => AllianceColor::Red,
        };
        Some(match self.score(us).cmp(&self.score(opponent)) {
            std::cmp::Ordering::Greater => Outcome::Win,
            std::cmp::Ordering::Less => Outcome::Loss,
            std::cmp::Ordering::Equal => Outcome::Tie,
        })
    }

    /// Score breakdown for our alliance, when TBA has one.
    pub fn breakdown_for(&self, team: u32) -> Option<&serde_json::Value> {
        let color = self.color_of(team)?;
        let breakdown = self.score_breakdown.as_ref()?;
        let per_alliance = match breakdown {
            serde_json::Value::Array(entries) => entries.first()?,
            other => other,
        };
        per_alliance.get(color.as_str())
    }

    /// `qm12`, `sf3m2`, … as FRC fans write it.
    pub fn display_name(&self) -> String {
        match self.comp_level.as_str() {
            "qm" => format!("qm{}", self.match_number),
            "" => self.key.clone(),
            level => format!("{level}{}m{}", self.set_number, self.match_number),
        }
    }

    /// First usable video, preferring YouTube (highest quality source).
    pub fn preferred_video(&self) -> Option<VideoRef> {
        let resolved: Vec<VideoRef> = self.videos.iter().filter_map(Video::resolve).collect();
        resolved
            .iter()
            .find(|video| matches!(video, VideoRef::YouTube(_)))
            .or_else(|| resolved.first())
            .cloned()
    }
}

/// Sort matches the way a schedule reads: by time, then by bracket depth.
pub fn sort_matches(matches: &mut [Match]) {
    fn level_rank(level: &str) -> u8 {
        match level {
            "qm" => 0,
            "ef" => 1,
            "qf" => 2,
            "sf" => 3,
            "f" => 4,
            _ => 5,
        }
    }
    matches.sort_by(|a, b| {
        a.time
            .unwrap_or(i64::MAX)
            .cmp(&b.time.unwrap_or(i64::MAX))
            .then_with(|| a.event_key.cmp(&b.event_key))
            .then_with(|| level_rank(&a.comp_level).cmp(&level_rank(&b.comp_level)))
            .then_with(|| a.set_number.cmp(&b.set_number))
            .then_with(|| a.match_number.cmp(&b.match_number))
    });
}

/// Human-readable table of every match plus season totals.
pub fn render_summary(matches: &[Match], team: u32, year: u16) -> String {
    let mut out = String::new();

    let ours: Vec<&Match> = matches
        .iter()
        .filter(|m| m.color_of(team).is_some())
        .collect();
    let played: Vec<&Match> = ours.iter().copied().filter(|m| m.is_played()).collect();

    let count = |outcome: Outcome| {
        played
            .iter()
            .filter(|m| m.outcome_for(team) == Some(outcome))
            .count()
    };
    let (wins, losses, ties) = (
        count(Outcome::Win),
        count(Outcome::Loss),
        count(Outcome::Tie),
    );

    let _ = writeln!(
        out,
        "frc{team} · {year} · {} matches · {} played · {wins}W-{losses}L-{ties}T",
        ours.len(),
        played.len()
    );
    let _ = writeln!(
        out,
        "{:<10} {:<12} {:>3} {:<4} {:>8} {:>14}  video",
        "match", "event", "all", "res", "score", "our/opp"
    );

    for m in &ours {
        let color = m.color_of(team).map(|c| c.as_str()).unwrap_or("?");
        let outcome = m.outcome_for(team).map(Outcome::as_str).unwrap_or("-");
        let (us, them) = (m.our_score(team), m.opponent_score(team));
        let score = match (us, them) {
            (Some(us), Some(them)) => format!("{us}-{them}"),
            _ => "-".to_string(),
        };
        let pair = match (us, them) {
            (Some(us), Some(them)) => format!("{us}/{them}"),
            _ => "-".to_string(),
        };
        let video = m
            .preferred_video()
            .map(|v| v.describe())
            .unwrap_or_else(|| "none".to_string());
        let breakdown = if m.breakdown_for(team).is_some() {
            ""
        } else {
            "  [no breakdown]"
        };
        let _ = writeln!(
            out,
            "{:<10} {:<12} {:>3} {:<4} {:>8} {:>14}  {video}{breakdown}",
            m.display_name(),
            m.event_key,
            color,
            outcome,
            score,
            pair
        );
    }

    fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
        let values: Vec<f64> = values.into_iter().collect();
        if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }
    let avg_our = mean(
        played
            .iter()
            .filter_map(|m| m.our_score(team))
            .map(|s| s as f64),
    );
    let avg_opp = mean(
        played
            .iter()
            .filter_map(|m| m.opponent_score(team))
            .map(|s| s as f64),
    );
    if !avg_our.is_nan() {
        let _ = writeln!(
            out,
            "averages: our {avg_our:.1} · opponent {avg_opp:.1} (over {} played)",
            played.len()
        );
    }

    let no_video = ours
        .iter()
        .filter(|m| m.preferred_video().is_none())
        .count();
    let no_breakdown = ours
        .iter()
        .filter(|m| m.breakdown_for(team).is_none())
        .count();
    let _ = writeln!(
        out,
        "coverage: {}/{} with video · {}/{} with score breakdown",
        ours.len() - no_video,
        ours.len(),
        ours.len() - no_breakdown,
        ours.len()
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matches() -> Vec<Match> {
        // A trimmed slice of a real /team/frc254/matches/2025 response.
        let json = serde_json::json!([
            {
                "key": "2025casj_qm1",
                "comp_level": "qm",
                "set_number": 1,
                "match_number": 1,
                "event_key": "2025casj",
                "time": 1744588800,
                "winning_alliance": "red",
                "alliances": {
                    "red": { "score": 123, "team_keys": ["frc254", "frc1678", "frc971"],
                             "surrogate_team_keys": [], "dq_team_keys": [] },
                    "blue": { "score": 45, "team_keys": ["frc1", "frc2", "frc3"],
                              "surrogate_team_keys": [], "dq_team_keys": [] }
                },
                "videos": [
                    { "type": "youtube", "key": "AbCdEf12345" },
                    { "type": "tba", "key": "https://example.com/full.mp4" }
                ],
                "score_breakdown": {
                    "red": { "totalPoints": 123 },
                    "blue": { "totalPoints": 45 }
                }
            },
            {
                "key": "2025casj_qm2",
                "comp_level": "qm",
                "set_number": 1,
                "match_number": 2,
                "event_key": "2025casj",
                "time": 1744589100,
                "winning_alliance": "blue",
                "alliances": {
                    "red": { "score": 60, "team_keys": ["frc254", "frc1", "frc2"],
                             "surrogate_team_keys": [], "dq_team_keys": [] },
                    "blue": { "score": 60, "team_keys": ["frc3", "frc4", "frc5"],
                              "surrogate_team_keys": [], "dq_team_keys": [] }
                },
                "videos": [],
                "score_breakdown": null
            },
            {
                "key": "2025casj_sf1m1",
                "comp_level": "sf",
                "set_number": 1,
                "match_number": 1,
                "event_key": "2025casj",
                "time": 1744589700,
                "winning_alliance": "",
                "alliances": {
                    "red": { "score": -1, "team_keys": ["frc254", "frc1678", "frc971"] },
                    "blue": { "score": -1, "team_keys": ["frc1", "frc2", "frc3"] }
                },
                "videos": [],
                "score_breakdown": null
            }
        ]);
        serde_json::from_value(json).expect("sample matches deserialize")
    }

    #[test]
    fn decodes_full_match_stats() {
        let matches = sample_matches();
        let first = &matches[0];

        assert_eq!(first.display_name(), "qm1");
        assert_eq!(first.event_key, "2025casj");
        assert_eq!(first.color_of(254), Some(AllianceColor::Red));
        assert_eq!(first.opponent(254), Some(AllianceColor::Blue));
        assert_eq!(first.our_score(254), Some(123));
        assert_eq!(first.opponent_score(254), Some(45));
        assert_eq!(first.outcome_for(254), Some(Outcome::Win));
        assert_eq!(
            first.breakdown_for(254).unwrap(),
            &serde_json::json!({ "totalPoints": 123 })
        );
        assert_eq!(first.alliances.red.team_keys.len(), 3);
    }

    #[test]
    fn handles_ties_and_unplayed_matches() {
        let matches = sample_matches();
        assert_eq!(matches[1].outcome_for(254), Some(Outcome::Tie));
        assert!(!matches[2].is_played());
        assert_eq!(matches[2].outcome_for(254), Some(Outcome::Unknown));
        assert_eq!(matches[2].score(AllianceColor::Red), -1);
    }

    #[test]
    fn picks_the_youtube_video_first() {
        let matches = sample_matches();
        assert_eq!(
            matches[0].preferred_video(),
            Some(VideoRef::YouTube("AbCdEf12345".to_string()))
        );
        assert_eq!(matches[1].preferred_video(), None);
    }

    #[test]
    fn resolves_direct_and_rejects_empty_videos() {
        let youtube = Video {
            kind: "youtube".into(),
            key: Some("abc".into()),
        };
        assert_eq!(youtube.resolve(), Some(VideoRef::YouTube("abc".into())));
        assert_eq!(
            youtube.resolve().unwrap().url(),
            "https://www.youtube.com/watch?v=abc"
        );

        let tba = Video {
            kind: "tba".into(),
            key: Some("https://example.com/a.mp4".into()),
        };
        assert_eq!(
            tba.resolve(),
            Some(VideoRef::Direct("https://example.com/a.mp4".into()))
        );

        let not_a_url = Video {
            kind: "tba".into(),
            key: Some("a.mp4".into()),
        };
        assert_eq!(not_a_url.resolve(), None);

        let empty = Video {
            kind: "youtube".into(),
            key: None,
        };
        assert_eq!(empty.resolve(), None);
    }

    #[test]
    fn summary_includes_totals_and_gaps() {
        let mut matches = sample_matches();
        sort_matches(&mut matches);
        let summary = render_summary(&matches, 254, 2025);

        assert!(summary.contains("frc254 · 2025 · 3 matches · 2 played · 1W-0L-1T"));
        assert!(summary.contains("qm1"));
        assert!(summary.contains("sf1m1"));
        assert!(summary.contains("youtube/AbCdEf12345"));
        assert!(summary.contains("[no breakdown]"));
        assert!(summary.contains("1/3 with video"));
    }
}

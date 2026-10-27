//! End-to-end tests for the whole program, run against a stub of the TBA API.
//!
//! These drive the real binary: `camcorder.toml` and `.env` are read from a temp
//! working directory, the match JSON comes from a local HTTP server, the
//! "TBA-hosted" video is a real mp4 served by that same server, and the split
//! views are checked on disk. The only part not covered is yt-dlp reaching
//! YouTube.

use std::{
    io::{BufRead as _, BufReader, Write as _},
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc::Receiver,
    thread,
    time::Duration,
};

use camcorder::video;

/// A stub of `https://www.thebluealliance.com/api/v3`.
struct Stub {
    base_url: String,
    requests: Receiver<String>,
}

/// Bind a port, let the caller build routes knowing the address, then serve
/// `limit` requests and stop.
fn start_stub(
    limit: usize,
    build_routes: impl FnOnce(&str) -> Vec<(String, u16, Vec<u8>)>,
) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a port");
    let address = listener.local_addr().expect("local address");
    let base_url = format!("http://{address}/api/v3");
    let routes = build_routes(&base_url);
    let (sender, requests) = std::sync::mpsc::channel();

    thread::spawn(move || {
        for stream in listener.incoming().take(limit) {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));

            let mut head = String::new();
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if line == "\r\n" || line == "\n" {
                            break;
                        }
                        head.push_str(&line);
                    }
                    Err(_) => break,
                }
            }

            let path = head
                .lines()
                .next()
                .and_then(|request| request.split_whitespace().nth(1))
                .unwrap_or("/")
                .to_string();
            let _ = sender.send(head);

            let (status, body) = routes
                .iter()
                .find(|(route, _, _)| route == &path)
                .map(|(_, status, body)| (*status, body.clone()))
                .unwrap_or((404, b"not found".to_vec()));

            let reason = match status {
                200 => "OK",
                401 => "Unauthorized",
                _ => "Not Found",
            };
            let header = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/octet-stream\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
        }
    });

    Stub { base_url, requests }
}

impl Stub {
    /// Request heads seen so far, waiting briefly for any still in flight.
    fn seen_requests(&self) -> Vec<String> {
        let mut heads = Vec::new();
        while let Ok(head) = self.requests.recv_timeout(Duration::from_secs(10)) {
            heads.push(head);
        }
        heads
    }
}

fn ffmpeg(args: &[&str]) {
    let output = Command::new("ffmpeg")
        .args(args)
        .output()
        .expect("ffmpeg is required to build test fixtures");
    assert!(
        output.status.success(),
        "ffmpeg {args:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A plain single-view clip, standing in for a real match video: detailed and
/// always moving, with no straight edges or flat patches to mistake for dividers.
fn build_video(dest: &Path) {
    ffmpeg(&[
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "mandelbrot=s=1280x720:r=30",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        dest.to_str().expect("utf-8 path"),
    ]);
}

/// Two matches: one played with a TBA-hosted video, one not played and without
/// a video.
fn matches_json(video_url: &str) -> Vec<u8> {
    serde_json::json!([
        {
            "key": "2025test_qm1",
            "comp_level": "qm",
            "set_number": 1,
            "match_number": 1,
            "event_key": "2025test",
            "time": 1744588800,
            "winning_alliance": "red",
            "alliances": {
                "red": { "score": 123, "team_keys": ["frc254", "frc1678", "frc971"] },
                "blue": { "score": 45, "team_keys": ["frc1", "frc2", "frc3"] }
            },
            "videos": [{ "type": "tba", "key": video_url }],
            "score_breakdown": {
                "red": { "totalPoints": 123 },
                "blue": { "totalPoints": 45 }
            }
        },
        {
            "key": "2025test_qm2",
            "comp_level": "qm",
            "set_number": 1,
            "match_number": 2,
            "event_key": "2025test",
            "time": 1744589100,
            "winning_alliance": "",
            "alliances": {
                "red": { "score": -1, "team_keys": ["frc254", "frc1", "frc2"] },
                "blue": { "score": -1, "team_keys": ["frc3", "frc4", "frc5"] }
            },
            "videos": [],
            "score_breakdown": null
        }
    ])
    .to_string()
    .into_bytes()
}

struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().expect("temp dir"),
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// The two files the program reads before it does anything: the config with
    /// the team number, and the API key in `.env`.
    fn write_inputs(&self) {
        std::fs::write(
            self.path("camcorder.toml"),
            "team = 254\nyear = 2025\n\n[output]\ndir = \"output\"\n",
        )
        .expect("write config");
        std::fs::write(self.path(".env"), "TBA_API_KEY=test-key\n").expect("write .env");
    }

    fn camcorder(&self, base_url: &str, args: &[&str]) -> std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_camcorder"))
            .args(args)
            .current_dir(self.dir.path())
            .env("TBA_BASE_URL", base_url)
            .output()
            .expect("run camcorder")
    }
}

#[test]
fn run_pulls_stats_downloads_the_video_and_splits_it() {
    let sandbox = Sandbox::new();
    sandbox.write_inputs();

    let source = sandbox.path("source.mp4");
    build_video(&source);
    let video_bytes = std::fs::read(&source).expect("read fixture video");

    let stub = start_stub(2, |base_url| {
        vec![
            (
                "/api/v3/team/frc254/matches/2025".to_string(),
                200,
                matches_json(&format!("{base_url}/video/full.mp4")),
            ),
            ("/api/v3/video/full.mp4".to_string(), 200, video_bytes),
        ]
    });

    let output = sandbox.camcorder(&stub.base_url, &["--verbose", "run"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "camcorder run failed:\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Stats for the team, read from the .env key against the stub.
    assert!(
        stdout.contains("frc254 · 2025 · 2 matches · 1 played · 1W-0L-0T"),
        "summary missing:\n{stdout}"
    );
    assert!(
        stdout.contains("2025test_qm1") && stdout.contains("direct/http"),
        "match row missing:\n{stdout}"
    );
    // --verbose dumps the raw breakdown for our alliance.
    assert!(
        stdout.contains("totalPoints"),
        "breakdown missing:\n{stdout}"
    );

    // …and the full response is saved untouched.
    let saved: serde_json::Value = serde_json::from_slice(
        &std::fs::read(sandbox.path("output/frc254/matches-2025.json")).expect("saved JSON"),
    )
    .expect("valid JSON");
    assert_eq!(saved.as_array().expect("array").len(), 2);

    // The stub was asked with the key from .env.
    let requests = stub.seen_requests();
    assert_eq!(
        requests.len(),
        2,
        "expected the matches and the video request"
    );
    let matches_request = requests
        .iter()
        .find(|head| head.contains("/team/frc254/matches/2025"))
        .expect("matches request");
    assert!(
        matches_request
            .to_ascii_lowercase()
            .contains("x-tba-auth-key: test-key"),
        "auth header missing from:\n{matches_request}"
    );

    // The match video was downloaded, and split into views.
    assert!(
        stderr.contains("[download] 1 video(s) ready · 1 without a downloadable video"),
        "download summary missing:\n{stderr}"
    );
    let downloaded = sandbox.path("output/frc254/videos/2025test_qm1.mp4");
    assert!(downloaded.is_file(), "video not downloaded");

    let views = sandbox.path("output/frc254/views/2025test_qm1");
    let view = views.join("seg01_view1.mp4");
    assert!(view.is_file(), "views not produced in {}", views.display());
    let info = video::probe(&view).expect("view probes");
    assert_eq!((info.width, info.height), (1280, 720));
    assert_eq!(
        std::fs::read_dir(&views).expect("views dir").count(),
        1,
        "a single-view clip should yield exactly one view"
    );
}

#[test]
fn a_rejected_api_key_is_explained() {
    let sandbox = Sandbox::new();
    sandbox.write_inputs();

    let stub = start_stub(1, |_base_url| {
        vec![(
            "/api/v3/team/frc254/matches/2025".to_string(),
            401,
            Vec::new(),
        )]
    });

    let output = sandbox.camcorder(&stub.base_url, &["stats"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "should fail on 401");
    assert!(
        stderr.contains("401") && stderr.contains("TBA_API_KEY"),
        "expected a hint about the key, got:\n{stderr}"
    );
}

#[test]
fn a_missing_api_key_is_explained() {
    let sandbox = Sandbox::new();
    std::fs::write(sandbox.path("camcorder.toml"), "team = 254\nyear = 2025\n")
        .expect("write config");

    let output = sandbox.camcorder("http://127.0.0.1:1/api/v3", &["stats"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "should fail without a key");
    assert!(
        stderr.contains("TBA_API_KEY"),
        "expected a hint about the key, got:\n{stderr}"
    );
}

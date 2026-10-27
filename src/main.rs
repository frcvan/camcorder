use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::Result;
use clap::{Parser, Subcommand};

use camcorder::{
    config::{self, CamcorderConfig, Overrides},
    pipeline::{self, Paths},
    tba::{self, Match, TbaClient},
};

#[derive(Parser, Debug)]
#[command(
    name = "camcorder",
    version,
    about = "Pull FRC match stats and match video from The Blue Alliance, then split each video into its individual views."
)]
struct Cli {
    /// Config file to read (team number, season, split settings).
    #[arg(long, short, global = true, value_name = "FILE",
          default_value = config::DEFAULT_CONFIG_FILE)]
    config: PathBuf,

    /// Team number; overrides the config file.
    #[arg(long, global = true)]
    team: Option<u32>,

    /// Season; overrides the config file.
    #[arg(long, global = true)]
    year: Option<u16>,

    /// Event key such as 2025casj; overrides the config file.
    #[arg(long, global = true)]
    event: Option<String>,

    /// Output directory; overrides the config file.
    /// Defaults to $TMPDIR/camcorder.
    #[arg(long, global = true, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Re-download and re-cut files that already exist.
    #[arg(long, global = true)]
    force: bool,

    /// Also print each match's raw score breakdown.
    #[arg(long, short, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Pull match stats for the configured team and print them.
    Stats,
    /// Download the match videos for the configured team.
    Download,
    /// Split a video file into its individual views.
    Views {
        /// Video file to process.
        video: PathBuf,
        /// Where to write the views (defaults to <output dir>/views/<file name>).
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
    },
    /// Estimate the camera's field-relative pose from the AprilTags in a video.
    Pose {
        /// Video file to measure.
        video: PathBuf,
        /// Bundled tag layout name, or a path to a WPILib field layout JSON.
        #[arg(long, value_name = "NAME|FILE", default_value = "2025-reefscape-welded")]
        tag_layout: String,
        /// Horizontal field of view of the camera, in degrees.
        #[arg(long, value_name = "DEG", default_value_t = 60.0)]
        fov: f64,
        /// Tag edge length, in metres.
        #[arg(long, value_name = "M", default_value_t = camcorder::pose::FRC_TAG_SIZE_M)]
        tag_size: f64,
        /// How many frames to measure.
        #[arg(long, value_name = "N", default_value_t = 10)]
        samples: usize,
        /// Only look at this part of the video, in seconds.
        #[arg(long, value_name = "SEC", num_args = 2, value_names = ["START", "END"])]
        window: Option<Vec<f64>>,
        /// Where to write the JSON report (defaults to <output dir>/poses/<stem>.json).
        #[arg(long, value_name = "FILE")]
        json: Option<PathBuf>,
    },
    /// Whole pipeline: stats, then video, then views. This is the default.
    Run,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let mut config = CamcorderConfig::load(&cli.config)?;
    config.apply_overrides(&Overrides {
        team: cli.team,
        year: cli.year,
        event: cli.event.clone(),
        output_dir: cli.output_dir.clone(),
    });
    config.validate()?;

    let paths = Paths::new(config.output.dir.clone());

    let command = cli.command.unwrap_or(Command::Run);
    if let Command::Pose {
        video,
        tag_layout,
        fov,
        tag_size,
        samples,
        window,
        json,
    } = &command
    {
        let dest = json
            .clone()
            .unwrap_or_else(|| default_pose_file(&paths, video));
        return measure_pose(video, tag_layout, *fov, *tag_size, *samples, window.clone(), &dest);
    }
    if let Command::Views { video, out } = &command {
        let dest = out
            .clone()
            .unwrap_or_else(|| default_views_dir(&paths, video));
        return split_one(&config, video, &dest, cli.force);
    }

    let api_key = config::load_api_key(&cli.config)?;
    let client = TbaClient::new(api_key)?;

    let fetched = pipeline::fetch_matches(&config, &client, &paths)?;
    print!(
        "{}",
        tba::render_summary(&fetched.matches, config.team, config.year)
    );
    if cli.verbose {
        print_breakdowns(&fetched.matches, config.team);
    }
    eprintln!(
        "[stats] {} matches · raw response written to {}",
        fetched.matches.len(),
        fetched.json_path.display()
    );

    if matches!(command, Command::Stats) {
        return Ok(());
    }

    let downloads =
        pipeline::download_match_videos(&config, &paths, &fetched.matches, cli.force)?;
    eprintln!(
        "[download] {} video(s) ready · {} without a downloadable video",
        downloads.ok.len(),
        downloads.failed.len()
    );
    for (match_key, reason) in &downloads.failed {
        eprintln!("  - {match_key}: {reason}");
    }

    if matches!(command, Command::Download) {
        return Ok(());
    }

    println!();
    let mut written = 0usize;
    let mut failures = 0usize;
    for download in &downloads.ok {
        let dest = paths.views_dir(config.team, &download.match_key);
        match pipeline::split_video(&download.path, &dest, &config, cli.force) {
            Ok(report) => {
                written += report.outputs.len();
                eprintln!(
                    "[views] {} ({} · {:.0}fps · {:.0}s) -> {} segment(s) · {} view(s) · {} new file(s)",
                    download.match_key,
                    format!("{}x{}", report.info.width, report.info.height),
                    report.info.fps,
                    report.info.duration,
                    report.segments.len(),
                    report.views(),
                    report.outputs.len(),
                );
                print_segments(&report);
            }
            Err(error) => {
                failures += 1;
                eprintln!("[views] {} failed: {error:#}", download.match_key);
            }
        }
    }
    eprintln!("[views] {written} file(s) written · {failures} video(s) failed");
    Ok(())
}

fn view_word(count: usize) -> String {
    if count == 1 {
        "1 view".to_string()
    } else {
        format!("{count} views")
    }
}

/// One line per segment: how the frame was divided for that piece of the stream.
fn print_segments(report: &camcorder::video::SplitReport) {
    for segment in &report.segments {
        eprintln!(
            "[views]   seg{:02}  {:>15}  {:<7} [{}]  {}",
            segment.segment.index + 1,
            format!(
                "{:.1}s-{:.1}s",
                segment.segment.start, segment.segment.end
            ),
            view_word(segment.layout.views()),
            segment.source,
            segment.layout.describe(),
        );
    }
}

fn split_one(config: &CamcorderConfig, input: &Path, dest: &Path, force: bool) -> Result<()> {
    let report = pipeline::split_video(input, dest, config, force)?;
    eprintln!(
        "[views] {} ({}x{} · {:.0}fps · {:.1}s) -> {} segment(s) · {} view(s)",
        input.display(),
        report.info.width,
        report.info.height,
        report.info.fps,
        report.info.duration,
        report.segments.len(),
        report.views(),
    );
    print_segments(&report);
    if report.is_single_view_throughout() && config.video.mode.crops_views() {
        eprintln!(
            "  note: no tiled views found; pin video.layout in {} if the frame really holds several views",
            config::DEFAULT_CONFIG_FILE
        );
    }
    eprintln!(
        "  wrote {} file(s) to {} · skipped {} existing",
        report.outputs.len(),
        dest.display(),
        report.skipped.len()
    );
    for path in &report.outputs {
        println!("{}", path.display());
    }
    Ok(())
}

fn default_pose_file(paths: &Paths, video: &Path) -> PathBuf {
    let stem = video
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("video");
    paths.root.join("poses").join(format!("{stem}.json"))
}

fn measure_pose(
    video: &Path,
    tag_layout: &str,
    fov: f64,
    tag_size: f64,
    samples: usize,
    window: Option<Vec<f64>>,
    dest: &Path,
) -> Result<()> {
    let options = camcorder::pose::PoseOptions {
        tag_layout: tag_layout.to_string(),
        horizontal_fov_deg: fov,
        tag_size_m: tag_size,
        samples,
        window: window.map(|window| (window[0], window[1])),
        ..camcorder::pose::PoseOptions::default()
    };
    let report = camcorder::pose::estimate(video, &options)?;
    eprintln!(
        "[pose] {} ({}x{} · {} tags seen · {} · fov {:.0}° · tag {:.1}cm)",
        video.display(),
        report.frame_width,
        report.frame_height,
        report.tags_seen.len(),
        report.tag_layout,
        report.horizontal_fov_deg,
        report.tag_size_m * 100.0,
    );
    print!("{}", report.render_table());
    if let Some(height) = report.median_height() {
        eprintln!("[pose] median camera height {height:.2}m");
    }
    if let Some(best) = report.best() {
        eprintln!(
            "[pose] best frame {:.1}s · rms {:.1}px · at ({:.2}, {:.2}, {:.2})m",
            best.time, best.rms_px, best.position[0], best.position[1], best.position[2]
        );
    }
    report.write_json(dest)?;
    eprintln!("  wrote {}", dest.display());
    Ok(())
}

fn default_views_dir(paths: &Paths, video: &Path) -> PathBuf {
    let stem = video
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("video");
    paths.root.join("views").join(stem)
}

fn print_breakdowns(matches: &[Match], team: u32) {
    println!("\nraw score breakdowns for {}:", tba::team_key(team));
    for m in matches {
        match m.breakdown_for(team) {
            Some(breakdown) => {
                let rendered = serde_json::to_string_pretty(breakdown).unwrap_or_default();
                println!("{}:\n{rendered}", m.key);
            }
            None => println!("{}: none on TBA", m.key),
        }
    }
}

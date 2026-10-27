//! AprilTag field layouts, straight from WPILib.
//!
//! The files under `assets/tags/` are byte-for-byte copies of WPILib's own
//! field layouts (`wpilibsuite/allwpilib`, `fields/src/main/native/resources/
//! org/wpilib/fields/frc/`); `assets/tags/README.md` records where each one came
//! from. Both of WPILib's schemas are read: the current field layout format
//! (`field-dimensions` and `field-tags`) and the older `AprilTagFieldLayout`
//! format (`field` and `tags`) that teams still hand around. Pass any other
//! layout file with `--tag-layout path/to.json`.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use super::types::{Mat3, Quat, Vec3};

/// Layouts compiled into the binary, by the name accepted on the command line.
pub const BUNDLED: &[(&str, &str)] = &[
    (
        "2025-reefscape-welded",
        include_str!("../../assets/tags/2025-reefscape-welded.json"),
    ),
    (
        "2025-reefscape-andymark",
        include_str!("../../assets/tags/2025-reefscape-andymark.json"),
    ),
    (
        "2026-rebuilt-welded",
        include_str!("../../assets/tags/2026-rebuilt-welded.json"),
    ),
    (
        "2026-rebuilt-andymark",
        include_str!("../../assets/tags/2026-rebuilt-andymark.json"),
    ),
];

/// Where one tag sits on the field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagPose {
    /// Centre of the tag face, metres, in the field frame.
    pub translation: Vec3,
    /// Tag axes to field axes, in WPILib's tag frame: `+x` out of the tag face,
    /// `+y` to the left when looking at the tag, `+z` up the tag.
    pub rotation: Quat,
}

/// A set of tags and the field they are on.
#[derive(Debug, Clone)]
pub struct TagLayout {
    pub name: String,
    pub field_length: f64,
    pub field_width: f64,
    tags: BTreeMap<u16, TagPose>,
}

impl TagLayout {
    /// Load a bundled layout by name, or a layout file by path.
    pub fn load(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() {
            bail!("tag layout must be a bundled name or a path");
        }
        if let Some((name, text)) = BUNDLED.iter().find(|(name, _)| *name == spec) {
            return Self::from_json(name, text)
                .with_context(|| format!("reading bundled layout {name}"));
        }
        let path = Path::new(spec);
        if path.is_file() {
            let text = fs::read_to_string(path)
                .with_context(|| format!("reading tag layout {}", path.display()))?;
            return Self::from_json(spec, &text)
                .with_context(|| format!("parsing tag layout {}", path.display()));
        }
        let names: Vec<&str> = BUNDLED.iter().map(|(name, _)| *name).collect();
        bail!(
            "unknown tag layout {spec:?} — use one of {} or a path to a WPILib \
             AprilTagFieldLayout JSON file",
            names.join(", ")
        )
    }

    /// Parse WPILib's `AprilTagFieldLayout` JSON.
    pub fn from_json(name: &str, text: &str) -> Result<Self> {
        let raw: RawLayout = serde_json::from_str(text).context("parsing JSON")?;
        if raw.tags.is_empty() {
            bail!("layout has no tags");
        }
        let mut tags = BTreeMap::new();
        for entry in raw.tags {
            let quaternion = entry.pose.rotation.quaternion;
            let rotation = Quat::new(quaternion.w, quaternion.x, quaternion.y, quaternion.z);
            // A zero quaternion cannot be normalised into anything useful. Note
            // that a perfectly good rotation can have `w` and `x` both zero
            // (180° about `y` or `z`), so only the norm is a valid check.
            let norm = (rotation.w * rotation.w
                + rotation.x * rotation.x
                + rotation.y * rotation.y
                + rotation.z * rotation.z)
                .sqrt();
            if norm.is_nan() || norm <= 1e-9 {
                bail!("tag {} has an empty rotation", entry.id);
            }
            tags.insert(
                entry.id,
                TagPose {
                    translation: Vec3::new(
                        entry.pose.translation.x,
                        entry.pose.translation.y,
                        entry.pose.translation.z,
                    ),
                    rotation: rotation.normalized(),
                },
            );
        }
        Ok(TagLayout {
            name: name.to_string(),
            field_length: raw.field.length,
            field_width: raw.field.width,
            tags,
        })
    }

    pub fn tag(&self, id: u16) -> Option<&TagPose> {
        self.tags.get(&id)
    }

    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }

    pub fn ids(&self) -> impl Iterator<Item = u16> + '_ {
        self.tags.keys().copied()
    }

    /// The four corners of a tag in field coordinates, ordered to match the
    /// detector: top-right, bottom-right, bottom-left, top-left as seen when
    /// facing the tag.
    ///
    /// WPILib's tag frame puts `+x` out of the tag face, `+y` to the left when
    /// looking at the tag and `+z` up the tag, so a corner "right and up" in the
    /// picture is `-y`, `+z` here. Getting this wrong does not fail loudly: the
    /// solve still fits every corner and quietly returns a pose balanced on the
    /// tag's edge.
    pub fn corners(&self, id: u16, tag_size: f64) -> Option<[Vec3; 4]> {
        let tag = self.tag(id)?;
        let half = tag_size * 0.5;
        let rotation = Mat3::from_quaternion(tag.rotation);
        let local = [
            Vec3::new(0.0, -half, half),
            Vec3::new(0.0, -half, -half),
            Vec3::new(0.0, half, -half),
            Vec3::new(0.0, half, half),
        ];
        Some(local.map(|corner| rotation.mul_vec(corner) + tag.translation))
    }
}

#[derive(Debug, Deserialize)]
struct RawLayout {
    #[serde(rename = "field-dimensions", alias = "field")]
    field: RawField,
    #[serde(rename = "field-tags", alias = "tags")]
    tags: Vec<RawTag>,
}

#[derive(Debug, Deserialize)]
struct RawField {
    length: f64,
    width: f64,
}

#[derive(Debug, Deserialize)]
struct RawTag {
    #[serde(rename = "ID")]
    id: u16,
    pose: RawPose,
}

#[derive(Debug, Deserialize)]
struct RawPose {
    translation: RawTranslation,
    rotation: RawRotation,
}

#[derive(Debug, Deserialize)]
struct RawTranslation {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Deserialize)]
struct RawRotation {
    quaternion: RawQuaternion,
}

#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct RawQuaternion {
    #[serde(rename = "W", alias = "w")]
    w: f64,
    #[serde(rename = "X", alias = "x")]
    x: f64,
    #[serde(rename = "Y", alias = "y")]
    y: f64,
    #[serde(rename = "Z", alias = "z")]
    z: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_layouts_parse_with_their_tags() {
        for (name, _) in BUNDLED {
            let layout = TagLayout::load(name).unwrap_or_else(|error| panic!("{name}: {error:#}"));
            assert!(
                layout.tag_count() >= 22,
                "{name} has {} tags",
                layout.tag_count()
            );
            assert!(layout.field_length > 10.0, "{name} length");
            assert!(layout.tag(1).is_some(), "{name} should have tag 1");
        }
    }

    #[test]
    fn the_2025_layout_matches_wpilib() {
        let layout = TagLayout::load("2025-reefscape-welded").expect("bundled layout");
        assert_eq!(layout.tag_count(), 22);
        assert!((layout.field_length - 17.548).abs() < 1e-3);
        assert!((layout.field_width - 8.052).abs() < 1e-3);

        let tag = layout.tag(22).expect("tag 22");
        assert!((tag.translation.x - 4.90474).abs() < 1e-4, "{tag:?}");
        assert!((tag.translation.y - 3.306318).abs() < 1e-4, "{tag:?}");
        assert!((tag.translation.z - 0.308102).abs() < 1e-4, "{tag:?}");
    }

    #[test]
    fn corners_are_a_tag_width_apart() {
        let layout = TagLayout::load("2025-reefscape-welded").expect("bundled layout");
        let size = 0.1651;
        for id in layout.ids() {
            let corners = layout.corners(id, size).expect("corners");
            for index in 0..4 {
                let next = (index + 1) % 4;
                let side = (corners[index] - corners[next]).norm();
                assert!(
                    (side - size).abs() < 1e-9,
                    "tag {id} side {index} is {side}, expected {size}"
                );
            }
            // Opposite corners are a diagonal apart, so the quad is square.
            let diagonal = (corners[0] - corners[2]).norm();
            assert!((diagonal - size * 2f64.sqrt()).abs() < 1e-9, "tag {id}");
        }
    }

    #[test]
    fn bundled_files_are_wpilibs_current_format() {
        // The files are copied straight from WPILib, so they parse with the
        // current schema. If one ever turns up in the older `AprilTagFieldLayout`
        // shape it has been hand-edited or replaced by something else.
        for (name, text) in BUNDLED {
            let raw: serde_json::Value =
                serde_json::from_str(text).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(raw.get("field-dimensions").is_some(), "{name} dimensions");
            assert!(raw.get("field-tags").is_some(), "{name} tags");
            assert!(raw.get("season").is_some(), "{name} season");
        }
    }

    #[test]
    fn corners_are_ordered_as_the_detector_reports_them() {
        use super::super::types::{CameraPose, Intrinsics, Mat3, Vec3};

        // A tag facing the field's `+x`, seen head on by a camera on the `-x`
        // side. The detector reports corners top-right, bottom-right,
        // bottom-left, top-left, so that is the order this must come out in.
        let json = r#"{
            "field": { "length": 16.0, "width": 8.0 },
            "tags": [{
                "ID": 7,
                "pose": {
                    "translation": { "x": 5.0, "y": 4.0, "z": 1.0 },
                    "rotation": { "quaternion": { "W": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0 } }
                }
            }]
        }"#;
        let layout = TagLayout::from_json("test", json).expect("layout");
        let corners = layout.corners(7, 0.165).expect("corners");

        // Camera axes in field coordinates: looking along `+x`, up is `+z`.
        let pose = CameraPose {
            position: Vec3::new(2.0, 4.0, 1.0),
            rotation: Mat3::from_rows([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
                .to_quaternion(),
        };
        let camera = Intrinsics::from_horizontal_fov(1280, 720, 60.0);
        let projected: Vec<(f64, f64)> = corners
            .iter()
            .map(|corner| {
                let point = Mat3::from_quaternion(pose.rotation).mul_vec(*corner - pose.position);
                camera.project(point).expect("in front of the camera")
            })
            .collect();

        // Right is increasing `u`, down is increasing `v`. The two right-hand
        // corners share a column, likewise the two left-hand ones.
        let (top_right, bottom_right, bottom_left, top_left) =
            (projected[0], projected[1], projected[2], projected[3]);
        assert!(
            (top_right.0 - bottom_right.0).abs() < 0.5,
            "right column {top_right:?} {bottom_right:?}"
        );
        assert!(
            (top_left.0 - bottom_left.0).abs() < 0.5,
            "left column {top_left:?} {bottom_left:?}"
        );
        assert!(
            top_right.0 > top_left.0,
            "right column should be right of left"
        );
        assert!(
            top_right.1 < bottom_right.1,
            "top-right should be above bottom-right"
        );
        assert!(
            top_left.1 < bottom_left.1,
            "top-left should be above bottom-left"
        );
        // Seen head on, the tag is centred and its width is the tag size at
        // the 3 m standoff: focal * size / distance.
        let centre_u = projected.iter().map(|corner| corner.0).sum::<f64>() / 4.0;
        let centre_v = projected.iter().map(|corner| corner.1).sum::<f64>() / 4.0;
        assert!((centre_u - camera.cx).abs() < 2.0, "centre u {centre_u}");
        assert!((centre_v - camera.cy).abs() < 2.0, "centre v {centre_v}");
        let width = projected[0].0 - projected[3].0;
        assert!(
            (width - camera.fx * 0.165 / 3.0).abs() < 1.0,
            "width {width}"
        );
    }

    #[test]
    fn the_older_april_tag_field_layout_schema_still_parses() {
        let legacy = r#"{
            "field": { "length": 16.0, "width": 8.0 },
            "tags": [
                {
                    "ID": 5,
                    "pose": {
                        "translation": { "x": 1.0, "y": 2.0, "z": 0.5 },
                        "rotation": { "quaternion": { "W": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0 } }
                    }
                }
            ]
        }"#;
        let layout = TagLayout::from_json("legacy", legacy).expect("legacy layout");
        assert_eq!(layout.field_length, 16.0);
        assert_eq!(layout.field_width, 8.0);
        assert_eq!(
            layout.tag(5).expect("tag 5").translation,
            Vec3::new(1.0, 2.0, 0.5)
        );
    }

    #[test]
    fn unknown_names_are_rejected_with_the_options() {
        let error = TagLayout::load("no-such-field").expect_err("must fail");
        let message = error.to_string();
        assert!(message.contains("2025-reefscape-welded"), "got: {message}");
    }
}

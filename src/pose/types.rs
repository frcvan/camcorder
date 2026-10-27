//! Small geometry types for the pose subsystem, in plain `f64`.
//!
//! Coordinates follow the WPILib field convention, which is also what the
//! bundled tag maps use: metres, origin at a corner of the field, `+x` along
//! the field's length, `+y` across its width, `+z` up.

use std::ops::{Add, Mul, Sub};

/// A point or direction in 3D.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    pub fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn scaled(self, factor: f64) -> Vec3 {
        Vec3::new(self.x * factor, self.y * factor, self.z * factor)
    }

    pub fn normalized(self) -> Vec3 {
        let norm = self.norm();
        if norm < 1e-12 {
            self
        } else {
            self.scaled(1.0 / norm)
        }
    }

    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    pub fn from_array(values: [f64; 3]) -> Self {
        Vec3::new(values[0], values[1], values[2])
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, factor: f64) -> Vec3 {
        self.scaled(factor)
    }
}

/// A 3x3 matrix, row major.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Mat3 {
    pub rows: [[f64; 3]; 3],
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 {
        rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    pub fn from_rows(rows: [[f64; 3]; 3]) -> Self {
        Mat3 { rows }
    }

    pub fn from_columns(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Mat3 {
            rows: [[a.x, b.x, c.x], [a.y, b.y, c.y], [a.z, b.z, c.z]],
        }
    }

    pub fn column(&self, index: usize) -> Vec3 {
        Vec3::new(
            self.rows[0][index],
            self.rows[1][index],
            self.rows[2][index],
        )
    }

    pub fn transpose(&self) -> Mat3 {
        let mut rows = [[0.0; 3]; 3];
        for (r, row) in rows.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = self.rows[c][r];
            }
        }
        Mat3 { rows }
    }

    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.rows[0][0] * v.x + self.rows[0][1] * v.y + self.rows[0][2] * v.z,
            self.rows[1][0] * v.x + self.rows[1][1] * v.y + self.rows[1][2] * v.z,
            self.rows[2][0] * v.x + self.rows[2][1] * v.y + self.rows[2][2] * v.z,
        )
    }

    pub fn mul_mat(&self, other: &Mat3) -> Mat3 {
        let mut rows = [[0.0; 3]; 3];
        for (r, row) in rows.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = (0..3).map(|k| self.rows[r][k] * other.rows[k][c]).sum();
            }
        }
        Mat3 { rows }
    }

    /// Nearest rotation matrix, by Gram-Schmidt on the first two columns.
    pub fn orthonormalized(&self) -> Mat3 {
        let first = self.column(0).normalized();
        let second = self.column(1) - first.scaled(first.dot(self.column(1)));
        let second = second.normalized();
        let third = first.cross(second);
        Mat3::from_columns(first, second, third)
    }

    /// Rotation by `angle` radians about the unit `axis` (Rodrigues).
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Mat3 {
        let axis = axis.normalized();
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        let Vec3 { x, y, z } = axis;
        Mat3 {
            rows: [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
        }
    }

    /// Rotation built from a unit quaternion.
    pub fn from_quaternion(q: Quat) -> Mat3 {
        let Quat { w, x, y, z } = q.normalized();
        Mat3 {
            rows: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
        }
    }

    pub fn to_quaternion(&self) -> Quat {
        let m = self.rows;
        let trace = m[0][0] + m[1][1] + m[2][2];
        let (w, x, y, z) = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            (
                0.25 * s,
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
            )
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            (
                (m[2][1] - m[1][2]) / s,
                0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
            )
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            (
                (m[0][2] - m[2][0]) / s,
                (m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s,
            )
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            (
                (m[1][0] - m[0][1]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                0.25 * s,
            )
        };
        Quat::new(w, x, y, z).normalized()
    }
}

/// A unit quaternion, `w + xi + yj + zk`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quat { w, x, y, z }
    }

    pub fn normalized(self) -> Quat {
        let norm = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if norm < 1e-12 {
            Quat::IDENTITY
        } else {
            Quat::new(self.w / norm, self.x / norm, self.y / norm, self.z / norm)
        }
    }
}

/// Pinhole camera intrinsics, in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    /// Intrinsics from a horizontal field of view, assuming square pixels and
    /// the principal point at the centre of the frame.
    pub fn from_horizontal_fov(width: u32, height: u32, fov_deg: f64) -> Self {
        let focal = (width as f64 * 0.5) / (fov_deg.to_radians() * 0.5).tan();
        Intrinsics {
            fx: focal,
            fy: focal,
            cx: (width as f64 - 1.0) * 0.5,
            cy: (height as f64 - 1.0) * 0.5,
            width,
            height,
        }
    }

    /// Project a point in camera coordinates (x right, y down, z forward).
    pub fn project(&self, point: Vec3) -> Option<(f64, f64)> {
        if point.z <= 1e-9 {
            return None;
        }
        Some((
            self.fx * point.x / point.z + self.cx,
            self.fy * point.y / point.z + self.cy,
        ))
    }
}

/// Where a camera sits and which way it looks, in field coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    /// Camera centre, metres, in the field frame.
    pub position: Vec3,
    /// Field axes to camera axes (camera: x right, y down, z along the view).
    pub rotation: Quat,
}

impl CameraPose {
    pub fn to_camera(&self, field_point: Vec3) -> Vec3 {
        Mat3::from_quaternion(self.rotation).mul_vec(field_point - self.position)
    }

    /// Unit vector the camera looks along, in field coordinates.
    pub fn view_direction(&self) -> Vec3 {
        let rotation = Mat3::from_quaternion(self.rotation).transpose();
        rotation.mul_vec(Vec3::new(0.0, 0.0, 1.0)).normalized()
    }

    /// Heading of the optical axis in the field plane (degrees, 0 = `+x`).
    pub fn yaw_deg(&self) -> f64 {
        let view = self.view_direction();
        view.y.atan2(view.x).to_degrees()
    }

    /// Elevation of the optical axis above the field plane (degrees).
    pub fn pitch_deg(&self) -> f64 {
        let view = self.view_direction();
        view.z.clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// Tilt of the camera's own up axis about the optical axis (degrees,
    /// positive when the camera is rolled clockwise).
    pub fn roll_deg(&self) -> f64 {
        let rotation = Mat3::from_quaternion(self.rotation).transpose();
        let up = rotation.mul_vec(Vec3::new(0.0, -1.0, 0.0));
        let forward = self.view_direction();
        let right = forward.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
        if right.norm() < 1e-9 {
            return 0.0;
        }
        let level_up = right.cross(forward).normalized();
        let across = up.dot(right);
        let along = up.dot(level_up);
        across.atan2(along).to_degrees()
    }
}

/// One tag seen in one frame, corners in the detector's order:
/// top-right, bottom-right, bottom-left, top-left (image coordinates, y down).
#[derive(Debug, Clone, PartialEq)]
pub struct TagObservation {
    pub id: u16,
    pub corners: [[f64; 2]; 4],
    /// Detection confidence reported by the decoder.
    pub margin: f32,
    /// Bit errors corrected while decoding; zero is a clean read.
    pub hamming: u8,
}

impl TagObservation {
    /// Rough size of the tag in pixels, useful for ranking detections.
    pub fn pixel_width(&self) -> f64 {
        let dx = self.corners[0][0] - self.corners[3][0];
        let dy = self.corners[0][1] - self.corners[3][1];
        (dx * dx + dy * dy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quaternion_and_matrix_round_trip() {
        let rotation = Mat3::from_axis_angle(Vec3::new(0.3, -1.0, 0.5), 0.7);
        let back = Mat3::from_quaternion(rotation.to_quaternion());
        for (row, (a, b)) in rotation.rows.iter().zip(back.rows.iter()).enumerate() {
            for (column, (a, b)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "row {row} column {column}: {a} vs {b}"
                );
            }
        }
    }

    /// Camera axes in field coordinates, written out: looking along `+x`, level.
    fn looking_along_x() -> Mat3 {
        // right = forward x up, down = forward x right, so the rows of the
        // field-to-camera rotation are those axes.
        Mat3::from_rows([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    }

    #[test]
    fn a_camera_looking_along_plus_x_reads_zero_yaw() {
        let pose = CameraPose {
            position: Vec3::new(0.0, 0.0, 5.0),
            rotation: looking_along_x().to_quaternion(),
        };
        assert!(pose.yaw_deg().abs() < 1e-6, "yaw {}", pose.yaw_deg());
        assert!(pose.pitch_deg().abs() < 1e-6, "pitch {}", pose.pitch_deg());
        assert!(pose.roll_deg().abs() < 1e-6, "roll {}", pose.roll_deg());
    }

    #[test]
    fn tilting_the_camera_up_reads_a_positive_pitch() {
        // Built from the axes rather than a rotation about some axis: looking
        // 0.3 rad above the horizon means forward = (cos, 0, sin).
        let (sin, cos) = 0.3f64.sin_cos();
        let forward = Vec3::new(cos, 0.0, sin);
        let right = forward.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
        let down = forward.cross(right);
        let pose = CameraPose {
            position: Vec3::new(0.0, 0.0, 5.0),
            rotation: Mat3::from_rows([right.to_array(), down.to_array(), forward.to_array()])
                .to_quaternion(),
        };
        assert!(
            (pose.pitch_deg() - 17.188_733).abs() < 1e-3,
            "pitch {}",
            pose.pitch_deg()
        );
        assert!(pose.roll_deg().abs() < 1e-6, "roll {}", pose.roll_deg());
    }

    #[test]
    fn projection_matches_the_pinhole_model() {
        let k = Intrinsics::from_horizontal_fov(1920, 1080, 60.0);
        let centre = k.project(Vec3::new(0.0, 0.0, 1.0)).expect("in front");
        assert!((centre.0 - k.cx).abs() < 1e-9);
        assert!((centre.1 - k.cy).abs() < 1e-9);
        assert!(k.project(Vec3::new(0.0, 0.0, -1.0)).is_none());
    }
}

"""Regression tests for the Rerun video's frames and timeline."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import field_model, rerun_poses as poses


class FieldModelTests(unittest.TestCase):
    def test_model_tape_and_tag_coordinates_match_wpilib(self):
        import json
        import numpy as np

        model = field_model.gltf_json(field_model.ASSET / "model.glb")
        layout = json.loads((field_model.ROOT / "assets/tags/2026-rebuilt-welded.json").read_text())
        length, width = layout["field-dimensions"]["length"], layout["field-dimensions"]["width"]
        config = json.loads((field_model.ASSET / "config.json").read_text())
        tag = next(tag for tag in config["aprilTags"] if tag["id"] == 16)
        gltf_tag = next(node for node in model["nodes"] if node.get("name", "").startswith("FE-00083-ID16: AprilTag"))
        base = next(model["nodes"][index] for index in gltf_tag["children"] if model["nodes"][index].get("name", "").startswith("FE-00082:"))
        base_point = field_model.transform(base) @ np.array([0.0, 0.0, 0.0059944, 1.0])
        x, y, z = base_point[:3]
        field = np.array([length / 2 - x, width / 2 + z, y])
        official = next(t["pose"]["translation"] for t in layout["field-tags"] if t["ID"] == 16)
        self.assertLess(np.linalg.norm(field - np.array([official["x"], official["y"], official["z"]])), 0.002)
        self.assertLess(abs(field[0] - (length / 2 - tag["position"][0])), 0.001)

        markings = field_model.field_markings()
        centre = next(line for line in markings if "White, 2in Wide, 316.19in" in line["name"])
        for point in (centre["start"], centre["end"]):
            self.assertAlmostEqual(point[0], length / 2, places=5)
            self.assertLess(abs(point[2]), 0.001)
        self.assertGreater(len(markings), 5)

        # Only the full-width alliance tape is usable without a pose already:
        # its colour says which line it is. The check above pins the transform
        # against a line the geometry itself fixes, and this pins what the
        # matcher is allowed to look for. A short strip of the same colour is a
        # different line, and matching it would be matching the wrong thing.
        tape = field_model.alliance_tape(markings)
        self.assertEqual(sorted(line["name"].split(",")[1].strip() for line in tape),
                         ["Blue", "Red"])
        for line in tape:
            span = abs(line["end"][1] - line["start"][1])
            self.assertGreater(span, width - 0.1, f"{line['name']} is not a full-width line")
        self.assertEqual(len(tape), 2)


class VideoTimelineTests(unittest.TestCase):
    def test_variable_rate_frames_keep_their_timestamps_and_last_image(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "variable.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=5:duration=1",
                    "-vf", "setpts=if(gte(N\\,2)\\,PTS+0.4/TB\\,PTS)",
                    "-fps_mode", "vfr", "-pix_fmt", "yuv420p", str(video),
                ],
                check=True,
            )
            times = poses.video_times(video)
            self.assertEqual(len(times), 5)
            self.assertAlmostEqual(times[0], 0.0)
            self.assertAlmostEqual(times[2], 0.8)
            self.assertAlmostEqual(times[-1], 1.2)

            # Ten minutes of 1080p60 as one JPEG per frame is tens of gigabytes,
            # so the file goes in once and each frame is a reference into it.
            # The fixture is also the check that references carry the video's
            # own presentation times, VFR gaps included: a picture at the wrong
            # time is a picture of a different moment than its pose.
            assets, references = [], []

            def record_asset(**kwargs):
                assets.append(kwargs)
                return kwargs

            def record_reference(**kwargs):
                references.append(kwargs)
                return kwargs

            camera = poses.Camera("test", {}, video, None)
            clusters = [[{"time": 0.0}, {"time": 1.2}], [{"time": 0.9}]]
            with patch.object(poses.rr, "set_time") as set_time, \
                 patch.object(poses.rr, "log") as log, \
                 patch.object(poses.rr, "AssetVideo", record_asset), \
                 patch.object(poses.rr, "VideoFrameReference", record_reference):
                poses.log_video(camera, clusters, ["first", "second"], None)

            asset, *frames = log.call_args_list
            self.assertEqual(asset.args[0], "world/videos/test")
            self.assertEqual(asset.args[1]["path"], str(video))
            self.assertTrue(asset.kwargs.get("static"), "the blob is logged once")
            self.assertEqual(len(assets), 1, "the file must not be re-encoded per frame")
            self.assertEqual(len(references), 5, "every video frame is a reference")
            self.assertEqual(
                [reference["video_reference"] for reference in references],
                ["world/videos/test"] * 5,
            )
            self.assertEqual([reference["seconds"] for reference in references], times)
            self.assertEqual([call.args[0] for call in frames], [
                "first/image", "first/image", "first/image", "second/image", "first/image"
            ])
            duration_calls = [call.kwargs["duration"] for call in set_time.call_args_list if call.args[0] == "video_time"]
            self.assertEqual(duration_calls, times)
            frame_calls = [call.kwargs["sequence"] for call in set_time.call_args_list if call.args[0] == "frame"]
            self.assertEqual(frame_calls, [int(time * 1000) for time in times])

            # `--max-frames` cuts references, never the asset: the file is the
            # recording's size, and it is in there once either way.
            references.clear()
            with patch.object(poses.rr, "set_time"), \
                 patch.object(poses.rr, "log"), \
                 patch.object(poses.rr, "AssetVideo", record_asset), \
                 patch.object(poses.rr, "VideoFrameReference", record_reference):
                poses.log_video(camera, clusters, ["first", "second"], 2)
            self.assertEqual(len(references), 2)

    def test_the_default_height_floor_keeps_a_camera_at_ankle_height(self):
        """The Orlando panes stand at 0.28m and 0.43m.

        The floor used to default to half a metre, on the theory that half a
        metre is where the mirror solution starts. It is not: on real footage it
        dropped every frame that had solved to the right branch and kept only the
        ones that had solved to the wrong one, so the recording showed a camera
        in the middle of the field. Only a pose below the carpet is impossible.
        """
        floor = poses.parse_args(["pose.json"]).min_height
        low = {"position": [16.68, -1.10, 0.28]}
        buried = {"position": [8.0, 4.0, -0.4]}
        self.assertTrue(
            poses.usable(low, floor),
            f"a camera at 0.28m must survive the default {floor:g}m floor",
        )
        self.assertFalse(poses.usable(buried, floor))
        self.assertTrue(poses.usable(low, None))
        # An explicit floor is still honoured, for anyone who wants one.
        self.assertFalse(poses.usable(low, 0.5))

    def test_a_rival_pose_is_drawn_beside_the_camera_that_was_reported(self):
        """A lone tag leaves a second camera, and the recording has to show it.

        The fit cannot choose between the two — a square is the same square
        turned half round — so the report carries the loser, and a viewer that
        dropped it would show only the answer that happened to win. The two are
        metres apart on the field, which is exactly why seeing both is worth
        more than reading about it: the wrong one sits at the wrong end of the
        trench, and the model behind it shows it.
        """
        camera = poses.Camera("clip", {
            "frame_width": 160, "frame_height": 120, "horizontal_fov_deg": 60.0,
            "solved": [
                {
                    "time": 1.0,
                    "check": {
                        "rival": {
                            "position": [8.88, -0.34, 2.09],
                            "rejected": "the mount is not there",
                        }
                    },
                },
                {
                    "time": 2.0,
                    "check": {
                        "rival": {
                            "position": [8.90, -0.35, 2.08],
                            "rejected": "the mount is not there",
                        }
                    },
                },
                {"time": 3.0, "check": {"pattern_agreement": 0.9, "rival": None}},
            ],
        }, Path("clip.mp4"), None)
        with patch.object(poses.rr, "log") as log, patch.object(
            poses.rr, "Points3D"
        ) as markers:
            places = poses.log_rivals(camera)

        # Two frames at the same half-metre place are one marker, not two.
        self.assertEqual(len(places), 1)
        self.assertIn("2 frame(s)", places[0])
        self.assertEqual(log.call_args.args[0], "world/rivals/clip")
        drawn = markers.call_args
        self.assertEqual(drawn.args[0], [[9.0, -0.5, 2.0]])
        # The marker is bare on purpose: a sentence pinned to every rival buries
        # the field it is drawn on. The reason lives in the report, not here.
        self.assertNotIn(
            "labels",
            drawn.kwargs,
            "rival markers carry no text — a label per place covers the field",
        )

        # A file with no rival anywhere logs nothing at all: an empty marker set
        # in the viewer reads as "the pose is certain", which is a lie.
        alone = poses.Camera("sure", {
            "solved": [{"time": 1.0, "check": {"rival": None}}],
        }, None, None)
        with patch.object(poses.rr, "log") as log:
            self.assertEqual(poses.log_rivals(alone), [])
        log.assert_not_called()

    def test_unmeasured_start_has_no_camera_pinhole_or_fabricated_pose(self):
        camera = poses.Camera("clip", {
            "frame_width": 160, "frame_height": 120, "horizontal_fov_deg": 60.0,
        }, Path("clip.mp4"), None)
        clusters = [[{"time": 5.0}], [{"time": 8.0}]]
        with patch.object(poses.rr, "set_time") as set_time, patch.object(poses.rr, "log") as log:
            poses.log_camera_pinhole("camera/one", camera, 5.0)
            self.assertEqual(set_time.call_args_list[0].kwargs["duration"], 5.0)
            self.assertNotIn("static", log.call_args.kwargs)

        with patch.object(poses, "video_times", return_value=[0.0, 5.0, 7.0, 8.0]), \
             patch.object(poses.rr, "AssetVideo"), \
             patch.object(poses.rr, "VideoFrameReference"), \
             patch.object(poses.rr, "set_time"), patch.object(poses.rr, "log") as log:
            poses.log_video(camera, clusters, ["camera/one", "camera/two"], None)
        self.assertEqual([call.args[0] for call in log.call_args_list], [
            "world/videos/clip", "unposed/clip/image", "camera/one/image",
            "camera/one/image", "camera/two/image",
        ])


def _matrix_from_xyzw(xyzw) -> "np.ndarray":
    """The rotation matrix a Rerun quaternion describes, `xyzw` order."""
    import numpy as np

    x, y, z, w = np.asarray(xyzw, dtype=float) / np.linalg.norm(xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _report_rotation(wxyz) -> "np.ndarray":
    """The report's field-to-camera rotation as a matrix, `wxyz` order."""
    w, x, y, z = wxyz
    return _matrix_from_xyzw([x, y, z, w])


class CameraProjectionTests(unittest.TestCase):
    """What the recording draws has to be the model the solve put there.

    A frustum that is a little too wide, or turned half a step off, still looks
    like a camera: the field and its tags are drawn and the picture is inside a
    cone. It is only detectable by asking where a known field point lands. These
    tests ask that of the values actually logged, so a rounded field of view, a
    un-conjugated quaternion or a station's single lens used for a zoomed frame
    all fail here rather than as "the blue camera looks a degree off".
    """

    #: Camera at (2, 3, 1.5) looking along field `+x`, level and unrolled:
    #: camera x is field -y, camera y is field -z, camera z is field +x.
    #: `wxyz`, field-to-camera, as the report stores it.
    ROTATION = [0.5, 0.5, -0.5, 0.5]
    POSITION = [2.0, 3.0, 1.5]

    def report(self, **overrides) -> dict:
        report = {
            "video": "clip.mp4",
            "tag_layout": None,
            "frame_width": 640,
            "frame_height": 480,
            # The `--fov` guess is 30°, which is 1194px here; the solve used
            # 900px. A recording drawn from the angle is a fifth too wide.
            "horizontal_fov_deg": 30.0,
            "focal_px": 900.0,
            "solved": [],
        }
        report.update(overrides)
        return report

    def frame(self, time: float = 4.0) -> dict:
        return {
            "time": time,
            "position": list(self.POSITION),
            "rotation": list(self.ROTATION),
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "rms_px": 0.2,
            "tags": [13],
        }

    def logged(self, report: dict, *frames: dict) -> list:
        """The `Pinhole` and `Transform3D` the script builds for these frames."""
        camera = poses.Camera("clip", report, None, None)
        made = []

        def record(**kwargs):
            made.append(kwargs)
            return kwargs

        with patch.object(poses.rr, "set_time"), patch.object(poses.rr, "log"), \
             patch.object(poses.rr, "Pinhole", record), \
             patch.object(poses.rr, "Transform3D", record):
            poses.log_frames(camera, list(frames), "world/cameras/clip")
        return made

    def test_the_drawn_frustum_puts_the_model_where_the_solve_put_it(self):
        import numpy as np

        made = self.logged(self.report(), self.frame())
        pinholes = [call for call in made if "focal_length" in call]
        transforms = [call for call in made if "rotation" in call]
        self.assertEqual(len(pinholes), 1, "one station frame, one lens")
        self.assertEqual(len(transforms), 1)

        # The lens is the one solved with, not the guessed angle rounded back
        # into pixels: 30° here would draw a frustum a fifth too wide, which at
        # the frame's edge is a tag tens of pixels from where the picture has it.
        self.assertAlmostEqual(pinholes[0]["focal_length"], 900.0)
        self.assertAlmostEqual(pinholes[0]["principal_point"][0], (640 - 1) / 2)
        self.assertAlmostEqual(pinholes[0]["principal_point"][1], (480 - 1) / 2)

        point = np.array([7.0, 2.0, 0.5])
        position = np.array(self.POSITION)
        to_camera = _report_rotation(self.ROTATION)
        expected = to_camera @ (point - position)
        self.assertGreater(expected[2], 0.0, "the fixture's point must be in front")

        # The transform goes the other way round: it places the camera's axes in
        # the field, so it is the transpose of the report's rotation. Reading the
        # report's quaternion as if it were that transform draws a camera looking
        # backwards — a pose that still projects, onto the wrong pixels.
        placed = _matrix_from_xyzw(transforms[0]["rotation"].xyzw)
        np.testing.assert_allclose(placed @ placed.T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(transforms[0]["translation"], position)
        np.testing.assert_allclose(placed.T @ (point - position), expected, atol=1e-9)

        focal = pinholes[0]["focal_length"]
        cx, cy = pinholes[0]["principal_point"]
        # Where the solve says the point is, from the report's own convention…
        solved = np.array([
            self.report()["focal_px"] * expected[0] / expected[2] + cx,
            self.report()["focal_px"] * expected[1] / expected[2] + cy,
        ])
        # …and where the recording puts it, from the logged transform and lens.
        through = placed.T @ (point - position)
        drawn = np.array([
            focal * through[0] / through[2] + cx,
            focal * through[1] / through[2] + cy,
        ])
        np.testing.assert_allclose(drawn, solved, atol=1e-9)

        # …and the fixture would notice the sign, rather than agreeing with the
        # bug the way a test that projects from behind does.
        wrong = _matrix_from_xyzw(self.ROTATION)
        wrong_camera = wrong.T @ (point - position)
        wrong_pixel = np.array([
            focal * wrong_camera[0] / wrong_camera[2] + cx,
            focal * wrong_camera[1] / wrong_camera[2] + cy,
        ])
        self.assertGreater(np.linalg.norm(wrong_pixel - solved), 20.0)

    def test_a_zoomed_pane_is_drawn_at_each_frame_s_own_lens(self):
        """A fixed camera that zooms was solved with a lens per frame.

        The station reports one focal length, but `ptz_station.frames` carries
        each frame's own: drawing the whole pane at one of them leaves the
        model drifting out of the other pictures by the zoom, which is easy to
        mistake for the camera being aimed a fraction wrong.
        """
        station = {
            "focal_px": 900.0,
            "frames": [
                {"time": 4.0, "rotation": self.ROTATION, "focal_px": 700.0},
                {"time": 9.0, "rotation": self.ROTATION, "focal_px": 1100.0},
            ],
        }
        report = self.report(ptz_station=station)
        made = self.logged(report, self.frame(4.0), self.frame(9.0))
        lenses = [call["focal_length"] for call in made if "focal_length" in call]
        self.assertEqual(lenses, [700.0, 1100.0])

        camera = poses.Camera("clip", report, None, None)
        self.assertAlmostEqual(camera.intrinsics(4.0)[0], 700.0)
        self.assertAlmostEqual(camera.intrinsics(9.0)[0], 1100.0)
        # A sample the station did not fit has no lens of its own here, and
        # borrowing its neighbour's would draw it at the wrong zoom.
        self.assertAlmostEqual(camera.intrinsics(30.0)[0], 900.0)

    def test_a_report_without_a_lens_falls_back_to_its_field_of_view(self):
        import numpy as np

        camera = poses.Camera("old", {
            "frame_width": 160, "frame_height": 120, "horizontal_fov_deg": 60.0,
        }, None, None)
        self.assertAlmostEqual(camera.intrinsics()[0], (160 * 0.5) / np.tan(np.radians(30.0)))

    def test_a_fitted_distortion_is_reported_because_the_viewer_cannot_draw_it(self):
        """Rerun's pinhole has no radial term.

        When the run fitted one, the rendered tags sit a few pixels from the
        picture near the frame's edge — which reads as a small aim error and is
        not one. Saying so is cheaper than letting the reader chase it.
        """
        camera = poses.Camera("clip", self.report(ptz_station={"k1": -0.06}), None, None)
        note = poses.lens_note(camera)
        self.assertIsNotNone(note)
        self.assertIn("-0.060", note)
        self.assertIsNone(poses.lens_note(poses.Camera("clip", self.report(), None, None)))


if __name__ == "__main__":
    unittest.main()

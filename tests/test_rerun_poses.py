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
            frames = list(poses.iter_video_jpegs(video))
            self.assertEqual(len(times), 5)
            self.assertEqual(len(frames), 5)
            self.assertAlmostEqual(times[0], 0.0)
            self.assertAlmostEqual(times[2], 0.8)
            self.assertAlmostEqual(times[-1], 1.2)
            self.assertTrue(all(image.endswith(b"\xff\xd9") for _, image in frames))
            self.assertEqual(len(list(poses.iter_video_jpegs(video, 2))), 2)

            camera = poses.Camera("test", {}, video, None)
            clusters = [[{"time": 0.0}, {"time": 1.2}], [{"time": 0.9}]]
            with patch.object(poses.rr, "set_time") as set_time, patch.object(poses.rr, "log") as log:
                poses.log_video(camera, clusters, ["first", "second"], None)
            self.assertEqual([call.args[0] for call in log.call_args_list], [
                "first/image", "first/image", "first/image", "second/image", "first/image"
            ])
            duration_calls = [call.kwargs["duration"] for call in set_time.call_args_list if call.args[0] == "video_time"]
            self.assertEqual(duration_calls, times)
            frame_calls = [call.kwargs["sequence"] for call in set_time.call_args_list if call.args[0] == "frame"]
            self.assertEqual(frame_calls, [int(time * 1000) for time in times])

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
             patch.object(poses, "iter_video_jpegs", return_value=iter(enumerate([b"jpeg"] * 4))), \
             patch.object(poses.rr, "set_time"), patch.object(poses.rr, "log") as log:
            poses.log_video(camera, clusters, ["camera/one", "camera/two"], None)
        self.assertEqual([call.args[0] for call in log.call_args_list], [
            "unposed/clip/image", "camera/one/image", "camera/one/image", "camera/two/image",
        ])


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the Rerun video's frames and timeline."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import rerun_poses as poses


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

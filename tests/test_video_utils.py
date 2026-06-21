import numpy as np
import pytest

from video_feedback.video_utils import load_frames


def _make_dummy_video(tmp_path):
    import cv2

    path = tmp_path / "dummy.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
    )
    for _ in range(30):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_load_frames_shape(tmp_path):
    video_path = _make_dummy_video(tmp_path)
    frames = load_frames(str(video_path), num_frames=8)
    assert frames.shape[0] == 8
    assert frames.shape[-1] == 3
    assert frames.dtype == np.uint8


def test_load_frames_invalid_path():
    with pytest.raises(ValueError):
        load_frames("does_not_exist.mp4")

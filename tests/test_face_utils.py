import numpy as np
import pytest

from video_feedback.face_utils import center_crop_square


def test_center_crop_square_shape_from_wide():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = center_crop_square(frame, size=64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_center_crop_square_shape_from_tall():
    frame = np.zeros((200, 80, 3), dtype=np.uint8)
    out = center_crop_square(frame, size=224)
    assert out.shape == (224, 224, 3)


def test_center_crop_square_takes_center():
    # 가운데에 흰 블록을 두면 중앙 크롭 결과에 흰색이 남아야 한다.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:60, 40:60] = 255
    out = center_crop_square(frame, size=50)
    assert out.max() == 255


@pytest.mark.gpu
def test_crop_largest_no_face_returns_none():
    from video_feedback.face_utils import FaceDetector

    det = FaceDetector(image_size=224)
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    assert det.crop_largest(black) is None

import numpy as np
import pytest

from video_feedback.embedding import l2_normalize


def test_l2_normalize_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(v)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_l2_normalize_zero_vector_safe():
    v = np.zeros(4, dtype=np.float32)
    out = l2_normalize(v)
    assert not np.any(np.isnan(out))


@pytest.mark.gpu
def test_embed_returns_unit_vector(tmp_path):
    import cv2

    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(30):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()

    from video_feedback.embedding import VideoEmbedder

    emb = VideoEmbedder().embed(str(path))
    assert emb.ndim == 1
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

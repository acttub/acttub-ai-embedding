import numpy as np
import pytest

from video_feedback.embedding import combine_embeddings, l2_normalize


def test_l2_normalize_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(v)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_l2_normalize_zero_vector_safe():
    v = np.zeros(4, dtype=np.float32)
    out = l2_normalize(v)
    assert not np.any(np.isnan(out))


def test_combine_embeddings_unit_length_and_dim():
    rs = np.random.RandomState(0)
    v = l2_normalize(rs.randn(1024).astype(np.float32))
    a = l2_normalize(rs.randn(512).astype(np.float32))
    out = combine_embeddings(v, a, w_audio=0.5)
    assert out.shape == (1536,)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_combine_embeddings_cosine_decomposes_by_weight():
    # 결합 벡터의 코사인 = (1-w)*영상코사인 + w*음성코사인 이어야 한다.
    rs = np.random.RandomState(1)
    v1 = l2_normalize(rs.randn(1024).astype(np.float32))
    v2 = l2_normalize(rs.randn(1024).astype(np.float32))
    a1 = l2_normalize(rs.randn(512).astype(np.float32))
    a2 = l2_normalize(rs.randn(512).astype(np.float32))
    w = 0.3
    c1 = combine_embeddings(v1, a1, w_audio=w)
    c2 = combine_embeddings(v2, a2, w_audio=w)
    expected = (1 - w) * float(v1 @ v2) + w * float(a1 @ a2)
    assert np.isclose(float(c1 @ c2), expected, atol=1e-5)


def test_combine_embeddings_zero_audio_weight_keeps_video_only():
    rs = np.random.RandomState(2)
    v = l2_normalize(rs.randn(1024).astype(np.float32))
    a = l2_normalize(rs.randn(512).astype(np.float32))
    out = combine_embeddings(v, a, w_audio=0.0)
    # 오디오 가중치 0 → 오디오 부분 전부 0, 영상 부분은 원본 영상 벡터
    assert np.allclose(out[1024:], 0.0, atol=1e-6)
    assert np.allclose(out[:1024], v, atol=1e-6)


@pytest.mark.gpu
def test_embed_segments_shape_and_normalized():
    import glob

    clips = sorted(glob.glob("연기영상/clips/*.mp4"))
    if not clips:
        pytest.skip("클립 데이터 없음")
    from video_feedback.embedding import VideoEmbedder

    segs = VideoEmbedder().embed_segments(clips[0], num_segments=4)
    assert segs.shape == (4, 1024)  # 4구간 × VJEPA 1024차원
    norms = np.linalg.norm(segs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)  # 각 구간 L2 정규화


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

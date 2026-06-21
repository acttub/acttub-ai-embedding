import glob

import numpy as np
import pytest


def _sample_clip():
    clips = sorted(glob.glob("연기영상/clips/*.mp4"))
    return clips[0] if clips else None


@pytest.mark.gpu
def test_audio_embed_returns_unit_vector():
    if _sample_clip() is None:
        pytest.skip("클립 데이터 없음")
    from video_feedback.audio_embedding import AudioEmbedder

    emb = AudioEmbedder().embed(_sample_clip())
    assert emb.ndim == 1
    assert emb.shape[0] > 0
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)


@pytest.mark.gpu
def test_audio_embed_is_deterministic():
    if _sample_clip() is None:
        pytest.skip("클립 데이터 없음")
    from video_feedback.audio_embedding import AudioEmbedder

    embedder = AudioEmbedder()
    e1 = embedder.embed(_sample_clip())
    e2 = embedder.embed(_sample_clip())
    assert np.allclose(e1, e2, atol=1e-4)

import numpy as np
import pytest


@pytest.mark.gpu
def test_text_embed_returns_unit_vector():
    from video_feedback.text_embedding import TextEmbedder

    emb = TextEmbedder().embed("나는 지금 너무 화가 나고 슬프다.")
    assert emb.ndim == 1
    assert emb.shape[0] == 768  # ko-sroberta 768d
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)


@pytest.mark.gpu
def test_text_embed_empty_is_zero_vector():
    from video_feedback.text_embedding import TextEmbedder

    emb = TextEmbedder().embed("   ")
    assert emb.shape[0] == 768
    assert np.allclose(emb, 0.0)


@pytest.mark.gpu
def test_text_embed_semantic_order():
    # 의미가 가까운 문장이 먼 문장보다 코사인이 커야 한다.
    from video_feedback.text_embedding import TextEmbedder

    emb = TextEmbedder()
    base = emb.embed("정말 미안해, 내가 다 잘못했어.")
    near = emb.embed("미안하다고, 전부 내 잘못이야.")
    far = emb.embed("오늘 날씨가 참 맑고 좋네요.")
    assert float(base @ near) > float(base @ far)

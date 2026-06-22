import numpy as np

from video_feedback.explain import (
    explain_match,
    format_explanation,
    match_segments,
    seg_to_timerange,
)


def _rows(*vecs):
    m = np.array(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_match_segments_finds_best_pair():
    # q[1]==r[2] (둘 다 [0,1])만 완전 일치 → 유일한 best 쌍 (1, 2)
    q = _rows([1, 0], [0, 1])
    r = _rows([0.6, 0.8], [0.8, 0.6], [0, 1])
    qi, ri, score = match_segments(q, r)
    assert (qi, ri) == (1, 2)
    assert score > 0.99


def test_seg_to_timerange_splits_evenly():
    assert seg_to_timerange(0, 4, 60.0) == (0.0, 15.0)
    assert seg_to_timerange(3, 4, 60.0) == (45.0, 60.0)


def test_format_explanation_has_both_timeranges():
    text = format_explanation(
        q_start=8.0, q_end=12.0, r_start=20.0, r_end=24.0, score=0.91
    )
    # 양쪽 구간을 "초 부근" 범위로, 점수와 함께 보여줘야 한다
    assert "8~12" in text
    assert "20~24" in text
    assert "0.91" in text
    assert "비슷" in text


def test_explain_match_combines_pieces(tmp_path):
    import cv2

    def _mk(name):
        p = tmp_path / name
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
        for _ in range(40):  # 40프레임 @ 10fps = 4초
            w.write(np.zeros((64, 64, 3), dtype=np.uint8))
        w.release()
        return str(p)

    qp, rp = _mk("q.mp4"), _mk("r.mp4")

    class FakeEmbedder:
        # 질의 모든 구간 [1,0]. 기준은 구간3만 [1,0] → best 쌍의 ri=3
        def embed_segments(self, path, num_segments=4):
            if path == qp:
                return _rows([1, 0], [1, 0], [1, 0], [1, 0])
            return _rows([0.6, 0.8], [0.6, 0.8], [0.6, 0.8], [1, 0])

    text = explain_match(FakeEmbedder(), qp, rp, num_segments=4)
    assert "인풋 영상" in text
    assert "C영상" in text
    # 기준 구간3 = 4초/4구간 = 3~4초
    assert "3~4" in text

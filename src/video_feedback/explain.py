"""구간 매칭 기반 "왜 비슷한지" 설명 생성.

질의 영상과 매칭된 기준 영상을 각각 시간 구간으로 나눠 임베딩한 뒤,
가장 닮은 구간 쌍을 찾아 "어느 장면이 닮았는지"를 짚어준다.
"""

import numpy as np

from video_feedback.video_utils import get_duration


def match_segments(q_segs: np.ndarray, r_segs: np.ndarray) -> tuple[int, int, float]:
    """질의·기준 구간 임베딩에서 가장 닮은 구간 쌍을 찾는다.

    Args:
        q_segs: 질의 구간 임베딩 (Nq, D), 각 행 L2 정규화 가정.
        r_segs: 기준 구간 임베딩 (Nr, D), 각 행 L2 정규화 가정.

    Returns:
        (질의 구간 인덱스, 기준 구간 인덱스, 코사인 유사도).
    """
    sim = q_segs @ r_segs.T  # (Nq, Nr) 코사인 매트릭스 (정규화 가정)
    qi, ri = np.unravel_index(int(sim.argmax()), sim.shape)
    return int(qi), int(ri), float(sim[qi, ri])


def seg_to_timerange(
    index: int, num_segments: int, duration: float
) -> tuple[float, float]:
    """구간 인덱스를 (시작초, 끝초) 시간 범위로 환산한다.

    Args:
        index: 구간 인덱스 (0-based).
        num_segments: 전체 구간 수.
        duration: 영상 길이(초).

    Returns:
        (시작초, 끝초).
    """
    seg = duration / num_segments
    return index * seg, (index + 1) * seg


def format_explanation(
    q_start: float,
    q_end: float,
    r_start: float,
    r_end: float,
    score: float,
) -> str:
    """가장 닮은 구간 쌍을 사람이 읽을 설명 문장으로 만든다.

    Args:
        q_start: 질의 구간 시작초.
        q_end: 질의 구간 끝초.
        r_start: 기준 구간 시작초.
        r_end: 기준 구간 끝초.
        score: 두 구간의 코사인 유사도.

    Returns:
        설명 문장.
    """
    return (
        f"인풋 영상의 {int(round(q_start))}~{int(round(q_end))}초 부근이 "
        f"C영상의 {int(round(r_start))}~{int(round(r_end))}초 부근과 "
        f"가장 비슷했어요 (유사도 {score:.2f})"
    )


def explain_match(
    embedder, query_path: str, ref_path: str, num_segments: int = 4
) -> str:
    """질의·기준 영상을 구간 임베딩해 "어느 장면이 닮았는지" 설명을 만든다.

    Args:
        embedder: ``embed_segments(path, num_segments)``를 제공하는 임베더
            (VideoEmbedder).
        query_path: 질의 영상 경로.
        ref_path: 매칭된 기준 영상 경로.
        num_segments: 영상을 나눌 구간 수.

    Returns:
        가장 닮은 구간 쌍을 설명하는 문장.
    """
    q_segs = embedder.embed_segments(query_path, num_segments=num_segments)
    r_segs = embedder.embed_segments(ref_path, num_segments=num_segments)
    qi, ri, score = match_segments(q_segs, r_segs)

    q_start, q_end = seg_to_timerange(qi, num_segments, get_duration(query_path))
    r_start, r_end = seg_to_timerange(ri, num_segments, get_duration(ref_path))
    return format_explanation(q_start, q_end, r_start, r_end, score)

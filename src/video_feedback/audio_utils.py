"""영상에서 오디오 파형을 추출하는 유틸리티 (PyAV)."""

import av
import numpy as np


def load_audio(path: str, target_sr: int = 48000) -> np.ndarray:
    """영상 파일에서 오디오를 모노 파형으로 추출한다.

    Args:
        path: 영상(또는 오디오) 파일 경로.
        target_sr: 목표 샘플레이트(Hz). 이 값으로 리샘플한다.

    Returns:
        shape (N,), dtype float32, 모노 파형 (값 범위 약 [-1, 1]).

    Raises:
        ValueError: 파일을 열 수 없거나 오디오 스트림/샘플이 없을 때.
    """
    try:
        container = av.open(path)
    except (av.FFmpegError, OSError) as exc:
        raise ValueError(f"오디오를 열 수 없습니다: {path}") from exc

    if not any(s.type == "audio" for s in container.streams):
        container.close()
        raise ValueError(f"오디오 스트림이 없습니다: {path}")

    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray().reshape(-1))
    # 리샘플러 내부 버퍼 플러시
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray().reshape(-1))
    container.close()

    if not chunks:
        raise ValueError(f"오디오 샘플을 읽지 못했습니다: {path}")
    return np.concatenate(chunks).astype(np.float32)

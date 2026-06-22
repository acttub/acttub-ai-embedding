"""영상 로딩 및 프레임 샘플링 유틸리티."""

import cv2
import numpy as np


def load_frames(path: str, num_frames: int = 16) -> np.ndarray:
    """영상에서 균등 간격으로 프레임을 샘플링한다.

    Args:
        path: 영상 파일 경로.
        num_frames: 추출할 프레임 수.

    Returns:
        shape (num_frames, H, W, 3), dtype uint8, RGB 순서의 배열.

    Raises:
        ValueError: 영상을 열 수 없거나 프레임이 없을 때.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"프레임이 없습니다: {path}")

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"프레임을 읽지 못했습니다: {path}")

    # 부족한 프레임은 마지막 프레임으로 패딩
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames[:num_frames]).astype(np.uint8)


def get_duration(path: str) -> float:
    """영상 길이를 초 단위로 반환한다.

    Args:
        path: 영상 파일 경로.

    Returns:
        영상 길이(초).

    Raises:
        ValueError: 영상을 열 수 없거나 FPS를 읽을 수 없을 때.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise ValueError(f"FPS를 읽을 수 없습니다: {path}")
    return total / fps

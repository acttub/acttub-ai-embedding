"""V-JEPA 2 기반 영상 임베딩."""

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from video_feedback.video_utils import load_frames


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """벡터를 L2 정규화한다 (영벡터 안전).

    Args:
        vec: 입력 벡터.
        eps: 영벡터 판정 임계값.

    Returns:
        L2 정규화된 float32 벡터. 노름이 eps 미만이면 원본을 그대로 반환한다.
    """
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class VideoEmbedder:
    """영상을 단일 벡터로 임베딩한다 (V-JEPA 2)."""

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        device: str | None = None,
    ) -> None:
        """임베더를 초기화한다.

        Args:
            model_name: HuggingFace 모델 ID (V-JEPA 2 계열).
            device: 추론 장치. None이면 가능 시 cuda, 아니면 cpu.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embed(self, video_path: str, num_frames: int = 64) -> np.ndarray:
        """영상 경로를 L2 정규화된 1D float32 임베딩 벡터로 변환한다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수 (V-JEPA 2 기본 64).

        Returns:
            shape (D,), dtype float32, L2 정규화된 임베딩 벡터.
        """
        frames = load_frames(video_path, num_frames=num_frames)  # (T, H, W, C) uint8
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T, C, H, W
        inputs = self.processor(video, return_tensors="pt").to(self.device)
        # 토큰별 비전 특징 → 평균 풀링으로 영상 단일 벡터
        features = self.model.get_vision_features(**inputs)
        pooled = features.mean(dim=1).squeeze(0)
        return l2_normalize(pooled.float().cpu().numpy())

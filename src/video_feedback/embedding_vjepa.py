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


def combine_embeddings(
    video_vec: np.ndarray, audio_vec: np.ndarray, w_audio: float = 0.5
) -> np.ndarray:
    """영상·음성 임베딩을 가중치로 결합해 단일 벡터로 만든다.

    각 모달을 L2 정규화돼 있다고 가정하고, 가중치의 제곱근을 곱해 concat한다.
    이렇게 하면 결과 벡터가 자동으로 단위 길이가 되고, 두 결합 벡터의
    코사인 유사도가 ``(1-w_audio)*영상코사인 + w_audio*음성코사인``으로 분해된다.

    Args:
        video_vec: L2 정규화된 영상 임베딩 (D_v,).
        audio_vec: L2 정규화된 음성 임베딩 (D_a,).
        w_audio: 음성 가중치 [0, 1]. 0이면 영상만, 1이면 음성만.

    Returns:
        shape (D_v + D_a,), dtype float32, 단위 길이 결합 벡터.
    """
    w_v = 1.0 - w_audio
    v = np.sqrt(w_v) * video_vec.astype(np.float32)
    a = np.sqrt(w_audio) * audio_vec.astype(np.float32)
    return np.concatenate([v, a]).astype(np.float32)


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

    @torch.no_grad()
    def embed_segments(
        self, video_path: str, num_frames: int = 64, num_segments: int = 4
    ) -> np.ndarray:
        """영상을 시간 구간으로 나눠 각 구간의 임베딩을 반환한다.

        V-JEPA 2는 영상을 시공간 토큰 격자(시간 32 × 공간 256)로 들고 있다.
        전체 평균(embed) 대신 시간축을 num_segments 그룹으로 묶어 평균내면,
        추가 추론 없이 추론 1회로 구간별 임베딩을 얻는다. "어느 장면이
        닮았는지"를 짚는 구간 매칭에 쓴다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수 (V-JEPA 2 기본 64).
            num_segments: 나눌 구간 수. 시간 토큰 수(32)의 약수여야 한다.

        Returns:
            shape (num_segments, D), dtype float32, 각 행이 L2 정규화된 구간 임베딩.
        """
        frames = load_frames(video_path, num_frames=num_frames)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)
        inputs = self.processor(video, return_tensors="pt").to(self.device)
        feats = self.model.get_vision_features(**inputs).squeeze(0)  # (T*S, D)

        n_tokens, dim = feats.shape
        n_time = 32  # 실측: 토큰 8192 = 32(시간) × 256(공간)
        n_spatial = n_tokens // n_time
        feats = feats.reshape(n_time, n_spatial, dim)

        per = n_time // num_segments
        feats = feats[: per * num_segments].reshape(num_segments, per, n_spatial, dim)
        segs = feats.mean(dim=(1, 2)).float().cpu().numpy()  # (num_segments, D)
        return np.stack([l2_normalize(s) for s in segs])

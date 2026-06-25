"""얼굴 표정(FER) 기반 영상 임베딩.

연기영상은 표정·감정이 핵심이라, 영상 전체의 모션/장면을 보는 V-JEPA 2 대신
프레임별 얼굴 크롭을 표정 인식 ViT(`trpakov/vit-face-expression`)에 넣어
CLS 토큰(768d)을 표정 임베딩으로 쓴다. 기존 V-JEPA 구현은 ``embedding_vjepa.py``에
보존돼 있다.
"""

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from video_feedback.face_utils import FaceDetector, center_crop_square
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


def combine_weighted(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """여러 모달 임베딩을 가중치로 결합해 단일 벡터로 만든다 (N-모달 일반화).

    각 모달을 L2 정규화돼 있다고 가정하고, 가중치의 제곱근을 곱해 concat한다.
    가중치 합이 1이면 결과 벡터가 단위 길이가 되고, 두 결합 벡터의 코사인
    유사도가 ``Σ wᵢ·모달ᵢ코사인``으로 분해된다.

    Args:
        vectors: 모달별 L2 정규화 임베딩 리스트.
        weights: 모달별 가중치 리스트 (vectors와 같은 길이). 음수는 0으로 클램프.

    Returns:
        dtype float32, 결합 벡터.
    """
    parts = [
        np.sqrt(max(w, 0.0)) * vec.astype(np.float32)
        for vec, w in zip(vectors, weights)
    ]
    return np.concatenate(parts).astype(np.float32)


def combine_embeddings(
    video_vec: np.ndarray, audio_vec: np.ndarray, w_audio: float = 0.5
) -> np.ndarray:
    """영상·음성 임베딩을 가중치로 결합해 단일 벡터로 만든다 (2-모달, 하위호환).

    ``combine_weighted([video, audio], [1-w_audio, w_audio])``의 얇은 래퍼다.

    Args:
        video_vec: L2 정규화된 영상 임베딩 (D_v,).
        audio_vec: L2 정규화된 음성 임베딩 (D_a,).
        w_audio: 음성 가중치 [0, 1]. 0이면 영상만, 1이면 음성만.

    Returns:
        shape (D_v + D_a,), dtype float32, 단위 길이 결합 벡터.
    """
    return combine_weighted([video_vec, audio_vec], [1.0 - w_audio, w_audio])


class VideoEmbedder:
    """영상을 얼굴 표정 임베딩 단일 벡터로 변환한다 (ViT-FER, 768d)."""

    def __init__(
        self,
        model_name: str = "trpakov/vit-face-expression",
        device: str | None = None,
    ) -> None:
        """임베더를 초기화한다 (표정 ViT + 얼굴 검출기).

        Args:
            model_name: HuggingFace 표정 인식 ViT 모델 ID.
            device: 추론 장치. None이면 가능 시 cuda, 아니면 cpu.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = (
            AutoModelForImageClassification.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.detector = FaceDetector(device=self.device, image_size=224)

    @torch.no_grad()
    def _frame_embeddings(self, video_path: str, num_frames: int) -> np.ndarray:
        """프레임별 얼굴 크롭의 표정 임베딩(CLS 768d) 배열을 반환한다.

        프레임마다 가장 큰 얼굴을 크롭해 ViT-FER에 넣고 마지막 hidden state의
        CLS 토큰을 뽑는다. 얼굴이 검출된 프레임만 쓰되, 클립 전체에 얼굴이
        하나도 없으면 모든 프레임을 중앙 크롭으로 폴백한다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수.

        Returns:
            shape (T, 768), dtype float32. T는 사용된 프레임 수(≤ num_frames).
        """
        frames = load_frames(video_path, num_frames=num_frames)  # (T,H,W,3) RGB
        crops = [
            face
            for fr in frames
            if (face := self.detector.crop_largest(fr)) is not None
        ]
        if not crops:  # 얼굴이 전혀 안 잡히면 중앙 크롭 폴백
            crops = [center_crop_square(fr, size=224) for fr in frames]
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0]  # (T, 768) CLS 토큰
        return cls.float().cpu().numpy()

    def embed(self, video_path: str, num_frames: int = 64) -> np.ndarray:
        """영상 경로를 L2 정규화된 1D float32 표정 임베딩으로 변환한다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수.

        Returns:
            shape (768,), dtype float32, L2 정규화된 임베딩 벡터.
        """
        feats = self._frame_embeddings(video_path, num_frames)
        return l2_normalize(feats.mean(axis=0))

    def embed_segments(
        self, video_path: str, num_frames: int = 64, num_segments: int = 4
    ) -> np.ndarray:
        """영상을 시간 구간으로 나눠 각 구간의 표정 임베딩을 반환한다.

        프레임별 표정 임베딩을 시간 순서대로 ``num_segments`` 그룹으로 묶어
        각 그룹을 평균낸다. "어느 장면의 표정이 닮았는지" 구간 매칭에 쓴다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수.
            num_segments: 나눌 구간 수.

        Returns:
            shape (num_segments, 768), dtype float32, 각 행이 L2 정규화된 구간 임베딩.
        """
        feats = self._frame_embeddings(video_path, num_frames)  # (T, 768)
        t = feats.shape[0]
        bounds = np.linspace(0, t, num_segments + 1).astype(int)
        segs = []
        for s in range(num_segments):
            lo, hi = bounds[s], max(bounds[s] + 1, bounds[s + 1])
            segs.append(l2_normalize(feats[lo:hi].mean(axis=0)))
        return np.stack(segs)

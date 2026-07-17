"""얼굴 표정(HSEmotion) 기반 영상 임베딩.

ML 스코어러(acting-score)와 **같은 표정 모델** ``enet_b0_8_va_mtl``(HSEmotion, ONNX)을
공유한다. 프레임마다 YuNet으로 가장 큰 얼굴을 잘라 HSEmotion에 넣고, 8감정 확률 +
valence/arousal(=10d)을 뽑는다. 한 영상은 이 프레임 시퀀스의 시간 통계(평균·표준편차,
20d)로 요약한다 — 평균은 "어떤 감정을", 표준편차는 "얼마나 폭넓게 표현했는지"(표현
역동성)를 담는다. 스코어러의 감정 head 출력과 같은 공간이라 "비슷하게 검색된 영상"이
"비슷하게 채점"되는 일관성을 갖는다.

기존 V-JEPA 구현은 ``embedding_vjepa.py``에 보존돼 있다.
"""

import numpy as np

from video_feedback.face_utils import FaceDetector, center_crop_square
from video_feedback.video_utils import load_frames

# 스코어러(acting-score)와 동일 체크포인트. 바꾸면 인덱스도 같은 모델로 재빌드해야 한다.
HSEMOTION_MODEL = "enet_b0_8_va_mtl"
# 프레임 특징 10d(8감정 + valence + arousal)를 시간 통계 2종(평균·표준편차)으로 요약.
FRAME_DIM = 10
EMBED_DIM = FRAME_DIM * 2  # 20


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
    """영상·음성 임베딩을 가중치로 결합해 단일 벡터로 만든다 (2-모달).

    ``combine_weighted([video, audio], [1-w_audio, w_audio])``의 얇은 래퍼다.

    Args:
        video_vec: L2 정규화된 영상 임베딩 (D_v,).
        audio_vec: L2 정규화된 음성 임베딩 (D_a,).
        w_audio: 음성 가중치 [0, 1]. 0이면 영상만, 1이면 음성만.

    Returns:
        shape (D_v + D_a,), dtype float32, 단위 길이 결합 벡터.
    """
    return combine_weighted([video_vec, audio_vec], [1.0 - w_audio, w_audio])


def _summarize(frame_feats: np.ndarray) -> np.ndarray:
    """프레임별 표정 특징(T, 10)을 시간 통계(20,)로 요약한다.

    평균(무슨 감정을 냈나)과 표준편차(얼마나 폭넓게 = 표현 역동성)를 concat한다.

    Args:
        frame_feats: shape (T, 10) 프레임별 [8감정확률, valence, arousal].

    Returns:
        shape (20,), dtype float32 (정규화 전).
    """
    mean = frame_feats.mean(axis=0)
    std = frame_feats.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


class VideoEmbedder:
    """영상을 얼굴 표정 임베딩 단일 벡터로 변환한다 (HSEmotion, 20d)."""

    def __init__(
        self,
        model_name: str = HSEMOTION_MODEL,
        device: str | None = None,
    ) -> None:
        """임베더를 초기화한다 (HSEmotion ONNX + YuNet 얼굴 검출기).

        Args:
            model_name: HSEmotion ONNX 모델 이름 (스코어러와 동일 체크포인트).
            device: 하위호환용 인자. HSEmotion ONNX는 CPU로 실행되어 무시된다.
        """
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        # onnxruntime CPUExecutionProvider 고정. device는 인터페이스 호환용으로만 노출.
        self.device = "cpu"
        self.fer = HSEmotionRecognizer(model_name=model_name)
        self.detector = FaceDetector(image_size=224)

    def _frame_features(self, video_path: str, num_frames: int) -> np.ndarray:
        """프레임별 얼굴 크롭의 표정 특징(8감정 + VA, 10d) 배열을 반환한다.

        프레임마다 가장 큰 얼굴을 크롭해 HSEmotion에 넣는다. 얼굴이 검출된
        프레임만 쓰되, 클립 전체에 얼굴이 하나도 없으면 모든 프레임을 중앙
        크롭으로 폴백한다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수.

        Returns:
            shape (T, 10), dtype float32. T는 사용된 프레임 수(≤ num_frames).
        """
        frames = load_frames(video_path, num_frames=num_frames)  # (T,H,W,3) RGB
        crops = [
            face
            for fr in frames
            if (face := self.detector.crop_largest(fr)) is not None
        ]
        if not crops:  # 얼굴이 전혀 안 잡히면 중앙 크롭 폴백
            crops = [center_crop_square(fr, size=224) for fr in frames]
        # HSEmotion은 RGB 크롭을 받는다. 배치 추론으로 (T, 10) 스코어를 얻는다.
        _, scores = self.fer.predict_multi_emotions(crops, logits=False)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 1:  # 프레임 1개면 (10,) → (1, 10)
            scores = scores[None, :]
        return scores

    def embed(self, video_path: str, num_frames: int = 64) -> np.ndarray:
        """영상 경로를 L2 정규화된 1D float32 표정 임베딩으로 변환한다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수.

        Returns:
            shape (20,), dtype float32, L2 정규화된 임베딩 벡터.
        """
        feats = self._frame_features(video_path, num_frames)
        return l2_normalize(_summarize(feats))

    def embed_segments(
        self, video_path: str, num_frames: int = 64, num_segments: int = 4
    ) -> np.ndarray:
        """영상을 시간 구간으로 나눠 각 구간의 표정 임베딩을 반환한다.

        프레임별 표정 특징을 시간 순서대로 ``num_segments`` 그룹으로 묶어 각
        그룹을 시간 통계로 요약한다. "어느 장면의 표정이 닮았는지" 구간 매칭에 쓴다.

        Args:
            video_path: 영상 파일 경로.
            num_frames: 샘플링할 프레임 수.
            num_segments: 나눌 구간 수.

        Returns:
            shape (num_segments, 20), dtype float32, 각 행이 L2 정규화된 구간 임베딩.
        """
        feats = self._frame_features(video_path, num_frames)  # (T, 10)
        t = feats.shape[0]
        bounds = np.linspace(0, t, num_segments + 1).astype(int)
        segs = []
        for s in range(num_segments):
            lo, hi = bounds[s], max(bounds[s] + 1, bounds[s + 1])
            segs.append(l2_normalize(_summarize(feats[lo:hi])))
        return np.stack(segs)

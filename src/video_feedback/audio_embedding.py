"""CLAP 기반 오디오 임베딩."""

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from video_feedback.audio_utils import load_audio
from video_feedback.embedding import l2_normalize


class AudioEmbedder:
    """영상의 오디오를 단일 벡터로 임베딩한다 (CLAP)."""

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str | None = None,
    ) -> None:
        """임베더를 초기화한다.

        Args:
            model_name: HuggingFace 모델 ID (CLAP 계열).
            device: 추론 장치. None이면 가능 시 cuda, 아니면 cpu.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        # CLAP은 48kHz 입력을 기대한다.
        self.sample_rate = 48000

    @torch.no_grad()
    def embed(self, video_path: str) -> np.ndarray:
        """영상의 오디오를 L2 정규화된 1D float32 임베딩 벡터로 변환한다.

        CLAP은 10초를 넘는 입력을 랜덤 구간으로 잘라(rand_trunc) 비결정적이고
        일부만 본다. 이를 피하려고 오디오를 10초 청크로 나눠 각 청크를
        임베딩한 뒤 평균낸다. 각 청크는 10초 이하라 truncation이 걸리지 않아
        결정적이며, 전체 길이를 빠짐없이 반영한다.

        Args:
            video_path: 영상(또는 오디오) 파일 경로.

        Returns:
            shape (D,), dtype float32, L2 정규화된 오디오 임베딩 벡터.
        """
        wav = load_audio(video_path, target_sr=self.sample_rate)
        chunk_len = self.sample_rate * 10  # CLAP 기본 10초 윈도
        min_len = self.sample_rate  # 1초 미만 꼬리 청크는 버림
        chunks = [wav[i : i + chunk_len] for i in range(0, len(wav), chunk_len)]
        chunks = [c for c in chunks if len(c) >= min_len] or [wav]

        feats = []
        for chunk in chunks:
            inputs = self.processor(
                audio=chunk, sampling_rate=self.sample_rate, return_tensors="pt"
            ).to(self.device)
            f = self.model.get_audio_features(**inputs).pooler_output.squeeze(0)
            feats.append(f)
        mean = torch.stack(feats).mean(dim=0)
        return l2_normalize(mean.float().cpu().numpy())

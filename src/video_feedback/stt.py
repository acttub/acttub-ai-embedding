"""영상 대사 STT (Whisper large-v3-turbo, HuggingFace transformers).

기존 torch(cu126) 스택을 그대로 재사용한다 — 추가 CUDA 툴체인 불필요.
"""

import numpy as np
import torch
from transformers import pipeline

from video_feedback.audio_utils import load_audio

WHISPER_SR = 16000  # Whisper는 16kHz 모노 입력을 기대한다.


class Transcriber:
    """영상의 음성을 텍스트(대본)로 받아쓴다 (Whisper)."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        device: str | None = None,
        language: str = "korean",
    ) -> None:
        """STT 파이프라인을 초기화한다.

        Args:
            model_name: HuggingFace Whisper 모델 ID.
            device: 추론 장치. None이면 가능 시 cuda, 아니면 cpu.
            language: 받아쓰기 언어 (기본 한국어).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=dtype,
            device=self.device,
            chunk_length_s=30,  # 30초 청크로 긴 오디오도 처리
        )

    def transcribe(self, video_path: str) -> str:
        """영상의 음성을 받아써 대본 텍스트를 반환한다.

        오디오 스트림이 없거나 무음이면 빈 문자열을 반환한다 (호출부에서
        대본 모달을 비워 두는 신호로 사용).

        Args:
            video_path: 영상(또는 오디오) 파일 경로.

        Returns:
            받아쓴 텍스트 (공백 정리됨). 대사 없으면 "".
        """
        try:
            wav = load_audio(video_path, target_sr=WHISPER_SR)
        except ValueError:
            return ""  # 오디오 스트림 없음
        if wav.size == 0:
            return ""
        out = self.pipe(
            wav.astype(np.float32),
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        return (out.get("text") or "").strip()

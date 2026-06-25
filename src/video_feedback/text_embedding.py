"""대본(STT 텍스트) 문장 임베딩 (Sentence-Transformers, 한국어)."""

import numpy as np

from video_feedback.embedding import l2_normalize


class TextEmbedder:
    """대본 텍스트를 단일 벡터로 임베딩한다 (ko-sroberta, 768d)."""

    def __init__(
        self,
        model_name: str = "jhgan/ko-sroberta-multitask",
        device: str | None = None,
    ) -> None:
        """임베더를 초기화한다.

        Args:
            model_name: Sentence-Transformers 모델 ID (한국어 SBERT).
            device: 추론 장치. None이면 가능 시 cuda, 아니면 cpu.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, text: str) -> np.ndarray:
        """대본 텍스트를 L2 정규화된 1D float32 임베딩으로 변환한다.

        빈 문자열(대사 없음/STT 실패)은 영벡터를 반환한다 — 결합 시 대본
        모달이 자연스럽게 0 기여가 된다.

        Args:
            text: 임베딩할 대본 텍스트.

        Returns:
            shape (D,), dtype float32, L2 정규화된 텍스트 임베딩.
        """
        dim = self.model.get_sentence_embedding_dimension()
        if not text or not text.strip():
            return np.zeros(dim, dtype=np.float32)
        vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return l2_normalize(vec.astype(np.float32))

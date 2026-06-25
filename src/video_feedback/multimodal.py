"""영상+음성+대본 멀티모달 기준 인덱스 (런타임 가중치 결합 검색)."""

import numpy as np

from video_feedback.embedding import combine_weighted
from video_feedback.reference_db import ReferenceDB


class MultiModalReferenceDB:
    """기준 영상의 영상·음성·대본 임베딩을 따로 보관하고, 검색 시점에 가중치로 결합한다.

    모달 벡터를 분리 저장하므로 검색할 때 ``w_audio``/``w_text``를 바꿔
    "영상/음성/대본 중 무엇을 더 볼지"를 런타임에 조절할 수 있다.

    대본(text) 모달은 옵셔널이다. 텍스트 벡터 없이 add하면 기존 2-모달
    (영상+음성)로 동작하고, 이전 버전 인덱스(.npz)도 그대로 로드된다.
    """

    def __init__(self) -> None:
        """빈 DB를 생성한다."""
        self._ids: list[str] = []
        self._video: list[np.ndarray] = []
        self._audio: list[np.ndarray] = []
        self._text: list[np.ndarray] = []

    def add(
        self,
        ref_id: str,
        video_vec: np.ndarray,
        audio_vec: np.ndarray,
        text_vec: np.ndarray | None = None,
    ) -> None:
        """기준 임베딩 한 묶음을 추가한다 (각 모달 L2 정규화 가정).

        Args:
            ref_id: 기준 영상 식별자.
            video_vec: L2 정규화된 영상 임베딩.
            audio_vec: L2 정규화된 음성 임베딩.
            text_vec: L2 정규화된 대본 임베딩. None이면 대본 모달 없음.
        """
        self._ids.append(ref_id)
        self._video.append(video_vec.astype(np.float32))
        self._audio.append(audio_vec.astype(np.float32))
        if text_vec is not None:
            self._text.append(text_vec.astype(np.float32))

    def _has_text(self) -> bool:
        """모든 기준에 대본 임베딩이 갖춰졌는지."""
        return bool(self._ids) and len(self._text) == len(self._ids)

    def search(
        self,
        q_video: np.ndarray,
        q_audio: np.ndarray,
        q_text: np.ndarray | None = None,
        w_audio: float = 0.5,
        w_text: float = 0.0,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """질의 영상·음성·대본을 가중 결합해 top-K 기준을 찾는다.

        영상 가중치는 ``1 - w_audio - w_text``로 자동 결정된다. 대본 모달이
        없으면(q_text=None 또는 인덱스에 대본 없음) w_text는 0으로 무시되고
        영상 가중치는 ``1 - w_audio``가 된다.

        Args:
            q_video: 질의 영상 임베딩 (정규화 가정).
            q_audio: 질의 음성 임베딩 (정규화 가정).
            q_text: 질의 대본 임베딩 (정규화 가정). None이면 대본 안 봄.
            w_audio: 음성 가중치 [0, 1].
            w_text: 대본 가중치 [0, 1].
            k: 반환할 최대 개수.

        Returns:
            (ref_id, 결합 코사인 유사도) 리스트, 유사도 내림차순.

        Raises:
            ValueError: DB가 비어 있을 때.
        """
        if not self._ids:
            raise ValueError("기준 DB가 비어 있습니다.")

        use_text = q_text is not None and self._has_text()
        if not use_text:
            w_text = 0.0
        w_video = 1.0 - w_audio - w_text

        db = ReferenceDB()
        for i, ref_id in enumerate(self._ids):
            vecs = [self._video[i], self._audio[i]]
            ws = [w_video, w_audio]
            if use_text:
                vecs.append(self._text[i])
                ws.append(w_text)
            db.add(ref_id, combine_weighted(vecs, ws))

        q_vecs = [q_video, q_audio]
        q_ws = [w_video, w_audio]
        if use_text:
            q_vecs.append(q_text)
            q_ws.append(w_text)
        query = combine_weighted(q_vecs, q_ws)
        return db.search(query, k=k)

    def save(self, path: str) -> None:
        """npz로 저장한다 (ids, video/audio_vectors, 있으면 text_vectors).

        Args:
            path: 저장 경로.
        """
        arrays = dict(
            ids=np.array(self._ids),
            video_vectors=np.stack(self._video),
            audio_vectors=np.stack(self._audio),
        )
        if self._has_text():
            arrays["text_vectors"] = np.stack(self._text)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "MultiModalReferenceDB":
        """npz에서 로드한다 (text_vectors 없으면 2-모달로 복원).

        Args:
            path: npz 파일 경로.

        Returns:
            복원된 MultiModalReferenceDB.
        """
        data = np.load(path, allow_pickle=True)
        has_text = "text_vectors" in data.files
        texts = data["text_vectors"] if has_text else None
        db = cls()
        for i, (ref_id, vid, aud) in enumerate(
            zip(data["ids"], data["video_vectors"], data["audio_vectors"])
        ):
            db.add(str(ref_id), vid, aud, texts[i] if has_text else None)
        return db

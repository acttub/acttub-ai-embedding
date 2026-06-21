"""영상+음성 멀티모달 기준 인덱스 (런타임 가중치 결합 검색)."""

import numpy as np

from video_feedback.embedding import combine_embeddings
from video_feedback.reference_db import ReferenceDB


class MultiModalReferenceDB:
    """기준 영상의 영상·음성 임베딩을 따로 보관하고, 검색 시점에 가중치로 결합한다.

    영상/음성 벡터를 분리 저장하므로 검색할 때 ``w_audio``를 바꿔
    "영상을 더 볼지 음성을 더 볼지"를 런타임에 조절할 수 있다.
    """

    def __init__(self) -> None:
        """빈 DB를 생성한다."""
        self._ids: list[str] = []
        self._video: list[np.ndarray] = []
        self._audio: list[np.ndarray] = []

    def add(self, ref_id: str, video_vec: np.ndarray, audio_vec: np.ndarray) -> None:
        """기준 임베딩 한 쌍을 추가한다 (둘 다 L2 정규화 가정).

        Args:
            ref_id: 기준 영상 식별자.
            video_vec: L2 정규화된 영상 임베딩.
            audio_vec: L2 정규화된 음성 임베딩.
        """
        self._ids.append(ref_id)
        self._video.append(video_vec.astype(np.float32))
        self._audio.append(audio_vec.astype(np.float32))

    def search(
        self,
        q_video: np.ndarray,
        q_audio: np.ndarray,
        w_audio: float = 0.5,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """질의 영상·음성을 가중 결합해 top-K 기준을 찾는다.

        Args:
            q_video: 질의 영상 임베딩 (정규화 가정).
            q_audio: 질의 음성 임베딩 (정규화 가정).
            w_audio: 음성 가중치 [0, 1]. 0이면 영상만, 1이면 음성만 본다.
            k: 반환할 최대 개수.

        Returns:
            (ref_id, 결합 코사인 유사도) 리스트, 유사도 내림차순.

        Raises:
            ValueError: DB가 비어 있을 때.
        """
        if not self._ids:
            raise ValueError("기준 DB가 비어 있습니다.")
        db = ReferenceDB()
        for ref_id, vid, aud in zip(self._ids, self._video, self._audio):
            db.add(ref_id, combine_embeddings(vid, aud, w_audio))
        query = combine_embeddings(q_video, q_audio, w_audio)
        return db.search(query, k=k)

    def save(self, path: str) -> None:
        """npz로 저장한다 (ids, video_vectors, audio_vectors).

        Args:
            path: 저장 경로.
        """
        np.savez(
            path,
            ids=np.array(self._ids),
            video_vectors=np.stack(self._video),
            audio_vectors=np.stack(self._audio),
        )

    @classmethod
    def load(cls, path: str) -> "MultiModalReferenceDB":
        """npz에서 로드한다.

        Args:
            path: npz 파일 경로.

        Returns:
            복원된 MultiModalReferenceDB.
        """
        data = np.load(path, allow_pickle=True)
        db = cls()
        for ref_id, vid, aud in zip(
            data["ids"], data["video_vectors"], data["audio_vectors"]
        ):
            db.add(str(ref_id), vid, aud)
        return db

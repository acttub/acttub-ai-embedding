"""기준 영상 임베딩 저장/매칭 (numpy 기반)."""

import numpy as np


class ReferenceDB:
    """기준 영상 임베딩을 보관하고 코사인 유사도로 매칭한다."""

    def __init__(self) -> None:
        """빈 DB를 생성한다."""
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    def add(self, ref_id: str, vector: np.ndarray) -> None:
        """기준 임베딩을 추가한다 (벡터는 정규화돼 있다고 가정).

        Args:
            ref_id: 기준 영상 식별자.
            vector: L2 정규화된 임베딩 벡터.
        """
        self._ids.append(ref_id)
        self._vectors.append(vector.astype(np.float32))

    def match(self, vector: np.ndarray) -> tuple[str, float]:
        """가장 유사한 기준 id와 코사인 유사도를 반환한다.

        Args:
            vector: 질의 임베딩 (정규화 가정).

        Returns:
            (가장 유사한 ref_id, 코사인 유사도).

        Raises:
            ValueError: DB가 비어 있을 때.
        """
        if not self._vectors:
            raise ValueError("기준 DB가 비어 있습니다.")
        query = vector.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm >= 1e-8:
            query = query / norm  # 질의를 정규화해 진짜 코사인 유사도 보장
        matrix = np.stack(self._vectors)
        scores = matrix @ query  # 저장 벡터 정규화 가정 → 코사인
        best = int(np.argmax(scores))
        return self._ids[best], float(scores[best])

    def search(self, vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """가장 유사한 상위 k개 기준을 코사인 유사도 내림차순으로 반환한다.

        Args:
            vector: 질의 임베딩 (정규화 가정).
            k: 반환할 최대 개수. DB 크기보다 크면 DB 크기로 클램프된다.

        Returns:
            (ref_id, 코사인 유사도) 리스트. 유사도 내림차순 정렬.

        Raises:
            ValueError: DB가 비어 있을 때.
        """
        if not self._vectors:
            raise ValueError("기준 DB가 비어 있습니다.")
        query = vector.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm >= 1e-8:
            query = query / norm  # 질의 정규화 → 진짜 코사인 유사도
        matrix = np.stack(self._vectors)
        scores = matrix @ query  # 저장 벡터 정규화 가정 → 코사인
        k = min(k, len(self._ids))
        # 상위 k개 인덱스를 내림차순으로 정렬
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        """npz로 저장한다.

        Args:
            path: 저장 경로.
        """
        np.savez(path, ids=np.array(self._ids), vectors=np.stack(self._vectors))

    @classmethod
    def load(cls, path: str) -> "ReferenceDB":
        """npz에서 로드한다.

        Args:
            path: npz 파일 경로.

        Returns:
            복원된 ReferenceDB.
        """
        data = np.load(path, allow_pickle=True)
        db = cls()
        for ref_id, vec in zip(data["ids"], data["vectors"]):
            db.add(str(ref_id), vec)
        return db

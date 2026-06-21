"""전체 피드백 파이프라인 오케스트레이션."""

from dataclasses import dataclass


@dataclass
class FeedbackResult:
    """파이프라인 출력."""

    ref_id: str
    similarity: float
    passed: bool
    feedback: str


class FeedbackPipeline:
    """임베딩 매칭 + Gemini 피드백을 묶는다."""

    def __init__(
        self,
        embedder,
        db,
        feedback,
        ref_paths: dict[str, str],
        threshold: float = 0.8,
    ) -> None:
        """파이프라인을 구성한다.

        Args:
            embedder: `embed(path) -> np.ndarray` 인터페이스를 가진 임베더.
            db: `match(vector) -> (ref_id, similarity)` 인터페이스를 가진 DB.
            feedback: `generate(user_video_path, reference_video_path)` 피드백 생성기.
            ref_paths: ref_id → 기준 영상 경로 매핑.
            threshold: 합격 판정 코사인 유사도 임계값.
        """
        self.embedder = embedder
        self.db = db
        self.feedback = feedback
        self.ref_paths = ref_paths
        self.threshold = threshold

    def run(self, user_video_path: str) -> FeedbackResult:
        """사용자 영상을 매칭하고 피드백을 생성한다.

        Args:
            user_video_path: 사용자 영상 경로.

        Returns:
            매칭 결과와 텍스트 피드백을 담은 FeedbackResult.
        """
        vector = self.embedder.embed(user_video_path)
        ref_id, similarity = self.db.match(vector)
        passed = similarity >= self.threshold
        ref_path = self.ref_paths.get(ref_id)
        feedback_text = self.feedback.generate(
            user_video_path=user_video_path,
            reference_video_path=ref_path,
        )
        return FeedbackResult(
            ref_id=ref_id,
            similarity=similarity,
            passed=passed,
            feedback=feedback_text,
        )

import numpy as np

from video_feedback.pipeline import FeedbackPipeline, FeedbackResult


class _FakeEmbedder:
    def embed(self, path, num_frames=16):
        return np.array([1.0, 0.0], dtype=np.float32)


class _FakeDB:
    def match(self, vector):
        return "ref_a", 0.95


class _FakeFeedback:
    def generate(self, user_video_path, reference_video_path=None):
        return "0:05 팔꿈치가 내려갔습니다. 어깨 높이로 유지하세요."


def test_pipeline_run_passes_threshold():
    pipe = FeedbackPipeline(
        embedder=_FakeEmbedder(),
        db=_FakeDB(),
        feedback=_FakeFeedback(),
        ref_paths={"ref_a": "ref_a.mp4"},
        threshold=0.8,
    )
    result = pipe.run("user.mp4")
    assert isinstance(result, FeedbackResult)
    assert result.ref_id == "ref_a"
    assert result.passed is True
    assert "팔꿈치" in result.feedback

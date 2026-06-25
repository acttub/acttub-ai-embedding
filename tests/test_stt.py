import glob

import pytest


def _sample_clip():
    clips = sorted(glob.glob("연기영상/clips/*.mp4"))
    return clips[0] if clips else None


@pytest.mark.gpu
def test_transcribe_returns_text():
    if _sample_clip() is None:
        pytest.skip("클립 데이터 없음")
    from video_feedback.stt import Transcriber

    text = Transcriber().transcribe(_sample_clip())
    assert isinstance(text, str)
    # 연기영상엔 대사가 있으므로 비어있지 않아야 한다.
    assert len(text.strip()) > 0


@pytest.mark.gpu
def test_transcribe_missing_audio_returns_empty(tmp_path):
    import cv2
    import numpy as np

    # 오디오 스트림 없는 무음 영상 → 빈 문자열
    path = tmp_path / "silent.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(10):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()

    from video_feedback.stt import Transcriber

    assert Transcriber().transcribe(str(path)) == ""

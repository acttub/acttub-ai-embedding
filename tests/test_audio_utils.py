import glob

import numpy as np
import pytest

from video_feedback.audio_utils import load_audio


def _sample_clip():
    clips = sorted(glob.glob("연기영상/clips/*.mp4"))
    return clips[0] if clips else None


@pytest.mark.skipif(_sample_clip() is None, reason="클립 데이터 없음")
def test_load_audio_returns_mono_float_waveform():
    wav = load_audio(_sample_clip(), target_sr=48000)
    assert wav.ndim == 1  # mono로 다운믹스
    assert wav.dtype == np.float32
    assert len(wav) > 48000  # 클립이 ~60초라 1초보다 훨씬 길어야


def test_load_audio_invalid_path():
    with pytest.raises(ValueError):
        load_audio("does_not_exist.mp4")

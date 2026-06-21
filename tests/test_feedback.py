from video_feedback.feedback import build_prompt


def test_build_prompt_mentions_timestamp():
    prompt = build_prompt(reference_available=True)
    assert "시간" in prompt or "초" in prompt
    assert "기준" in prompt


def test_build_prompt_without_reference():
    prompt = build_prompt(reference_available=False)
    assert "기준" not in prompt or "없" in prompt

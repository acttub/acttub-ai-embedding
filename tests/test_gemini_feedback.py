import pytest

from video_feedback.gemini_feedback import (
    build_prompt,
    generate_feedback,
    load_env,
    parse_feedback_json,
    render_card_markdown,
)


class _FakeFile:
    def __init__(self, name):
        self.name = name
        self.state = type("S", (), {"name": "ACTIVE"})()


class _FakeFiles:
    def __init__(self):
        self.uploaded = []

    def upload(self, *, file):
        self.uploaded.append(file)
        return _FakeFile(file)

    def get(self, *, name):
        return _FakeFile(name)


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return type("R", (), {"text": '{"overallStrength": {"text": "ok"}}'})()


class _FakeClient:
    def __init__(self):
        self.files = _FakeFiles()
        self.models = _FakeModels()


def test_build_prompt_fills_all_four_fields():
    text = build_prompt(
        media="영화",
        situation="이별 통보를 받는 장면",
        character="20대 직장인",
        subtext="붙잡고 싶지만 자존심 때문에 참는다",
    )
    # 배우가 준 설정 4개가 모두 프롬프트에 박혀야 한다
    assert "영화" in text
    assert "이별 통보를 받는 장면" in text
    assert "20대 직장인" in text
    assert "붙잡고 싶지만 자존심 때문에 참는다" in text
    # 포맷팅 후 %s 자리표시자가 남아있으면 안 된다
    assert "%s" not in text


def test_build_prompt_empty_fields_marked_missing():
    # 빈 설정은 결측 표시("비어 있음")로 채워 모델이 설정으로 오인하지 않게 한다
    text = build_prompt(media="", situation="", character="", subtext="")
    assert "%s" not in text
    assert "비어 있음" in text


def test_parse_feedback_json_strips_markdown_codeblock():
    raw = '```json\n{"overallStrength": {"text": "전반적으로 좋아요"}}\n```'
    out = parse_feedback_json(raw)
    assert out["overallStrength"]["text"] == "전반적으로 좋아요"


def test_parse_feedback_json_plain_object():
    raw = '{"feedbackCards": [{"order": 1, "title": "톤 낮추기"}]}'
    out = parse_feedback_json(raw)
    assert out["feedbackCards"][0]["title"] == "톤 낮추기"


def test_parse_feedback_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_feedback_json("이건 JSON이 아니에요")


_SAMPLE_CARD = {
    "overallStrength": {"text": "전반적으로 시선 처리가 안정적이었어요"},
    "feedbackCards": [
        {
            "order": 1,
            "title": "끝음절을 삼키지 않기",
            "observations": [
                {
                    "timecode": "0:12",
                    "text": "대사: 끝음절 '발'의 음량이 줄어 거의 안 들렸어요",
                },
                {"timecode": "전반", "text": "몸: 어깨가 귀 쪽으로 올라가 있었어요"},
            ],
            "cause": "왜냐면 긴장이 호흡을 위로 끌어올린 것 같아요",
            "practiceSteps": [
                "첫 문장 전 2초간 숨을 내쉬기",
                "끝음절을 3번 또박또박 반복",
                "바로 녹화",
            ],
            "expectedEffect": "그럼 대사 끝까지 또렷이 전달돼요",
        }
    ],
}


def test_render_card_markdown_includes_key_fields():
    md = render_card_markdown(_SAMPLE_CARD)
    assert "전반적으로 시선 처리가 안정적이었어요" in md  # 강점
    assert "끝음절을 삼키지 않기" in md  # 카드 제목
    assert "0:12" in md  # 관찰 타임코드
    assert "왜냐면" in md  # 원인
    assert "끝음절을 3번" in md  # 연습 단계
    assert "그럼 대사 끝까지" in md  # 기대 효과


def test_render_card_markdown_numbers_practice_steps():
    md = render_card_markdown(_SAMPLE_CARD)
    # 연습 단계는 번호가 매겨져야 한다 (배우가 순서대로 따라하도록)
    assert "1." in md
    assert "3." in md


def test_render_card_markdown_handles_empty_cards():
    # 충돌 없음(layer=none) → 강점만 있고 카드 비어도 깨지지 않아야 한다
    md = render_card_markdown(
        {"overallStrength": {"text": "좋아요"}, "feedbackCards": []}
    )
    assert "좋아요" in md


def test_load_env_sets_environ(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nFOO_KEY=bar123\n\n", encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    load_env(str(p))
    import os

    assert os.environ["FOO_KEY"] == "bar123"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("FOO_KEY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO_KEY", "preset")
    load_env(str(p))
    import os

    # 이미 설정된 환경변수는 덮어쓰지 않는다
    assert os.environ["FOO_KEY"] == "preset"


def test_generate_feedback_input_only_uploads_one_video():
    client = _FakeClient()
    out = generate_feedback("input.mp4", None, situation="이별 장면", client=client)
    assert out["overallStrength"]["text"] == "ok"
    # 입력 영상 1개만 업로드
    assert client.files.uploaded == ["input.mp4"]
    contents = client.models.calls[0]["contents"]
    # contents = [입력영상, 프롬프트] — 영상 1개 + 마지막은 프롬프트 문자열
    assert len(contents) == 2
    assert isinstance(contents[-1], str)
    assert "C레벨" not in contents[-1]  # 참조 안내문(preamble) 없음


def test_generate_feedback_with_ref_puts_cpro_first_and_adds_preamble():
    client = _FakeClient()
    generate_feedback("input.mp4", "cpro.mp4", situation="이별 장면", client=client)
    # C-pro가 먼저, 입력이 나중에 업로드된다
    assert client.files.uploaded == ["cpro.mp4", "input.mp4"]
    contents = client.models.calls[0]["contents"]
    # contents = [C-pro영상, 입력영상, 프롬프트]
    assert len(contents) == 3
    assert isinstance(contents[-1], str)
    # 프롬프트 앞에 C-pro를 "기준선"으로만 쓰라는 참조 안내(preamble)가 붙는다
    assert "C레벨" in contents[-1]
    assert "기준선" in contents[-1]


def test_generate_feedback_default_model_is_flash():
    client = _FakeClient()
    generate_feedback("input.mp4", None, client=client)
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"

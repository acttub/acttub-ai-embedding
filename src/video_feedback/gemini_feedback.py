"""Gemini로 연기영상 피드백 카드(JSON)를 생성한다.

prompt.txt(수석 연기코치 프롬프트)에 배우 설정을 채워 영상과 함께 Gemini에
보내고, 반환된 피드백 카드 JSON을 파싱한다. 유사 C-pro 영상을 "전문가 기준"
으로 함께 첨부하는 경로도 지원한다.
"""

import json
import os

PROMPT_PATH = "prompt.txt"
DEFAULT_MODEL = "gemini-2.5-flash"
_MISSING = "비어 있음"  # 빈 설정을 결측으로 표시 (prompt.txt가 결측 문구로 인식)


def build_prompt(
    media: str = "",
    situation: str = "",
    character: str = "",
    subtext: str = "",
    prompt_path: str = PROMPT_PATH,
) -> str:
    """배우 설정 4개를 prompt.txt의 자리표시자(%s)에 채워 최종 프롬프트를 만든다.

    Args:
        media: 매체/장르.
        situation: 상황.
        character: 인물 설정.
        subtext: 서브텍스트(속마음).
        prompt_path: 프롬프트 템플릿 경로.

    Returns:
        설정이 채워진 프롬프트 문자열. 빈 값은 "비어 있음"으로 채운다.
    """
    with open(prompt_path, encoding="utf-8") as f:
        template = f.read()
    fields = tuple(
        v.strip() if v and v.strip() else _MISSING
        for v in (media, situation, character, subtext)
    )
    return template % fields


def parse_feedback_json(text: str) -> dict:
    """Gemini 응답 텍스트에서 피드백 카드 JSON 객체를 파싱한다.

    모델이 종종 ```json ... ``` 코드블록으로 감싸 반환하므로 벗겨낸다.

    Args:
        text: 모델 응답 원문.

    Returns:
        파싱된 JSON 객체(dict).

    Raises:
        ValueError: JSON 객체를 찾거나 파싱할 수 없을 때.
    """
    s = text.strip()
    if s.startswith("```"):
        # ```json\n...\n``` → 첫 줄(펜스)과 마지막 펜스 제거
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # 혹시 앞뒤에 설명문이 붙어도 첫 '{' ~ 마지막 '}'만 취한다
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"JSON 객체를 찾을 수 없습니다: {text[:80]!r}")
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}") from e


def load_env(path: str = ".env") -> None:
    """간단한 .env 로더 — KEY=VALUE 줄을 환경변수로 올린다 (외부 의존성 없음).

    이미 설정된 환경변수는 덮어쓰지 않는다(setdefault). 파일이 없으면 조용히
    넘어간다.

    Args:
        path: .env 파일 경로.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def render_card_markdown(data: dict) -> str:
    """피드백 카드 JSON을 사람이 읽기 좋은 마크다운으로 렌더한다.

    Args:
        data: ``overallStrength`` / ``feedbackCards`` 구조의 카드 JSON.

    Returns:
        마크다운 문자열.
    """
    lines: list[str] = []
    strength = (data.get("overallStrength") or {}).get("text", "")
    if strength:
        lines.append(f"**💪 강점**\n\n{strength}\n")

    for card in data.get("feedbackCards") or []:
        title = card.get("title", "")
        lines.append(f"### 🎯 {title}")

        obs = card.get("observations") or []
        if obs:
            lines.append("\n**관찰**")
            for o in obs:
                tc = o.get("timecode", "")
                txt = o.get("text", "")
                lines.append(f"- `{tc}` {txt}")

        cause = card.get("cause", "")
        if cause:
            lines.append(f"\n**원인** — {cause}")

        steps = card.get("practiceSteps") or []
        if steps:
            lines.append("\n**연습**")
            for i, s in enumerate(steps, 1):
                lines.append(f"{i}. {s}")

        effect = card.get("expectedEffect", "")
        if effect:
            lines.append(f"\n↗ {effect}")

    return "\n".join(lines).strip()


def _client(api_key: str | None = None):
    """genai 클라이언트를 만든다 (지연 import — 호출 시점에만 SDK 필요)."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다 (.env 확인).")
    return genai.Client(api_key=key)


def _upload_active(client, path: str, poll_interval: float = 1.0):
    """영상을 업로드하고 ACTIVE 상태가 될 때까지 기다린다."""
    import time

    f = client.files.upload(file=path)
    while f.state.name == "PROCESSING":
        time.sleep(poll_interval)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"영상 업로드 실패({path}): state={f.state.name}")
    return f


# C-pro 참조 영상을 "평가 대상이 아닌 전문가 기준"으로 못박는 안내문
_REF_PREAMBLE = (
    "[참고용 전문가(C레벨) 영상]\n"
    "아래 첫 번째 영상은 같은 결의 전문 배우 연기입니다. 이 영상은 평가·피드백 "
    "대상이 아니라 '잘된 연기는 이런 모습'이라는 기준선으로만 참고하세요. "
    "피드백 카드는 오직 두 번째 [배우 영상]에 대해서만, 그 영상에서 실제로 "
    "보이고 들린 것에 근거해 작성합니다.\n\n"
    "[배우 영상]\n"
)


def generate_feedback(
    input_video: str,
    ref_video: str | None = None,
    *,
    media: str = "",
    situation: str = "",
    character: str = "",
    subtext: str = "",
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    client=None,
) -> dict:
    """입력 영상(+선택적 C-pro 참조)으로 피드백 카드를 생성한다.

    Args:
        input_video: 피드백 대상(배우) 영상 경로.
        ref_video: 함께 줄 유사 C-pro 영상 경로. None이면 입력 영상만 보낸다.
        media/situation/character/subtext: 배우가 준 설정.
        model: Gemini 모델 id.
        api_key: 직접 전달할 API 키 (없으면 환경변수).
        client: 미리 만든 genai 클라이언트 (테스트 주입용).

    Returns:
        파싱된 피드백 카드 JSON(dict).
    """
    client = client or _client(api_key)
    prompt = build_prompt(media, situation, character, subtext)

    contents = []
    if ref_video is not None:
        contents.append(_upload_active(client, ref_video))
        prompt = _REF_PREAMBLE + prompt
    contents.append(_upload_active(client, input_video))
    contents.append(prompt)

    resp = client.models.generate_content(model=model, contents=contents)
    return parse_feedback_json(resp.text)

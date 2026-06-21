"""Gemini 기반 영상 피드백 생성."""

import time

from google import genai


def build_prompt(reference_available: bool) -> str:
    """피드백 프롬프트를 생성한다.

    Args:
        reference_available: 기준 영상 동봉 여부.

    Returns:
        Gemini에 전달할 한국어 프롬프트 문자열.
    """
    base = (
        "이 영상에서 잘못된 부분을 찾아주세요. "
        "각 문제마다 (1) 몇 초 지점인지(예: 0:12), "
        "(2) 무엇이 잘못됐는지, (3) 어떻게 고쳐야 하는지를 한국어로 설명하세요."
    )
    if reference_available:
        base += (
            " 첫 번째 영상이 올바른 기준이고, 두 번째 영상이 사용자 영상입니다. "
            "기준과 비교해 차이를 짚어주세요."
        )
    return base


class GeminiFeedback:
    """Gemini로 영상 피드백을 생성한다."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        """피드백 생성기를 초기화한다.

        Args:
            api_key: Gemini API 키.
            model: 사용할 모델 이름.
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _upload_active(self, path: str, timeout: float = 120.0, interval: float = 1.0):
        """파일을 업로드하고 ACTIVE 상태가 될 때까지 대기한다.

        Args:
            path: 업로드할 파일 경로.
            timeout: 최대 대기 시간(초).
            interval: 폴링 간격(초).

        Returns:
            ACTIVE 상태가 된 업로드 파일 객체.

        Raises:
            RuntimeError: 처리 실패 또는 타임아웃 시.
        """
        uploaded = self.client.files.upload(file=path)
        deadline = time.monotonic() + timeout
        while uploaded.state and uploaded.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                raise RuntimeError(f"파일 처리 시간 초과: {path}")
            time.sleep(interval)
            uploaded = self.client.files.get(name=uploaded.name)
        if uploaded.state and uploaded.state.name == "FAILED":
            raise RuntimeError(f"파일 처리 실패: {path}")
        return uploaded

    def generate(
        self,
        user_video_path: str,
        reference_video_path: str | None = None,
    ) -> str:
        """영상을 Gemini에 전달해 텍스트 피드백을 받는다.

        Args:
            user_video_path: 사용자 영상 경로.
            reference_video_path: 기준 영상 경로(선택).

        Returns:
            Gemini가 생성한 텍스트 피드백.
        """
        parts = []
        if reference_video_path is not None:
            parts.append(self._upload_active(reference_video_path))
        parts.append(self._upload_active(user_video_path))
        parts.append(build_prompt(reference_video_path is not None))

        response = self.client.models.generate_content(
            model=self.model, contents=parts
        )
        return response.text

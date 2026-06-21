# 비디오 피드백 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자 영상을 받아 기준 영상과 비교하고, Gemini로 "어디가 잘못됐고 어떻게 고쳐라" 피드백을 생성하는 MVP 파이프라인 구축.

**Architecture:** VideoMAE v2로 영상을 단일 벡터로 임베딩해 기준 DB와 매칭/유사도 점수를 내고, 원본 영상을 Gemini에 통째로 넘겨 위치 탐지+설명을 받는다. 임베딩은 위치 탐지를 하지 않는다.

**Tech Stack:** Python 3.11+, PyTorch, transformers(VideoMAE v2), decord/opencv(영상 로딩), numpy, google-genai(Gemini), pytest, uv

## Global Constraints

- 패키지 매니저: `uv` (uv venv 기반). 새 패키지는 `uv add <pkg>`.
- Python 3.11+, PyTorch GPU(RTX 4060, VRAM 8GB) 추론 가정.
- 코드 스타일: PEP8, 스페이스 4칸, double quote, 타입 힌트 항상, Google 스타일 docstring, ruff 포매팅.
- 커밋은 각 Task 끝에서 수행 (사용자가 git 사용 결정 시). git 미사용이면 커밋 스텝 생략.
- MVP 기본값(열린 질문): 영상 ≤ 60초 가정, 도메인 혼합(사람 동작+일반 장면), 유사도 임계값은 설정값으로 노출.

---

### Task 0: 프로젝트 부트스트랩

**Files:**
- Create: `pyproject.toml`
- Create: `video_feedback/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: 없음
- Produces: `video_feedback` 패키지, `uv` 가상환경, pytest 실행 가능 상태

- [ ] **Step 1: uv 프로젝트 초기화**

Run:
```bash
cd /c/Users/RYU/projects/video-feedback
uv init --package --name video-feedback
uv add --dev pytest
```
Expected: `pyproject.toml`, `.venv/` 생성

- [ ] **Step 2: 디렉터리/패키지 파일 생성**

`video_feedback/__init__.py` (빈 파일), `tests/__init__.py` (빈 파일) 생성.

- [ ] **Step 3: smoke 테스트로 환경 검증**

`tests/test_smoke.py`:
```python
def test_import_package():
    import video_feedback  # noqa: F401
```

- [ ] **Step 4: 테스트 실행**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS

---

### Task 1: 영상 로딩 / 프레임 샘플링

**Files:**
- Create: `video_feedback/video_utils.py`
- Test: `tests/test_video_utils.py`

**Interfaces:**
- Consumes: 없음
- Produces: `load_frames(path: str, num_frames: int = 16) -> np.ndarray` — 균등 샘플된 프레임 배열, shape `(num_frames, H, W, 3)`, dtype uint8(RGB)

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_video_utils.py`:
```python
import numpy as np
from video_feedback.video_utils import load_frames


def test_load_frames_shape(tmp_path):
    video_path = _make_dummy_video(tmp_path)  # 아래 헬퍼
    frames = load_frames(str(video_path), num_frames=8)
    assert frames.shape[0] == 8
    assert frames.shape[-1] == 3
    assert frames.dtype == np.uint8


def _make_dummy_video(tmp_path):
    import cv2
    path = tmp_path / "dummy.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
    )
    for _ in range(30):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    return path
```

- [ ] **Step 2: 의존성 추가 + 테스트 실패 확인**

Run:
```bash
uv add opencv-python numpy
uv run pytest tests/test_video_utils.py -v
```
Expected: FAIL ("No module named ... load_frames" 또는 ImportError)

- [ ] **Step 3: 최소 구현**

`video_feedback/video_utils.py`:
```python
"""영상 로딩 및 프레임 샘플링 유틸리티."""

import cv2
import numpy as np


def load_frames(path: str, num_frames: int = 16) -> np.ndarray:
    """영상에서 균등 간격으로 프레임을 샘플링한다.

    Args:
        path: 영상 파일 경로.
        num_frames: 추출할 프레임 수.

    Returns:
        shape (num_frames, H, W, 3), dtype uint8, RGB 순서의 배열.

    Raises:
        ValueError: 영상을 열 수 없거나 프레임이 없을 때.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"프레임이 없습니다: {path}")

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"프레임을 읽지 못했습니다: {path}")

    # 부족한 프레임은 마지막 프레임으로 패딩
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames[:num_frames]).astype(np.uint8)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_video_utils.py -v`
Expected: PASS

- [ ] **Step 5: 에러 케이스 테스트 추가**

`tests/test_video_utils.py`에 추가:
```python
import pytest


def test_load_frames_invalid_path():
    with pytest.raises(ValueError):
        load_frames("does_not_exist.mp4")
```
Run: `uv run pytest tests/test_video_utils.py -v` → Expected: PASS

---

### Task 2: VideoMAE v2 임베딩 모듈

**Files:**
- Create: `video_feedback/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `load_frames` (Task 1)
- Produces:
  - `class VideoEmbedder` — `__init__(self, model_name: str = "OpenGVLab/VideoMAEv2-Base", device: str | None = None)`
  - `embed(self, video_path: str) -> np.ndarray` — L2 정규화된 1D float32 벡터 반환

- [ ] **Step 1: 실패 테스트 작성 (모델 로딩은 mock으로 격리)**

`tests/test_embedding.py`:
```python
import numpy as np
from video_feedback.embedding import l2_normalize


def test_l2_normalize_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(v)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_l2_normalize_zero_vector_safe():
    v = np.zeros(4, dtype=np.float32)
    out = l2_normalize(v)
    assert not np.any(np.isnan(out))
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: FAIL (ImportError: l2_normalize)

- [ ] **Step 3: 최소 구현 (정규화 + 임베더 골격)**

```bash
uv add torch transformers
```
`video_feedback/embedding.py`:
```python
"""VideoMAE v2 기반 영상 임베딩."""

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from video_feedback.video_utils import load_frames


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """벡터를 L2 정규화한다 (영벡터 안전)."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class VideoEmbedder:
    """영상을 단일 벡터로 임베딩한다."""

    def __init__(
        self,
        model_name: str = "OpenGVLab/VideoMAEv2-Base",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embed(self, video_path: str, num_frames: int = 16) -> np.ndarray:
        """영상 경로 → L2 정규화된 1D float32 임베딩 벡터."""
        frames = load_frames(video_path, num_frames=num_frames)
        inputs = self.processor(list(frames), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # 마지막 hidden state 평균 풀링 → 영상 단일 벡터
        pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return l2_normalize(pooled.cpu().numpy())
```

> 참고: VideoMAE v2 정확한 모델 ID/프로세서 클래스는 구현 시 transformers 버전에 맞춰 확인. 풀링 방식(mean)도 실제 출력 형태 보고 조정.

- [ ] **Step 4: 정규화 테스트 통과 확인**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: PASS

- [ ] **Step 5: 통합 스모크(옵션, GPU 필요)**

`tests/test_embedding.py`에 추가:
```python
import pytest


@pytest.mark.gpu
def test_embed_returns_unit_vector(tmp_path):
    import cv2
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(30):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()

    from video_feedback.embedding import VideoEmbedder
    emb = VideoEmbedder().embed(str(path))
    assert emb.ndim == 1
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
```
Run (GPU 있을 때만): `uv run pytest tests/test_embedding.py -m gpu -v`
Expected: PASS (모델 다운로드 시간 소요)

`pyproject.toml`에 마커 등록:
```toml
[tool.pytest.ini_options]
markers = ["gpu: GPU/모델 다운로드가 필요한 테스트"]
```

---

### Task 3: 기준 영상 DB

**Files:**
- Create: `video_feedback/reference_db.py`
- Test: `tests/test_reference_db.py`

**Interfaces:**
- Consumes: numpy 벡터 (Task 2의 출력 형식)
- Produces:
  - `class ReferenceDB`
  - `add(self, ref_id: str, vector: np.ndarray) -> None`
  - `match(self, vector: np.ndarray) -> tuple[str, float]` — (가장 유사한 ref_id, 코사인 유사도)
  - `save(self, path: str) -> None` / `load(path: str) -> "ReferenceDB"` (classmethod)

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_reference_db.py`:
```python
import numpy as np
from video_feedback.reference_db import ReferenceDB


def test_match_returns_closest():
    db = ReferenceDB()
    db.add("a", np.array([1.0, 0.0], dtype=np.float32))
    db.add("b", np.array([0.0, 1.0], dtype=np.float32))
    ref_id, score = db.match(np.array([0.9, 0.1], dtype=np.float32))
    assert ref_id == "a"
    assert score > 0.9


def test_save_load_roundtrip(tmp_path):
    db = ReferenceDB()
    db.add("a", np.array([1.0, 0.0], dtype=np.float32))
    p = tmp_path / "db.npz"
    db.save(str(p))
    loaded = ReferenceDB.load(str(p))
    ref_id, _ = loaded.match(np.array([1.0, 0.0], dtype=np.float32))
    assert ref_id == "a"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_reference_db.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 최소 구현**

`video_feedback/reference_db.py`:
```python
"""기준 영상 임베딩 저장/매칭 (numpy 기반)."""

import numpy as np


class ReferenceDB:
    """기준 영상 임베딩을 보관하고 코사인 유사도로 매칭한다."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    def add(self, ref_id: str, vector: np.ndarray) -> None:
        """기준 임베딩을 추가한다 (벡터는 정규화돼 있다고 가정)."""
        self._ids.append(ref_id)
        self._vectors.append(vector.astype(np.float32))

    def match(self, vector: np.ndarray) -> tuple[str, float]:
        """가장 유사한 기준 id와 코사인 유사도를 반환한다."""
        if not self._vectors:
            raise ValueError("기준 DB가 비어 있습니다.")
        matrix = np.stack(self._vectors)
        scores = matrix @ vector.astype(np.float32)  # 정규화 가정 → 코사인
        best = int(np.argmax(scores))
        return self._ids[best], float(scores[best])

    def save(self, path: str) -> None:
        """npz로 저장한다."""
        np.savez(path, ids=np.array(self._ids), vectors=np.stack(self._vectors))

    @classmethod
    def load(cls, path: str) -> "ReferenceDB":
        """npz에서 로드한다."""
        data = np.load(path, allow_pickle=True)
        db = cls()
        for ref_id, vec in zip(data["ids"], data["vectors"]):
            db.add(str(ref_id), vec)
        return db
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_reference_db.py -v`
Expected: PASS

---

### Task 4: Gemini 피드백 모듈

**Files:**
- Create: `video_feedback/feedback.py`
- Test: `tests/test_feedback.py`

**Interfaces:**
- Consumes: 사용자 영상 경로, (선택) 기준 영상 경로
- Produces:
  - `class GeminiFeedback` — `__init__(self, api_key: str, model: str = "gemini-2.0-flash")`
  - `generate(self, user_video_path: str, reference_video_path: str | None = None) -> str`
  - `build_prompt(reference_available: bool) -> str` (모듈 함수, 순수 함수라 단위 테스트 대상)

- [ ] **Step 1: 실패 테스트 작성 (순수 함수만 테스트, API는 mock)**

`tests/test_feedback.py`:
```python
from video_feedback.feedback import build_prompt


def test_build_prompt_mentions_timestamp():
    prompt = build_prompt(reference_available=True)
    assert "시간" in prompt or "초" in prompt
    assert "기준" in prompt


def test_build_prompt_without_reference():
    prompt = build_prompt(reference_available=False)
    assert "기준" not in prompt or "없" in prompt
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_feedback.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 최소 구현**

```bash
uv add google-genai
```
`video_feedback/feedback.py`:
```python
"""Gemini 기반 영상 피드백 생성."""

from google import genai


def build_prompt(reference_available: bool) -> str:
    """피드백 프롬프트를 생성한다.

    Args:
        reference_available: 기준 영상 동봉 여부.
    """
    base = (
        "이 영상에서 잘못된 부분을 찾아주세요. "
        "각 문제마다 (1) 몇 초 지점인지(예: 0:12), "
        "(2) 무엇이 잘못됐는지, (3) 어떻게 고쳐야 하는지를 한국어로 설명하세요."
    )
    if reference_available:
        base += " 첫 번째 영상이 올바른 기준이고, 두 번째 영상이 사용자 영상입니다. 기준과 비교해 차이를 짚어주세요."
    return base


class GeminiFeedback:
    """Gemini로 영상 피드백을 생성한다."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self,
        user_video_path: str,
        reference_video_path: str | None = None,
    ) -> str:
        """영상을 Gemini에 전달해 텍스트 피드백을 받는다."""
        parts = []
        if reference_video_path is not None:
            parts.append(self.client.files.upload(file=reference_video_path))
        parts.append(self.client.files.upload(file=user_video_path))
        parts.append(build_prompt(reference_video_path is not None))

        response = self.client.models.generate_content(
            model=self.model, contents=parts
        )
        return response.text
```

> 참고: google-genai 파일 업로드 후 ACTIVE 상태 대기가 필요할 수 있음(파일 처리). 구현 시 `client.files.get`으로 상태 폴링 추가 고려.

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_feedback.py -v`
Expected: PASS

---

### Task 5: 파이프라인 오케스트레이션

**Files:**
- Create: `video_feedback/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VideoEmbedder`(2), `ReferenceDB`(3), `GeminiFeedback`(4), 기준 id→영상경로 매핑
- Produces:
  - `@dataclass FeedbackResult` — `ref_id: str`, `similarity: float`, `passed: bool`, `feedback: str`
  - `class FeedbackPipeline` — `run(self, user_video_path: str) -> FeedbackResult`

- [ ] **Step 1: 실패 테스트 작성 (의존성 전부 fake 주입)**

`tests/test_pipeline.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 최소 구현**

`video_feedback/pipeline.py`:
```python
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
        self.embedder = embedder
        self.db = db
        self.feedback = feedback
        self.ref_paths = ref_paths
        self.threshold = threshold

    def run(self, user_video_path: str) -> FeedbackResult:
        """사용자 영상 → 매칭 + 피드백."""
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: 전체 테스트 실행**

Run: `uv run pytest -v -m "not gpu"`
Expected: 전체 PASS (gpu 마커 제외)

---

## Self-Review

- **Spec coverage:** 임베딩 모듈(Task 2), 기준 DB(Task 3), Gemini 피드백(Task 4), 출력=점수+텍스트(Task 5 `FeedbackResult`), 임베딩 벡터를 Gemini에 안 넣고 원본 영상 전달(Task 4/5) — 스펙 항목 모두 커버.
- **열린 질문 처리:** 영상 길이/도메인/임계값 → `threshold` 파라미터로 노출, 기본값 0.8. 영상 길이는 Gemini 업로드 한도 내 가정.
- **Type consistency:** `embed() -> np.ndarray`(1D, 정규화) → `ReferenceDB.match(vector)` 입력 일치, `match -> (str, float)` → 파이프라인에서 그대로 사용. 일관됨.
- **확장 경로:** numpy DB → FAISS, 전체 임베딩 → 클립/DTW는 스펙의 향후 확장으로 분리(현 계획 범위 외).

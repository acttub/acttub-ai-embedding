# 진행 상황 / 이어서 작업 (video-feedback)

> 최종 업데이트: 2026-06-25
> 한 줄 요약: **영상 던지면 유사한 전문가(C-pro) 연기영상 top-K를 뽑아주는 검색 시스템.** **얼굴 표정(FER)+음성(멀티모달)** 검색. 동작 + 웹 UI 완성.

---

## 지금 상태 (DONE)

- [x] 깨진 venv 복구 (uv, Python 3.11.15), torch GPU `2.12.1+cu126`, RTX 4060
- [x] `ReferenceDB.search(vector, k)` top-K 검색 (TDD)
- [x] 임베딩 모델: VideoMAE → V-JEPA 2 → **얼굴 표정 FER** (`trpakov/vit-face-expression`, ViT 768d) ← 2026-06-25 교체
  - 프레임별 MTCNN(`facenet-pytorch`)으로 가장 큰 얼굴 크롭 → ViT-FER CLS 768d → 시간평균
  - 얼굴 없는 프레임 스킵, 클립 전체 무얼굴이면 중앙 크롭 폴백
  - V-JEPA 구현은 `embedding_vjepa.py`, 기존 인덱스는 `index_vjepa.npz`로 보존
  - 신규 `face_utils.py`(MTCNN), `index.npz` 재구축(C-pro 34개: video 768d + audio 512d)
- [x] Gradio 웹 UI (`app.py`) — http://127.0.0.1:7860
- [x] **GitHub 푸시**: https://github.com/acttub/acttub-ai-embedding (영상 데이터 제외)
- [x] **오디오 임베딩(CLAP) + 멀티모달 결합 검색** ← 2026-06-21 추가
  - `AudioEmbedder`: CLAP(`laion/clap-htsat-unfused`, 512d). 10초 청크 평균
  - `audio_utils.load_audio`: PyAV로 영상→모노 파형(48kHz)
  - `combine_embeddings`: 영상+음성 가중 결합 (코사인이 `(1-w)*영상 + w*음성`로 분해)
  - `MultiModalReferenceDB`: 영상/음성 벡터 분리 저장 → 런타임 가중치 결합 검색
  - app에 **음성 가중치 슬라이더** (0=영상만 · 1=음성만)
  - `index.npz` 멀티모달 재구축 (C-pro 34개: video 1024d + audio 512d)
- [x] 테스트 23/23 통과

스코프: 원래 "Gemini 영상 피드백" → "유사 영상 검색"으로 축소. `feedback.py`/`pipeline.py`는 미사용 잔재. GPU 전용.

---

## 다음에 켰을 때 — 빠른 시작

```powershell
cd C:\Users\RJS\Desktop\project\video-feedback

# 1) 웹앱 실행 (모델 2개 로딩 ~30초 후 http://127.0.0.1:7860)
py -m uv run python app.py

# 2) CLI 검색 (--w-audio: 음성 가중치 0~1, 기본 0.5)
py -m uv run python scripts/query.py <영상경로> --k 5 --w-audio 0.5

# 3) 인덱스 재구축 (C영상 추가/모델 변경 시에만, 영상+음성 임베딩)
py -m uv run python scripts/build_index.py --pattern "C-pro__*.mp4" --out index.npz

# 4) 테스트 (GPU 테스트 포함하려면 -m gpu)
py -m uv run pytest -q
```

> `uv`가 PATH에 없음 → **`py -m uv ...`** 로 호출 (확인됨). `.venv`엔 pip 없음(uv 프로젝트).

---

## 핵심 구조 / 알아둘 점 (gotcha)

- **기준 영상은 미리 임베딩**(`index.npz`), **질의 영상만 실시간**. C영상 추가 시에만 재구축.
- torch는 반드시 **cu126** 인덱스(`pyproject.toml`의 `[tool.uv.sources]`). `facenet-pytorch` 추가해도 cu126 유지됨(확인).
- **표정 임베딩(FER)**: 프레임별 MTCNN 얼굴크롭 → `trpakov/vit-face-expression`에 `output_hidden_states=True` → `hidden_states[-1][:,0]`(CLS, **768d**) → 시간평균. logits(7d) 안 씀.
  - 얼굴 검출 실패 프레임 스킵, 클립 전체 무얼굴이면 `center_crop_square` 폴백. `embed_segments`는 프레임 임베딩을 시간 구간으로 묶어 평균(토큰 reshape 불필요).
  - FER 임베딩은 얼굴끼리 몰려서 코사인이 전반적으로 높게(덜 벌어지게) 나옴 → 변별력 정성 검증 필요.
- (구버전) V-JEPA: `embedding_vjepa.py` 보존. `model.get_vision_features`, 토큰 8192=32(시간)×256(공간). 인덱스는 `index_vjepa.npz`.
- **CLAP은 10초 초과 입력을 랜덤 구간으로 잘라(rand_trunc) 비결정적** → 10초 청크로 나눠 평균내서 회피 + 전체 반영.
- CLAP `get_audio_features`는 transformers 5.x에서 output 객체 반환 → `.pooler_output` (512d, joint space).
- 결합: 각 모달 L2정규화 후 `√w` 곱해 concat → 자동 단위벡터, 코사인 가중 분해.
- 검색은 numpy brute-force. 벡터DB 불필요(수십~수천 규모).

---

## 파일 맵

| 파일 | 역할 |
|---|---|
| `app.py` | Gradio 웹 UI (음성 가중치 슬라이더 포함) |
| `scripts/build_index.py` | 기준 영상 → 멀티모달 `index.npz` |
| `scripts/query.py` | CLI 유사검색 (`--w-audio`) |
| `scripts/probe_tokens.py` | (일회성) V-JEPA 토큰 레이아웃 실측 |
| `scripts/probe_audio.py` | (일회성) 클립 오디오 존재 실측 |
| `src/video_feedback/embedding.py` | **표정(FER) 임베딩** (`VideoEmbedder`) + `combine_embeddings` |
| `src/video_feedback/face_utils.py` | MTCNN 얼굴 크롭 (`FaceDetector`, `center_crop_square`) |
| `src/video_feedback/embedding_vjepa.py` | (보존) 구버전 V-JEPA 임베딩 |
| `src/video_feedback/audio_embedding.py` | CLAP 오디오 임베딩 (`AudioEmbedder`) |
| `src/video_feedback/audio_utils.py` | PyAV 오디오 파형 추출 (`load_audio`) |
| `src/video_feedback/multimodal.py` | 영상/음성 결합 검색 (`MultiModalReferenceDB`) |
| `src/video_feedback/reference_db.py` | numpy 코사인 검색 (`match`, `search`) |
| `src/video_feedback/video_utils.py` | opencv 프레임 샘플링 (`load_frames`) |
| `index.npz` | C-pro 34개 멀티모달 임베딩 (**video 768d** + audio 512d) |
| `index_vjepa.npz` | (백업) 구버전 V-JEPA 인덱스 (video 1024d + audio 512d) |
| `연기영상/clips/` | 영상 데이터 (A-amateur / B-student / C-pro, git 제외) |
| `feedback.py`, `pipeline.py` | (미사용, Gemini 시절 잔재) |

---

## 다음 후보 (TODO)

- [x] **"왜 비슷한지 설명" v1 — 구간 매칭** (`explain.py`, FER 프레임 임베딩 기반으로 재작동)
- [ ] "왜 비슷한지" v2 — CLAP 텍스트 질의로 음성 속성 설명 ("격앙된 목소리" 등)
- [ ] FER 표정 매칭 변별력 정성 검증 (아마추어 여러 개, 코사인 잘 벌어지는지)
- [ ] 아마추어 영상 여러 개로 매칭 품질 정성 검증 (영상/음성 각각)
- [ ] `build_index.py` 증분 모드 (이미 인덱스에 있는 건 스킵)
- [ ] 웹 UI: 질의 영상 동시 표시 / top-K 결과 그리드 재생
- [ ] (정리) `scripts/probe_*.py` 일회성 검증 스크립트 — 남길지/지울지 결정

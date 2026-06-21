# 진행 상황 / 이어서 작업 (video-feedback)

> 최종 업데이트: 2026-06-20
> 한 줄 요약: **영상 던지면 유사한 전문가(C-pro) 연기영상 top-K를 뽑아주는 검색 시스템.** 동작 + 웹 UI까지 완성.

---

## 지금 상태 (DONE)

- [x] 깨진 venv 복구 (uv 설치 + `uv sync`, Python 3.11.15)
- [x] torch GPU 전환 → `2.12.1+cu126`, RTX 4060 인식
- [x] `ReferenceDB.search(vector, k)` top-K 검색 추가 (TDD)
- [x] 기준 영상 인덱스 구축: C-pro 34개 → `index.npz`
- [x] 임베딩 모델 업그레이드: VideoMAE → **V-JEPA 2** (`facebook/vjepa2-vitl-fpc64-256`, 1024차원, 64프레임)
- [x] Gradio 웹 UI (`app.py`) — http://127.0.0.1:7860
- [x] 테스트 13/13 통과
- [x] HTML 보고서 `report-2026-06-20.html`

스코프: 원래 "Gemini 영상 피드백"이었으나 **"유사 영상 검색"으로 축소**. Gemini(`feedback.py`)/`pipeline.py`는 미사용(삭제 안 함). GPU 전용.

---

## 다음에 켰을 때 — 빠른 시작

```powershell
# 위치
cd C:\Users\RJS\Desktop\project\video-feedback

# 1) 웹앱 실행 (모델 로딩 ~15초 후 http://127.0.0.1:7860)
uv run python app.py

# 2) CLI 검색
uv run python scripts/query.py <영상경로> --k 5

# 3) 인덱스 재구축 (C영상 추가하거나 모델 바꿨을 때만)
uv run python scripts/build_index.py --pattern "C-pro__*.mp4" --out index.npz

# 4) 테스트
uv run pytest -q
```

> `uv`가 PATH에 없으면 `py -3.13 -m uv ...`로 호출.

---

## 핵심 구조 / 알아둘 점 (gotcha)

- **기준 영상은 미리 임베딩**(`index.npz`), **질의 영상만 실시간** 임베딩 → 검색 빠름. C영상 추가 시에만 재구축.
- venv는 원래 다른 계정(`C:\Users\RYU`)에서 만들어져 깨졌었음 → uv로 재생성함. PATH에 uv 없으면 `py -3.13 -m uv`.
- torch는 반드시 **cu126** 인덱스(`pyproject.toml`의 `[tool.uv.sources]`). cu124엔 torch 2.12.1 없음.
- VJEPA2 임베딩: `model.get_vision_features(**inputs)`, 입력은 `(T,C,H,W)` 텐서(우리 `load_frames`는 opencv, torchcodec 불필요), 64프레임.
- 검색은 numpy brute-force. **벡터DB 불필요** (수십~수천 규모). 10만+ 가면 FAISS, 서비스화하면 Qdrant/pgvector.
- 병목은 검색이 아니라 임베딩(질의당 ~2초, GPU). 웹앱 켜두면 GPU VRAM ~1.77GB 점유.

---

## 파일 맵

| 파일 | 역할 |
|---|---|
| `app.py` | Gradio 웹 UI |
| `scripts/build_index.py` | 기준 영상 → `index.npz` |
| `scripts/query.py` | CLI 유사검색 |
| `src/video_feedback/embedding.py` | VJEPA2 임베딩 (`VideoEmbedder`) |
| `src/video_feedback/reference_db.py` | numpy 코사인 검색 (`match`, `search`) |
| `src/video_feedback/video_utils.py` | opencv 프레임 샘플링 (`load_frames`) |
| `index.npz` | C-pro 34개 VJEPA2 임베딩 |
| `연기영상/clips/` | 영상 데이터 (A-amateur / B-student / C-pro) |
| `report-2026-06-20.html` | 작업 보고서 |
| `feedback.py`, `pipeline.py` | (미사용, Gemini 시절 잔재) |

---

## 다음 후보 (TODO)

- [ ] 아마추어 영상 여러 개로 매칭 품질 정성 검증
- [ ] `build_index.py` 증분 모드 (이미 인덱스에 있는 건 스킵)
- [ ] 웹 UI 개선: 질의 영상 동시 표시 / top-K 결과 그리드 재생
- [ ] 규모 커지면 FAISS로 전환
- [ ] (선택) `연기영상/clips/MANIFEST.md` 실제 클립으로 채우기 — 지금 빈 껍데기

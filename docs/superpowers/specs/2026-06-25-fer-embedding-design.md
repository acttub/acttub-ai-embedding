# V-JEPA → 얼굴 표정(FER) 임베딩 교체 설계

> 작성: 2026-06-25
> 한 줄: 연기영상 유사검색의 영상 임베딩을 V-JEPA 2(장면/모션) → **얼굴 표정 임베딩**으로 교체.

## 목표 / 의도

연기영상은 **표정·감정**이 핵심인데 V-JEPA 2는 영상 전체의 시공간 모션/장면을
잡지 표정에 특화돼 있지 않다. 영상 임베딩을 표정 인식(FER) 모델 기반으로 바꿔
"표정이 닮은 전문가 영상"을 더 정확히 찾는다. 음성(CLAP) 모달은 유지.

## 결정 사항 (승인됨)

- **통합 방식**: V-JEPA **완전 교체**. 영상 임베딩 = 표정 임베딩. 음성(CLAP 512d) 유지.
- **모델**: `trpakov/vit-face-expression` (ViT-base-patch16-224-in21k 파인튜닝, 표정 7종).
  - 임베딩 소스: 감정 logits(7d)가 아니라 **last hidden state의 CLS 토큰 768d** (풍부·코사인 호환).
  - 검증 완료: logits (1,7), last_hidden (1,197,768), labels = {angry, disgust, fear, happy, neutral, sad, surprise}.
- **얼굴 검출기**: `facenet-pytorch` MTCNN (GPU). 프레임별 **가장 큰 얼굴 1개** 크롭.
  - 검증 완료: torch 2.12.1+cu126 유지, MTCNN cuda 로드 OK, 얼굴 없으면 `None`.
- **얼굴 없는 프레임 정책**: 해당 프레임 스킵. 클립 전체에 얼굴이 하나도 없으면 → **중앙 크롭 폴백** + 경고.
- **V-JEPA 보존**: 기존 `embedding.py` 로직을 `embedding_vjepa.py`로 복사 보존(삭제 금지).
- **인덱스**: 기존 `index.npz` → `index_vjepa.npz`로 백업 후 표정 임베딩으로 재구축.

## 아키텍처 (3단)

1. **프레임 샘플링** — 기존 `video_utils.load_frames(path, num_frames)` 재사용. (T,H,W,3) RGB.
2. **얼굴 검출/크롭** — `face_utils.FaceDetector.crop_largest(frame) -> 크롭 | None`. MTCNN.
3. **표정 임베딩** — 크롭 → ViT-FER → CLS hidden 768d. 프레임들 평균 → L2 정규화.

데이터 흐름: 영상 → N프레임 → (프레임별 얼굴크롭 → 768d) → 시간평균 → 768d 단위벡터
→ 기존 `combine_embeddings`로 음성(512d) 결합 (차원 무관 함수, 그대로 동작).

## 컴포넌트 / 인터페이스

| 유닛 | 역할 | 의존 |
|---|---|---|
| `face_utils.FaceDetector` | 프레임 RGB → 가장 큰 얼굴 크롭(uint8) 또는 None | facenet-pytorch(MTCNN), torch |
| `embedding.VideoEmbedder` (재구현) | 영상 → 768d 표정 임베딩 (`embed`, `embed_segments`) | FaceDetector, transformers ViT-FER |
| `embedding.combine_embeddings`, `l2_normalize` | (변경 없음) | numpy |

공개 이름 `VideoEmbedder` 유지 → build_index/app/query 임포트 churn 최소화.

### embed / embed_segments

- `embed(path, num_frames=64) -> (768,)`: 프레임별 얼굴크롭 임베딩의 시간평균 → L2 정규화.
- `embed_segments(path, num_frames, num_segments) -> (num_segments, 768)`: 프레임을 시간 구간으로
  묶어 각 구간 평균 → 각 행 L2 정규화. (V-JEPA 토큰 reshape보다 단순·정확 → 구간 매칭 품질↑)

## 에러 처리

- 영상 못 열기/프레임 없음: `load_frames`가 `ValueError` (기존 동작).
- 프레임에 얼굴 없음: 그 프레임 스킵.
- 클립 전체 얼굴 0개: 모든 프레임 중앙 정사각 크롭으로 폴백(경고). 결과는 여전히 단위벡터.
- `build_index`는 이미 per-clip try/except로 깨진 클립 스킵.

## 테스트 (TDD)

1. `face_utils`: 얼굴 없는 검은 프레임 → None / 폴백 동작. 합성 얼굴 박스 크롭 형태.
2. `embedding`: `embed` 단위벡터·1D, `embed_segments` shape (S,768)·행별 정규화 (gpu 마크).
3. `combine_embeddings` 수학 테스트는 차원 무관 → 유지.
4. 스모크: 샘플 클립 1개로 검색 1회 동작.

## 영향 범위

- 신규: `src/video_feedback/face_utils.py`, `src/video_feedback/embedding_vjepa.py`(보존).
- 수정: `embedding.py`(재구현), `tests/test_embedding.py`(1024→768), build_index/app/query 주석.
- 비가역: `index.npz` 재구축 → `index_vjepa.npz` 백업으로 가역성 확보.
- 의존성 추가: `facenet-pytorch==2.5.3` (설치·검증 완료).

## 위험 / 가역성

- 위험: 中. 인덱스 재구축이 핵심 비가역 → 백업으로 커버. 코드는 보존 파일로 가역.
- 표정 검출 실패 클립(얼굴 안 보임/풀샷)은 폴백으로 품질 저하 가능 → 정성 검증 TODO.

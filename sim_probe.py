"""검색 변별력 측정 (SOMA-429). 라벨 없이 재는 세 가지 — 유료 호출 0.

  ① 영상 간 유사도 행렬(표정+소리 결합 0.5/0.5) — 다 0.9면 변별력 없음
  ② 모달별 분리: 표정만 / 소리만 행렬
  ③ 구간 응집도: 같은 영상 구간끼리 vs 다른 영상 구간 — 앞이 높아야 "장면"을 구분하는 것
사용: uv run python sim_probe.py <영상 폴더>
"""
import glob, json, sys
import numpy as np
from video_feedback.embedding import VideoEmbedder, combine_weighted
from video_feedback.audio_embedding import AudioEmbedder

videos = sorted(glob.glob(sys.argv[1].rstrip("/") + "/*.mp4"))
ve, va = VideoEmbedder(), AudioEmbedder()

names, V, A, C, SEG = [], [], [], [], []
for p in videos:
    names.append(p.split("/")[-1][:8])
    v = np.asarray(ve.embed(p), dtype="float32").ravel()
    a = np.asarray(va.embed(p), dtype="float32").ravel()
    v /= (np.linalg.norm(v) + 1e-9); a /= (np.linalg.norm(a) + 1e-9)
    V.append(v); A.append(a)
    C.append(combine_weighted([v, a], [0.5, 0.5]))
    SEG.append(np.asarray(ve.embed_segments(p), dtype="float32"))

def matrix(X):
    X = np.stack(X)
    return np.round(X @ X.T, 2).tolist()

# 구간 응집도
intra, inter = [], []
for i in range(len(SEG)):
    si = SEG[i]
    m = si @ si.T
    intra += [m[r, c] for r in range(len(si)) for c in range(r + 1, len(si))]
    for j in range(i + 1, len(SEG)):
        inter += (si @ SEG[j].T).ravel().tolist()

print(json.dumps({
    "videos": names,
    "결합_행렬": matrix(C),
    "표정_행렬": matrix(V),
    "소리_행렬": matrix(A),
    "구간_같은영상_평균": round(float(np.mean(intra)), 3),
    "구간_다른영상_평균": round(float(np.mean(inter)), 3),
    "구간_격차": round(float(np.mean(intra) - np.mean(inter)), 3),
}, ensure_ascii=False, indent=1))

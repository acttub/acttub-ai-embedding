"""RAG 서빙 산정용 벤치 (SOMA-429). 유료 호출 0 — 전부 로컬 모델.

영상마다: 표정 임베딩(HSEmotion) · 소리 임베딩(CLAP) · 검색 시간을 따로 잰다.
사용: uv run python bench_query.py <영상 폴더>
"""
import glob, json, sys, time, platform

t0 = time.time()
from video_feedback.embedding import VideoEmbedder
from video_feedback.audio_embedding import AudioEmbedder
import numpy as np
load_import = time.time() - t0

t0 = time.time()
ve = VideoEmbedder()
va = AudioEmbedder()
load_models = time.time() - t0

videos = sorted(glob.glob(sys.argv[1].rstrip("/") + "/*.mp4"))[:3]
ref = np.random.rand(2000, 20).astype("float32")  # 2천 클립 인덱스 흉내
rows = []
for v in videos:
    t0 = time.time(); emb_v = ve.embed(v); tv = time.time() - t0
    t0 = time.time(); emb_a = va.embed(v); ta = time.time() - t0
    t0 = time.time()
    q = np.asarray(emb_v, dtype="float32").ravel()[:20]
    sims = ref @ q / (np.linalg.norm(ref, axis=1) * (np.linalg.norm(q) + 1e-9))
    top = np.argsort(-sims)[:5]
    ts = time.time() - t0
    rows.append({"video": v.split("/")[-1], "표정_s": round(tv, 1), "소리_s": round(ta, 1),
                 "검색2천클립_s": round(ts, 4), "합계_s": round(tv + ta + ts, 1)})
    print(rows[-1])

print(json.dumps({
    "machine": platform.machine(), "python": platform.python_version(),
    "import_s": round(load_import, 1), "model_load_s": round(load_models, 1),
    "rows": rows,
}, ensure_ascii=False))

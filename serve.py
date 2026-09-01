"""유사 영상 검색 HTTP 서빙 (SOMA-429).

POST /search  : 배우 영상 → top-5 {제목·배우·유튜브 링크(닮은 구간 t=초)·유사도}
                2단계 검색: 전체 임베딩으로 top-K → 구간(4분할)끼리 재검색으로 장면 지목
POST /reindex : 시트(RAG_SHEET_URL)를 다시 읽어 인덱스 갱신 후 리로드
GET  /health  : 상태·클립 수

실행: RAG_SHEET_URL=<시트URL> uv run uvicorn serve:app --host 0.0.0.0 --port 8080
유료 호출 0 — 전부 로컬 모델. 결과 영상은 유튜브가 재생하므로 원본을 재배포하지 않는다.
"""
import json, os, subprocess, sys, tempfile, threading, time

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from video_feedback.embedding import VideoEmbedder, combine_weighted
from video_feedback.audio_embedding import AudioEmbedder
from video_feedback.explain import match_segments

INDEX_PATH = os.environ.get("RAG_INDEX", "rag_index.npz")
W_VIDEO = float(os.environ.get("RAG_W_VIDEO", "0.5"))
W_AUDIO = float(os.environ.get("RAG_W_AUDIO", "0.5"))
TOKEN = os.environ.get("RAG_TOKEN", "")

app = FastAPI(title="acttub 유사 영상 검색")
_lock = threading.Lock()
_state: dict = {"index": None}
print("모델 로딩 중...", flush=True)
_ve, _va = VideoEmbedder(), AudioEmbedder()
print("모델 로딩 완료", flush=True)


def load_index() -> dict | None:
    if not os.path.exists(INDEX_PATH):
        return None
    z = np.load(INDEX_PATH, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    combined = np.stack([combine_weighted([v, a], [W_VIDEO, W_AUDIO])
                         for v, a in zip(z["vec_video"], z["vec_audio"])])
    return {"meta": meta, "combined": combined,
            "segs": z["segs"], "seg_bounds": z["seg_bounds"]}


_state["index"] = load_index()


def yt_link(meta: dict, at_seconds: float | None) -> str:
    url = meta["url"]
    if at_seconds is None or not url.startswith("http"):
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={max(0, int(at_seconds))}s"


@app.get("/health")
def health():
    idx = _state["index"]
    return {"ok": True, "clips": 0 if idx is None else len(idx["meta"]),
            "weights": {"video": W_VIDEO, "audio": W_AUDIO}}


@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 5):
    idx = _state["index"]
    if idx is None:
        raise HTTPException(503, "인덱스가 없다 — 먼저 /reindex 를 부르거나 ingest.py 를 돌려라")
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        v = np.asarray(_ve.embed(path), dtype="float32").ravel()
        a = np.asarray(_va.embed(path), dtype="float32").ravel()
        v /= np.linalg.norm(v) + 1e-9; a /= np.linalg.norm(a) + 1e-9
        q = combine_weighted([v, a], [W_VIDEO, W_AUDIO])
        q_segs = np.asarray(_ve.embed_segments(path), dtype="float32")
    finally:
        os.unlink(path)

    sims = idx["combined"] @ q
    order = np.argsort(-sims)[:top_k]
    results = []
    for i in order:
        meta = idx["meta"][int(i)]
        qi, ri, seg_sim = match_segments(q_segs, idx["segs"][int(i)])   # 2단계: 구간 재검색
        seg_start = float(idx["seg_bounds"][int(i)][ri][0])
        results.append({
            "title": meta["title"] or meta["key"], "actor": meta["actor"],
            "youtube_url": yt_link(meta, seg_start),
            "similarity": round(float(sims[int(i)]), 3),
            "matched": {"query_segment": int(qi), "clip_segment": int(ri),
                        "segment_similarity": round(seg_sim, 3)},
        })
    return {"results": results, "elapsed_s": round(time.time() - t0, 1)}


@app.post("/reindex")
def reindex(x_token: str = Header(default="")):
    if TOKEN and x_token != TOKEN:
        raise HTTPException(401, "잘못된 토큰")
    sheet = os.environ.get("RAG_SHEET_URL", "")
    if not sheet:
        raise HTTPException(500, "RAG_SHEET_URL 이 설정돼 있지 않다")
    with _lock:
        proc = subprocess.run([sys.executable, "ingest.py", "--sheet", sheet,
                               "--out", INDEX_PATH], capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(500, f"인덱스 갱신 실패: {proc.stderr[-500:]}")
        _state["index"] = load_index()
    return {"ok": True, "clips": len(_state["index"]["meta"]), "log": proc.stdout[-800:]}

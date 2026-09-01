"""구글 시트의 유튜브 링크들로 검색 인덱스를 만든다 (SOMA-429 서빙).

시트가 정본이다 — 팀이 시트에 줄을 추가하고 /reindex 를 부르면 인덱스가 갱신된다.
시트 헤더: url | start | end | title | actor   (start/end 는 "1:20" 꼴, 비우면 전체)

동작: 시트 CSV 읽기 → yt-dlp 로 구간만 내려받기(캐시) → 임베딩(전체 + 구간 4개)
→ rag_index.npz 저장. 임베딩이 끝나면 영상 파일 없이도 서빙이 돈다 — 사용자에게는
유튜브 링크(닮은 구간 시작 초 t= 포함)만 보여준다.

사용:
    uv run python ingest.py --sheet <시트URL|CSV경로> [--out rag_index.npz] [--cache clips_cache]
"""
import argparse, csv, io, json, os, re, subprocess, sys, urllib.request

import numpy as np

from video_feedback.embedding import VideoEmbedder
from video_feedback.audio_embedding import AudioEmbedder
from video_feedback.video_utils import get_duration


def sheet_csv(sheet: str) -> list[dict]:
    """시트 URL(공개 보기 링크)·CSV URL·로컬 CSV 어느 것이든 행 목록으로."""
    if os.path.exists(sheet):
        text = open(sheet, encoding="utf-8-sig").read()
    else:
        url = sheet
        m = re.search(r"docs.google.com/spreadsheets/d/([\w-]+)", sheet)
        if m and "format=csv" not in sheet:
            gid = re.search(r"[#&?]gid=(\d+)", sheet)
            url = (f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
                   + (f"&gid={gid.group(1)}" if gid else ""))
        text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    for r in rows:
        for k in list(r):
            if k and k.strip() != k:
                r[k.strip()] = r.pop(k)
    return [r for r in rows if (r.get("url") or "").strip()]


def to_seconds(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    parts = [float(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{6,})", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-16:]


def fetch(url: str, start: float | None, end: float | None, cache: str) -> str:
    if os.path.exists(url):                      # 로컬 파일도 허용(테스트용)
        return url
    os.makedirs(cache, exist_ok=True)
    tag = f"{video_id(url)}_{int(start or 0)}_{int(end or -1)}"
    dest = os.path.join(cache, tag + ".mp4")
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return dest
    # 영상+오디오를 반드시 합쳐 받는다 — 단일 mp4 포맷만 고르면 쇼츠에서
    # 오디오 없는 스트림이 잡혀 CLAP 임베딩이 실패한다.
    cmd = ["yt-dlp", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/b",
           "--merge-output-format", "mp4", "--force-keyframes-at-cuts", "-o", dest]
    if start is not None or end is not None:
        s = "" if start is None else int(start)
        e = "inf" if end is None else int(end)
        cmd += ["--download-sections", f"*{s}-{e}"]
    cmd.append(url)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out", default="rag_index.npz")
    ap.add_argument("--cache", default="clips_cache")
    args = ap.parse_args()

    old_meta, old = [], {}
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=False)
        old_meta = json.loads(str(z["meta_json"]))
        old = {m["key"]: i for i, m in enumerate(old_meta)}

    ve, va = VideoEmbedder(), AudioEmbedder()
    metas, vecs_v, vecs_a, segs, seg_bounds = [], [], [], [], []

    def keep(i):
        z = np.load(args.out, allow_pickle=False)
        metas.append(old_meta[i]); vecs_v.append(z["vec_video"][i]); vecs_a.append(z["vec_audio"][i])
        segs.append(z["segs"][i]); seg_bounds.append(z["seg_bounds"][i])

    for row in sheet_csv(args.sheet):
        url = row["url"].strip()
        start, end = to_seconds(row.get("start")), to_seconds(row.get("end"))
        key = f"{video_id(url)}_{int(start or 0)}_{int(end or -1)}"
        if key in old:
            keep(old[key]); print("유지:", key); continue
        try:
            path = fetch(url, start, end, args.cache)
            v = np.asarray(ve.embed(path), dtype="float32").ravel()
            a = np.asarray(va.embed(path), dtype="float32").ravel()
            v /= np.linalg.norm(v) + 1e-9; a /= np.linalg.norm(a) + 1e-9
            sg = np.asarray(ve.embed_segments(path), dtype="float32")
            dur = get_duration(path)
            base = start or 0.0
            bounds = np.asarray([[base + dur * i / len(sg), base + dur * (i + 1) / len(sg)]
                                 for i in range(len(sg))], dtype="float32")
        except Exception as exc:                 # 한 줄 실패가 전체를 막지 않는다
            print(f"건너뜀({key}): {exc}", file=sys.stderr); continue
        metas.append({"key": key, "url": url, "start_s": start, "end_s": end,
                      "title": (row.get("title") or "").strip(),
                      "actor": (row.get("actor") or "").strip()})
        vecs_v.append(v); vecs_a.append(a); segs.append(sg); seg_bounds.append(bounds)
        print("추가:", key, metas[-1]["title"])

    if not metas:
        print("인덱스에 넣을 행이 없다", file=sys.stderr); return 1
    np.savez(args.out, meta_json=json.dumps(metas, ensure_ascii=False),
             vec_video=np.stack(vecs_v), vec_audio=np.stack(vecs_a),
             segs=np.stack(segs), seg_bounds=np.stack(seg_bounds))
    print(f"저장: {args.out} (클립 {len(metas)}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""기존 2-모달 인덱스(영상+음성)에 대본(STT) 모달만 얹어 3-모달 인덱스로 만든다.

영상(FER)·음성(CLAP) 임베딩은 이미 ``index_2mod.npz``에 들어 있고 모델도
그대로이므로 재계산하지 않는다. 새로 필요한 대본(Whisper STT → ko-sroberta)만
클립별로 뽑아 붙인다 — 풀 재구축보다 훨씬 빠르다.

이어하기: ``--out`` 인덱스에 이미 text_vectors가 있으면 그 클립은 건너뛴다.
50개마다 체크포인트 저장하므로 중간에 죽어도 손해가 적다.

사용 예:
    uv run python scripts/add_text_modality.py \
        --base index_2mod.npz --clips-dir 연기영상/clips --out index.npz
"""

import argparse
import os
import sys
import time

import numpy as np

from video_feedback.multimodal import MultiModalReferenceDB
from video_feedback.stt import Transcriber
from video_feedback.text_embedding import TextEmbedder

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def _already_done(out_path: str) -> dict[str, np.ndarray]:
    """이미 텍스트까지 들어간 인덱스에서 {ref_id: text_vec}를 읽는다 (이어하기용)."""
    if not os.path.exists(out_path):
        return {}
    data = np.load(out_path, allow_pickle=True)
    if "text_vectors" not in data.files:
        return {}
    return {str(rid): tv for rid, tv in zip(data["ids"], data["text_vectors"])}


def add_text(base_path: str, clips_dir: str, out_path: str) -> None:
    """base 2-모달 인덱스에 대본 임베딩을 얹어 out으로 저장한다.

    Args:
        base_path: 영상+음성만 든 기존 인덱스(.npz).
        clips_dir: 원본 클립이 있는 디렉터리 (ref_id로 경로를 찾는다).
        out_path: 저장할 3-모달 인덱스(.npz).
    """
    base = MultiModalReferenceDB.load(base_path)
    done = _already_done(out_path)
    if done:
        print(f"이어하기: 이미 {len(done)}개 대본 완료 → 건너뜀")

    print("STT/텍스트 임베더 로딩 중...")
    transcriber = Transcriber()
    text_embedder = TextEmbedder()
    text_dim = text_embedder.model.get_sentence_embedding_dimension()
    print(f"장치: stt={transcriber.device}, text={text_embedder.device}")

    out = MultiModalReferenceDB()
    n = len(base._ids)
    t0 = time.perf_counter()
    for i, ref_id in enumerate(base._ids, 1):
        vvec, avec = base._video[i - 1], base._audio[i - 1]

        if ref_id in done:
            tvec = done[ref_id]
            note = "(이어하기)"
        else:
            path = os.path.join(clips_dir, ref_id)
            try:
                script = transcriber.transcribe(path)
                tvec = text_embedder.embed(script)
            except Exception as exc:  # STT/오디오 문제 → 빈 대본(영벡터)
                script, tvec = "", np.zeros(text_dim, dtype=np.float32)
                print(f"  [{i}/{n}] STT실패 {ref_id}: {exc}", flush=True)
            note = (script[:30].replace("\n", " ")) if script else "(무음)"

        out.add(ref_id, vvec, avec, tvec)
        print(f"  [{i}/{n}] {ref_id}  대본: {note}", flush=True)
        if i % 50 == 0:
            out.save(out_path)
            print(f"  ... 체크포인트 저장 ({len(out._ids)}개)", flush=True)

    out.save(out_path)
    dt = time.perf_counter() - t0
    print(f"\n3-모달 인덱스 저장 완료: {out_path} ({len(out._ids)}개, {dt:.1f}s)")


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="2-모달 인덱스에 대본 모달 추가")
    parser.add_argument("--base", default="index_2mod.npz")
    parser.add_argument("--clips-dir", default="연기영상/clips")
    parser.add_argument("--out", default="index.npz")
    args = parser.parse_args()
    add_text(args.base, args.clips_dir, args.out)


if __name__ == "__main__":
    main()

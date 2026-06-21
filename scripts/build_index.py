"""기준(전문가) 영상들을 임베딩해 검색 인덱스(.npz)로 저장한다.

사용 예:
    uv run python scripts/build_index.py \
        --clips-dir 연기영상/clips --pattern "C-pro__*.mp4" --out index.npz
"""

import argparse
import glob
import os
import time

from video_feedback.audio_embedding import AudioEmbedder
from video_feedback.embedding import VideoEmbedder
from video_feedback.multimodal import MultiModalReferenceDB


def build_index(
    clips_dir: str,
    pattern: str,
    out_path: str,
    num_frames: int = 64,
) -> None:
    """패턴에 맞는 영상들을 영상+음성 임베딩해 멀티모달 인덱스로 저장한다.

    Args:
        clips_dir: 영상이 들어 있는 디렉터리.
        pattern: 파일 glob 패턴 (예: "C-pro__*.mp4").
        out_path: 저장할 .npz 경로.
        num_frames: 영상당 샘플링 프레임 수.
    """
    paths = sorted(glob.glob(os.path.join(clips_dir, pattern)))
    if not paths:
        raise SystemExit(f"매칭되는 영상이 없습니다: {clips_dir}/{pattern}")

    print(f"대상 영상 {len(paths)}개 발견. 영상/음성 임베더 로딩 중...")
    video_embedder = VideoEmbedder()
    audio_embedder = AudioEmbedder()
    print(f"장치: video={video_embedder.device}, audio={audio_embedder.device}")

    db = MultiModalReferenceDB()
    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        ref_id = os.path.basename(path)
        try:
            vvec = video_embedder.embed(path, num_frames=num_frames)
            avec = audio_embedder.embed(path)
        except Exception as exc:  # 깨진 영상/오디오는 건너뛴다
            print(f"  [{i}/{len(paths)}] SKIP {ref_id}: {exc}")
            continue
        db.add(ref_id, vvec, avec)
        print(f"  [{i}/{len(paths)}] OK {ref_id}")

    db.save(out_path)
    dt = time.perf_counter() - t0
    print(f"\n인덱스 저장 완료: {out_path} ({len(db._ids)}개, {dt:.1f}s)")


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="기준 영상 임베딩 인덱스 구축")
    parser.add_argument("--clips-dir", default="연기영상/clips")
    parser.add_argument("--pattern", default="C-pro__*.mp4")
    parser.add_argument("--out", default="index.npz")
    parser.add_argument("--num-frames", type=int, default=64)
    args = parser.parse_args()
    build_index(args.clips_dir, args.pattern, args.out, args.num_frames)


if __name__ == "__main__":
    main()

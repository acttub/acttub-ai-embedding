"""질의 영상과 가장 유사한 기준 영상 top-K를 인덱스에서 찾는다.

사용 예:
    uv run python scripts/query.py 연기영상/clips/A-amateur__xxx.mp4 \
        --index index.npz --k 5
"""

import argparse

from video_feedback.audio_embedding import AudioEmbedder
from video_feedback.embedding import VideoEmbedder
from video_feedback.multimodal import MultiModalReferenceDB


def query(
    video_path: str,
    index_path: str,
    k: int = 5,
    num_frames: int = 64,
    w_audio: float = 0.5,
) -> list[tuple[str, float]]:
    """질의 영상을 영상+음성 임베딩해 인덱스에서 top-K 유사 영상을 반환한다.

    Args:
        video_path: 질의 영상 경로.
        index_path: build_index.py로 만든 .npz 멀티모달 인덱스 경로.
        k: 반환할 유사 영상 수.
        num_frames: 샘플링 프레임 수.
        w_audio: 음성 가중치 [0, 1]. 0이면 영상만, 1이면 음성만 본다.

    Returns:
        (ref_id, 결합 코사인 유사도) 리스트, 유사도 내림차순.
    """
    db = MultiModalReferenceDB.load(index_path)
    vvec = VideoEmbedder().embed(video_path, num_frames=num_frames)
    avec = AudioEmbedder().embed(video_path)
    return db.search(vvec, avec, w_audio=w_audio, k=k)


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="유사 기준 영상 검색")
    parser.add_argument("video", help="질의 영상 경로")
    parser.add_argument("--index", default="index.npz")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument(
        "--w-audio",
        type=float,
        default=0.5,
        help="음성 가중치 [0,1]. 0=영상만, 1=음성만 (기본 0.5)",
    )
    args = parser.parse_args()

    results = query(args.video, args.index, args.k, args.num_frames, args.w_audio)
    print(f"\n질의: {args.video} (w_audio={args.w_audio})")
    print(f"top-{args.k} 유사 기준 영상:")
    for rank, (ref_id, score) in enumerate(results, 1):
        print(f"  {rank}. {ref_id}  (유사도 {score:.4f})")


if __name__ == "__main__":
    main()

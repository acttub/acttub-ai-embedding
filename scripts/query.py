"""질의 영상과 가장 유사한 기준 영상 top-K를 인덱스에서 찾는다.

사용 예:
    uv run python scripts/query.py 연기영상/clips/A-amateur__xxx.mp4 \
        --index index.npz --k 5 --w-audio 0.4 --w-text 0.2
"""

import argparse

from video_feedback.audio_embedding import AudioEmbedder
from video_feedback.embedding import VideoEmbedder
from video_feedback.multimodal import MultiModalReferenceDB
from video_feedback.stt import Transcriber
from video_feedback.text_embedding import TextEmbedder


def query(
    video_path: str,
    index_path: str,
    k: int = 5,
    num_frames: int = 64,
    w_audio: float = 0.4,
    w_text: float = 0.2,
) -> list[tuple[str, float]]:
    """질의 영상을 영상+음성+대본 임베딩해 인덱스에서 top-K 유사 영상을 반환한다.

    Args:
        video_path: 질의 영상 경로.
        index_path: build_index.py로 만든 .npz 멀티모달 인덱스 경로.
        k: 반환할 유사 영상 수.
        num_frames: 샘플링 프레임 수.
        w_audio: 음성 가중치 [0, 1].
        w_text: 대본 가중치 [0, 1]. (영상 가중치 = 1 - w_audio - w_text)

    Returns:
        (ref_id, 결합 코사인 유사도) 리스트, 유사도 내림차순.
    """
    db = MultiModalReferenceDB.load(index_path)
    vvec = VideoEmbedder().embed(video_path, num_frames=num_frames)
    avec = AudioEmbedder().embed(video_path)
    script = Transcriber().transcribe(video_path)
    tvec = TextEmbedder().embed(script)
    return db.search(vvec, avec, tvec, w_audio=w_audio, w_text=w_text, k=k)


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
        default=0.4,
        help="음성 가중치 [0,1] (기본 0.4)",
    )
    parser.add_argument(
        "--w-text",
        type=float,
        default=0.2,
        help="대본 가중치 [0,1] (기본 0.2). 영상=1-w_audio-w_text",
    )
    args = parser.parse_args()

    results = query(
        args.video, args.index, args.k, args.num_frames, args.w_audio, args.w_text
    )
    w_video = 1.0 - args.w_audio - args.w_text
    print(
        f"\n질의: {args.video} "
        f"(영상={w_video:.2f}, 음성={args.w_audio}, 대본={args.w_text})"
    )
    print(f"top-{args.k} 유사 기준 영상:")
    for rank, (ref_id, score) in enumerate(results, 1):
        print(f"  {rank}. {ref_id}  (유사도 {score:.4f})")


if __name__ == "__main__":
    main()

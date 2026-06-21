"""유사 영상 검색 웹 UI (Gradio).

영상을 업로드하면 VJEPA2 임베딩으로 기준(전문가) 영상 인덱스에서
가장 유사한 top-K를 찾아 보여준다.

실행:
    uv run python app.py
브라우저에서 http://127.0.0.1:7860 접속.
"""

import glob
import os

import gradio as gr

from video_feedback.audio_embedding import AudioEmbedder
from video_feedback.embedding import VideoEmbedder
from video_feedback.multimodal import MultiModalReferenceDB

INDEX_PATH = "index.npz"
CLIPS_DIR = os.path.join("연기영상", "clips")

# 서버 기동 시 1회 로드 (GPU 상주)
print("인덱스/모델 로딩 중...")
_db = MultiModalReferenceDB.load(INDEX_PATH)
_video_embedder = VideoEmbedder()
_audio_embedder = AudioEmbedder()
# ref_id(파일명) -> 재생용 전체 경로
_ref_paths = {
    os.path.basename(p): p for p in glob.glob(os.path.join(CLIPS_DIR, "*.mp4"))
}
print(f"로딩 완료. 장치={_video_embedder.device}, 기준 영상={len(_ref_paths)}개")


def search_similar(video_path: str | None, k: int, w_audio: float):
    """업로드 영상과 가장 유사한 top-K 기준 영상을 찾는다 (영상+음성).

    Args:
        video_path: 업로드된 질의 영상 경로 (Gradio가 임시파일로 전달).
        k: 반환할 유사 영상 수.
        w_audio: 음성 가중치 [0, 1]. 0이면 영상만, 1이면 음성만 본다.

    Returns:
        (top-1 영상 경로, 순위 테이블 행 리스트). 입력이 없으면 (None, []).
    """
    if not video_path:
        return None, []
    vvec = _video_embedder.embed(video_path)
    avec = _audio_embedder.embed(video_path)
    results = _db.search(vvec, avec, w_audio=float(w_audio), k=int(k))
    rows = [
        [rank, ref_id, round(score, 4)]
        for rank, (ref_id, score) in enumerate(results, 1)
    ]
    best_path = _ref_paths.get(results[0][0]) if results else None
    return best_path, rows


with gr.Blocks(title="유사 연기영상 검색") as demo:
    gr.Markdown(
        "# 🎬 유사 연기영상 검색\n질의 영상을 올리면 가장 닮은 전문가 영상을 찾아줍니다."
    )
    with gr.Row():
        with gr.Column():
            query_video = gr.Video(label="질의 영상 업로드")
            k_slider = gr.Slider(1, 10, value=5, step=1, label="top-K")
            w_audio_slider = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.1,
                label="음성 가중치 (0=영상만 · 1=음성만)",
            )
            search_btn = gr.Button("유사 영상 검색", variant="primary")
        with gr.Column():
            best_video = gr.Video(label="가장 유사한 전문가 영상", interactive=False)
            result_table = gr.Dataframe(
                headers=["순위", "영상 파일", "유사도"],
                label="top-K 결과",
                interactive=False,
            )

    search_btn.click(
        fn=search_similar,
        inputs=[query_video, k_slider, w_audio_slider],
        outputs=[best_video, result_table],
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=[os.path.abspath(CLIPS_DIR)])

"""유사 영상 검색 웹 UI (Gradio).

영상을 업로드하면 얼굴 표정(FER) 임베딩으로 기준(전문가) 영상 인덱스에서
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
from video_feedback.explain import explain_match
from video_feedback.gemini_feedback import (
    generate_feedback,
    load_env,
    render_card_markdown,
)
from video_feedback.multimodal import MultiModalReferenceDB
from video_feedback.stt import Transcriber
from video_feedback.text_embedding import TextEmbedder

load_env()  # .env에서 GEMINI_API_KEY 등을 환경변수로 로드

# 3-모달 전용: 대본(text) 임베딩이 들어 있는 인덱스만 사용한다.
INDEX_PATH = os.environ.get("INDEX_PATH", "index_3mod.npz")
CLIPS_DIR = os.path.join("연기영상", "clips")

# 서버 기동 시 1회 로드 (GPU 상주)
print("인덱스/모델 로딩 중...")
_db = MultiModalReferenceDB.load(INDEX_PATH)
if not _db._has_text():
    raise SystemExit(
        f"3-모달 인덱스가 아닙니다(대본 임베딩 없음): {INDEX_PATH}\n"
        "scripts/add_text_modality.py 로 대본 모달을 채운 인덱스를 지정하세요."
    )
_video_embedder = VideoEmbedder()
_audio_embedder = AudioEmbedder()
_transcriber = Transcriber()
_text_embedder = TextEmbedder()
# ref_id(파일명) -> 재생용 전체 경로
_ref_paths = {
    os.path.basename(p): p for p in glob.glob(os.path.join(CLIPS_DIR, "*.mp4"))
}
print(f"로딩 완료(3-모달). 장치={_video_embedder.device}, 기준 영상={len(_db._ids)}개")


def _normalize3(w_video: float, w_audio: float, w_text: float) -> tuple[float, float]:
    """영상/음성/대본 가중치를 합=1로 정규화해 (w_audio, w_text)를 돌려준다.

    search()는 영상 가중치를 ``1 - w_audio - w_text``로 자동 계산하므로
    음성·대본 두 값만 넘기면 된다. 셋 다 0이면 1/3씩 균등 배분한다.
    """
    total = float(w_video) + float(w_audio) + float(w_text)
    if total <= 0:
        return 1 / 3, 1 / 3
    return float(w_audio) / total, float(w_text) / total


def search_similar(
    video_path: str | None,
    k: int,
    w_video: float,
    w_audio: float,
    w_text: float,
):
    """업로드 영상과 가장 유사한 top-K 기준 영상을 찾는다 (영상+음성+대본 3-모달).

    영상/음성/대본 가중치는 합=1로 자동 정규화돼 "상대 비율"로 동작한다.

    Args:
        video_path: 업로드된 질의 영상 경로 (Gradio가 임시파일로 전달).
        k: 반환할 유사 영상 수.
        w_video: 영상(표정) 비율.
        w_audio: 음성(파형) 비율.
        w_text: 대본(STT) 비율.

    Returns:
        (top-1 영상 경로, 순위 테이블, 구간설명, 질의 대본). 입력 없으면 빈 값.
    """
    if not video_path:
        return None, [], "", ""
    wa, wt = _normalize3(w_video, w_audio, w_text)
    vvec = _video_embedder.embed(video_path)
    avec = _audio_embedder.embed(video_path)
    script = _transcriber.transcribe(video_path)
    tvec = _text_embedder.embed(script)
    results = _db.search(vvec, avec, tvec, w_audio=wa, w_text=wt, k=int(k))
    rows = [
        [rank, ref_id, round(score, 4)]
        for rank, (ref_id, score) in enumerate(results, 1)
    ]
    best_path = _ref_paths.get(results[0][0]) if results else None
    explanation = ""
    if best_path:
        # top-1 영상과 질의를 구간 매칭해 "어느 장면이 닮았는지" 설명
        explanation = explain_match(_video_embedder, video_path, best_path)
    transcript = script or "(대사 없음/무음)"
    return best_path, rows, explanation, transcript


def compare_feedback(
    video_path: str | None,
    w_video: float,
    w_audio: float,
    w_text: float,
    media: str,
    situation: str,
    character: str,
    subtext: str,
):
    """입력 영상으로 (A) 단독 vs (B) 유사 C-pro 동봉 피드백 카드를 비교한다.

    먼저 멀티모달 검색으로 가장 닮은 C-pro 영상을 찾고, 그 영상을 "전문가
    기준"으로 함께 준 경우와 안 준 경우의 피드백을 각각 생성한다.

    Args:
        video_path: 배우(질의) 영상 경로.
        w_audio: 유사 영상 검색 시 음성 가중치.
        media/situation/character/subtext: 배우가 준 설정.

    Returns:
        (어떤 C-pro를 골랐는지 안내, A 카드 md, B 카드 md).
    """
    if not video_path:
        return "영상을 먼저 올려주세요.", "", ""
    wa, wt = _normalize3(w_video, w_audio, w_text)
    vvec = _video_embedder.embed(video_path)
    avec = _audio_embedder.embed(video_path)
    tvec = _text_embedder.embed(_transcriber.transcribe(video_path))
    results = _db.search(vvec, avec, tvec, w_audio=wa, w_text=wt, k=1)
    if not results:
        return "유사 C-pro 영상을 찾지 못했습니다.", "", ""
    ref_id, ref_score = results[0]
    ref_path = _ref_paths.get(ref_id)

    settings = dict(
        media=media, situation=situation, character=character, subtext=subtext
    )
    try:
        # (A) 입력 영상만 / (B) 입력 + 유사 C-pro(전문가 기준)
        card_a = generate_feedback(video_path, None, **settings)
        card_b = generate_feedback(video_path, ref_path, **settings)
    except Exception as e:  # noqa: BLE001 — UI에 그대로 노출
        return f"피드백 생성 실패: {e}", "", ""

    info = f"비교에 쓴 유사 C-pro 영상: **{ref_id}** (유사도 {ref_score:.3f})"
    return info, render_card_markdown(card_a), render_card_markdown(card_b)


with gr.Blocks(title="유사 연기영상 검색") as demo:
    gr.Markdown(
        "# 🎬 유사 연기영상 검색\n질의 영상을 올리면 가장 닮은 전문가 영상을 찾아줍니다."
    )
    with gr.Row():
        with gr.Column():
            query_video = gr.Video(label="질의 영상 업로드")
            k_slider = gr.Slider(1, 10, value=5, step=1, label="top-K")
            gr.Markdown("**모달 비율** (세 값은 합=1로 자동 정규화)")
            w_video_slider = gr.Slider(
                0.0, 1.0, value=0.4, step=0.05, label="영상(표정 FER) 비율"
            )
            w_audio_slider = gr.Slider(
                0.0, 1.0, value=0.4, step=0.05, label="음성(파형 CLAP) 비율"
            )
            w_text_slider = gr.Slider(
                0.0, 1.0, value=0.2, step=0.05, label="대본(STT) 비율"
            )
            search_btn = gr.Button("유사 영상 검색", variant="primary")
        with gr.Column():
            best_video = gr.Video(label="가장 유사한 전문가 영상", interactive=False)
            transcript_box = gr.Textbox(
                label="질의 영상 대본 (STT)",
                interactive=False,
            )
            explanation_box = gr.Textbox(
                label="왜 이 영상이 닮았나요? (구간 매칭)",
                interactive=False,
            )
            result_table = gr.Dataframe(
                headers=["순위", "영상 파일", "유사도"],
                label="top-K 결과",
                interactive=False,
            )

    search_btn.click(
        fn=search_similar,
        inputs=[
            query_video,
            k_slider,
            w_video_slider,
            w_audio_slider,
            w_text_slider,
        ],
        outputs=[best_video, result_table, explanation_box, transcript_box],
    )

    gr.Markdown("---\n## 🤖 AI 코치 피드백 비교 (A: 입력만 · B: 유사 C-pro 동봉)")
    gr.Markdown(
        "위에 올린 영상으로 피드백 카드를 두 가지로 생성합니다 — "
        "**A) 입력 영상만** 줬을 때 vs **B) 가장 닮은 전문가(C-pro) 영상을 "
        "기준으로 함께** 줬을 때. 설정은 비워도 됩니다(영상만으로 추론)."
    )
    with gr.Row():
        media_in = gr.Textbox(label="매체/장르", placeholder="예: 영화 / 연극 / 드라마")
        situation_in = gr.Textbox(label="상황", placeholder="예: 이별 통보를 받는 장면")
    with gr.Row():
        character_in = gr.Textbox(label="인물 설정", placeholder="예: 20대 직장인")
        subtext_in = gr.Textbox(
            label="서브텍스트(속마음)", placeholder="예: 붙잡고 싶지만 참는다"
        )
    feedback_btn = gr.Button("A/B 피드백 비교 생성", variant="primary")
    ref_info = gr.Markdown()
    with gr.Row():
        card_a_md = gr.Markdown(label="A: 입력 영상만")
        card_b_md = gr.Markdown(label="B: 유사 C-pro 동봉")

    feedback_btn.click(
        fn=compare_feedback,
        inputs=[
            query_video,
            w_video_slider,
            w_audio_slider,
            w_text_slider,
            media_in,
            situation_in,
            character_in,
            subtext_in,
        ],
        outputs=[ref_info, card_a_md, card_b_md],
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=[os.path.abspath(CLIPS_DIR)])

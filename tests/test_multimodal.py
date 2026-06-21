import numpy as np

from video_feedback.multimodal import MultiModalReferenceDB


def _v(*xs):
    return np.array(xs, dtype=np.float32)


def test_search_audio_weight_selects_modality():
    db = MultiModalReferenceDB()
    # a: 영상 닮음 / 음성 다름,  b: 영상 다름 / 음성 닮음
    db.add("a", video_vec=_v(1, 0), audio_vec=_v(0, 1))
    db.add("b", video_vec=_v(0, 1), audio_vec=_v(1, 0))
    q_video, q_audio = _v(1, 0), _v(1, 0)

    # w_audio=0 → 영상만 본다 → a
    assert db.search(q_video, q_audio, w_audio=0.0, k=1)[0][0] == "a"
    # w_audio=1 → 음성만 본다 → b
    assert db.search(q_video, q_audio, w_audio=1.0, k=1)[0][0] == "b"


def test_search_score_decomposes_by_weight():
    db = MultiModalReferenceDB()
    rs = np.random.RandomState(0)
    rv = rs.randn(8).astype(np.float32)
    rv /= np.linalg.norm(rv)
    ra = rs.randn(4).astype(np.float32)
    ra /= np.linalg.norm(ra)
    db.add("x", video_vec=rv, audio_vec=ra)

    qv = rs.randn(8).astype(np.float32)
    qv /= np.linalg.norm(qv)
    qa = rs.randn(4).astype(np.float32)
    qa /= np.linalg.norm(qa)

    w = 0.3
    score = db.search(qv, qa, w_audio=w, k=1)[0][1]
    expected = (1 - w) * float(rv @ qv) + w * float(ra @ qa)
    assert np.isclose(score, expected, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    db = MultiModalReferenceDB()
    db.add("a", video_vec=_v(1, 0), audio_vec=_v(0, 1))
    db.add("b", video_vec=_v(0, 1), audio_vec=_v(1, 0))
    p = tmp_path / "idx.npz"
    db.save(str(p))

    loaded = MultiModalReferenceDB.load(str(p))
    res = loaded.search(_v(1, 0), _v(1, 0), w_audio=0.0, k=1)
    assert res[0][0] == "a"

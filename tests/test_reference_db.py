import numpy as np

from video_feedback.reference_db import ReferenceDB


def test_match_returns_closest():
    db = ReferenceDB()
    db.add("a", np.array([1.0, 0.0], dtype=np.float32))
    db.add("b", np.array([0.0, 1.0], dtype=np.float32))
    ref_id, score = db.match(np.array([0.9, 0.1], dtype=np.float32))
    assert ref_id == "a"
    assert score > 0.9


def test_search_returns_topk_sorted():
    db = ReferenceDB()
    db.add("a", np.array([1.0, 0.0], dtype=np.float32))
    db.add("b", np.array([0.0, 1.0], dtype=np.float32))
    db.add("c", np.array([0.7, 0.7], dtype=np.float32))
    results = db.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert len(results) == 2
    assert results[0][0] == "a"  # 가장 유사
    assert results[0][1] >= results[1][1]  # 내림차순 정렬
    assert results[1][0] == "c"  # 그 다음 유사


def test_search_k_larger_than_db():
    db = ReferenceDB()
    db.add("a", np.array([1.0, 0.0], dtype=np.float32))
    results = db.search(np.array([1.0, 0.0], dtype=np.float32), k=10)
    assert len(results) == 1  # DB 크기로 클램프


def test_save_load_roundtrip(tmp_path):
    db = ReferenceDB()
    db.add("a", np.array([1.0, 0.0], dtype=np.float32))
    p = tmp_path / "db.npz"
    db.save(str(p))
    loaded = ReferenceDB.load(str(p))
    ref_id, _ = loaded.match(np.array([1.0, 0.0], dtype=np.float32))
    assert ref_id == "a"

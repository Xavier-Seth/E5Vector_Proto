import numpy as np
from app import _decide_label, SIMILARITY_THRESHOLD

def test_decide_label_above_threshold():
    labels = ["A", "B", "C"]
    sims = np.array([0.12, SIMILARITY_THRESHOLD + 0.01, 0.50])

    label, score = _decide_label("dummy text", sims, labels)

    assert label == "B"
    assert score >= SIMILARITY_THRESHOLD

def test_decide_label_below_threshold():
    labels = ["A", "B", "C"]
    sims = np.array([0.10, 0.20, 0.30])

    label, score = _decide_label("dummy text", sims, labels)

    assert label == "Uncategorized"
    assert score < SIMILARITY_THRESHOLD

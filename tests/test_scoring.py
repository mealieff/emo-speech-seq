import pytest

from emoseq.scoring.report import quartile_report
from emoseq.scoring.sequence import clip_spans, score_sequence

GOLD = [
    {"start": 0.0, "end": 3.2, "emotion": "Neutral"},
    {"start": 3.2, "end": 7.5, "emotion": "Happy"},
]


def test_perfect_prediction():
    s = score_sequence(GOLD, GOLD)
    assert s["parsed"] == 1.0
    assert s["micro_f1"] == 1.0
    assert s["mean_tiou"] == pytest.approx(1.0)
    assert s["mean_tiou_p"] == pytest.approx(1.0)
    assert s["boundary_error_sec"] == 0.0


def test_empty_prediction_is_unparsed():
    s = score_sequence(GOLD, [])
    assert s["parsed"] == 0.0
    assert s["micro_f1"] == 0.0
    assert s["mean_tiou"] == 0.0


def test_precision_side_tiou_clips_past_end():
    pred = GOLD + [{"start": 9.0, "end": 12.0, "emotion": "Sad"}]
    assert clip_spans(pred, 7.5)[-1]["start"] == clip_spans(pred, 7.5)[-1]["end"] == 7.5
    s = score_sequence(GOLD, pred, duration=7.5)
    assert s["mean_tiou"] == pytest.approx(1.0)
    assert s["mean_tiou_p"] < 1.0


def test_quartile_report_bins():
    rows = [
        {"seq_id": "a", "seq_len": 2, "length_bin": "2", "audio_duration": 7.5, "ground_truth": GOLD, "pred": GOLD},
        {"seq_id": "b", "seq_len": 2, "length_bin": "2", "audio_duration": 7.5, "ground_truth": GOLD, "pred": []},
    ]
    rep = quartile_report(rows)
    assert rep["by_quartile"]["2"]["n"] == 2
    assert rep["by_quartile"]["2"]["parse_rate"] == pytest.approx(0.5)

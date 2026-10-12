import numpy as np

from emoseq.windows import split_audio, stitch_windows

SR = 16000


def test_split_audio_non_overlapping_30s():
    wins = split_audio(np.zeros(int(65 * SR)), SR)
    assert [off for off, _ in wins] == [0.0, 30.0, 60.0]
    assert [len(w) / SR for _, w in wins] == [30.0, 30.0, 5.0]
    assert len(split_audio(np.zeros(10 * SR), SR)) == 1


def test_merge_only_at_seam_with_same_label():
    w0 = [{"start": 0.0, "end": 20.0, "emotion": "Neutral"}, {"start": 20.0, "end": 30.0, "emotion": "Happy"}]
    w1 = [{"start": 0.0, "end": 5.0, "emotion": "Happy"}, {"start": 5.0, "end": 12.0, "emotion": "Sad"}]
    out = stitch_windows([(0.0, w0), (30.0, w1)])
    assert out == [
        {"start": 0.0, "end": 20.0, "emotion": "Neutral"},
        {"start": 20.0, "end": 35.0, "emotion": "Happy"},
        {"start": 35.0, "end": 42.0, "emotion": "Sad"},
    ]


def test_no_merge_when_label_differs_or_gap():
    w0 = [{"start": 0.0, "end": 30.0, "emotion": "Neutral"}]
    w1 = [{"start": 0.0, "end": 5.0, "emotion": "Happy"}]
    assert len(stitch_windows([(0.0, w0), (30.0, w1)])) == 2
    w0 = [{"start": 0.0, "end": 28.0, "emotion": "Neutral"}]
    w1 = [{"start": 0.0, "end": 5.0, "emotion": "Neutral"}]
    assert len(stitch_windows([(0.0, w0), (30.0, w1)])) == 2


def test_same_label_spans_within_a_window_stay_separate():
    w0 = [{"start": 0.0, "end": 3.0, "emotion": "Neutral"}, {"start": 3.0, "end": 6.0, "emotion": "Neutral"}]
    assert len(stitch_windows([(0.0, w0)])) == 2

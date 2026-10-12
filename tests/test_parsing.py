from emoseq.parsing.sequence import parse_prediction
from emoseq.scoring.utterance import parse_task_a, parse_task_a_first_n


def test_seconds_lines():
    text = "(0.00, 3.20, Neutral)\n(3.20, 7.5, Happy)"
    assert parse_prediction(text) == [
        {"start": 0.0, "end": 3.2, "emotion": "Neutral"},
        {"start": 3.2, "end": 7.5, "emotion": "Happy"},
    ]


def test_clock_timestamps_only_in_default_grammar():
    text = "(0:00, 0:03.2, Neutral)"
    assert parse_prediction(text) == [{"start": 0.0, "end": 3.2, "emotion": "Neutral"}]
    assert parse_prediction(text, clock=False) == []


def test_unparseable_text():
    assert parse_prediction("The speaker sounds calm.") == []


def test_utterance_first_n_vs_strict():
    assert parse_task_a("Happy") == 2
    assert parse_task_a("NeutralNeutral") is None
    assert parse_task_a_first_n("NeutralNeutral") == 7

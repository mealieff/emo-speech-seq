import importlib
import os


def test_env_overrides_and_relative_paths(monkeypatch, tmp_path):
    env_file = tmp_path / "paths.env"
    env_file.write_text("MSP_ROOT=/nonexistent/msp\nEMOSEQ_DATA=somewhere\n")
    monkeypatch.setenv("EMOSEQ_ENV_FILE", str(env_file))
    monkeypatch.delenv("MSP_ROOT", raising=False)
    monkeypatch.delenv("EMOSEQ_DATA", raising=False)
    monkeypatch.setenv("EMOSEQ_RUNS", str(tmp_path / "runs"))
    import emoseq.config as config

    config = importlib.reload(config)
    try:
        assert str(config.MSP_AUDIO) == "/nonexistent/msp/Audios"
        assert config.DATA_DIR == (config.REPO / "somewhere").resolve()
        assert config.RUNS_DIR == tmp_path / "runs"
        assert config.sequence_jsonl("test1") == config.DATA_DIR / "sequences" / "test1.jsonl"
    finally:
        for k in ("MSP_ROOT", "EMOSEQ_DATA"):
            os.environ.pop(k, None)
        monkeypatch.delenv("EMOSEQ_ENV_FILE")
        importlib.reload(config)

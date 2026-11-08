from pathlib import Path
import subprocess

import pytest

import assistant.stt as stt


def test_transcribe_audio_converts_and_runs_local_whisper(monkeypatch, tmp_path):
    model = tmp_path / "model.bin"
    binary = tmp_path / "whisper-cli"
    model.touch()
    binary.touch()
    monkeypatch.setattr(stt, "MODEL_PATH", model)
    monkeypatch.setattr(stt, "WHISPER_BIN", str(binary))
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == stt.FFMPEG_BIN:
            Path(args[-1]).write_bytes(b"wav")
        else:
            output_base = Path(args[args.index("-of") + 1])
            output_base.with_suffix(".txt").write_text("  распознанный текст  ", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(stt.subprocess, "run", fake_run)
    assert stt.transcribe_audio(tmp_path / "voice.ogg") == "распознанный текст"
    assert calls[0][0] == stt.FFMPEG_BIN
    assert "16000" in calls[0]
    assert "-l" in calls[1] and calls[1][calls[1].index("-l") + 1] == "ru"
    assert "-t" in calls[1] and calls[1][calls[1].index("-t") + 1] == "1"


def test_transcribe_audio_rejects_invalid_audio(monkeypatch, tmp_path):
    model = tmp_path / "model.bin"
    binary = tmp_path / "whisper-cli"
    model.touch()
    binary.touch()
    monkeypatch.setattr(stt, "MODEL_PATH", model)
    monkeypatch.setattr(stt, "WHISPER_BIN", str(binary))
    monkeypatch.setattr(
        stt.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )
    with pytest.raises(stt.STTUnsupported):
        stt.transcribe_audio(tmp_path / "bad.ogg")


def test_transcribe_audio_reports_missing_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(stt, "MODEL_PATH", tmp_path / "missing-model.bin")
    monkeypatch.setattr(stt, "WHISPER_BIN", str(tmp_path / "missing-binary"))
    with pytest.raises(stt.STTConfigurationError):
        stt.transcribe_audio(tmp_path / "voice.ogg")

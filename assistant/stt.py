"""Local, free speech-to-text using whisper.cpp."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class STTError(Exception):
    """Base class for safe, user-facing transcription failures."""


class STTUnavailable(STTError):
    """A temporary local transcription failure that should be retried."""


class STTUnsupported(STTError):
    """Audio cannot be processed and should not be retried."""


class STTConfigurationError(STTError):
    """The local STT runtime or model is missing."""


MODEL_PATH = Path(os.environ.get("WHISPER_MODEL_PATH", "/opt/whisper/models/ggml-tiny-q5_1.bin"))
WHISPER_BIN = os.environ.get("WHISPER_BIN", "/usr/local/bin/whisper-cli")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def transcribe_audio(source: str | Path) -> str:
    """Convert Telegram audio to 16 kHz mono WAV and transcribe it in Russian."""
    if not MODEL_PATH.is_file() or not os.path.isfile(WHISPER_BIN):
        raise STTConfigurationError("local_stt_runtime_missing")

    with tempfile.TemporaryDirectory(prefix="klodd-stt-") as directory:
        wav_path = Path(directory) / "voice.wav"
        output_base = Path(directory) / "transcript"
        try:
            converted = subprocess.run(
                [
                    FFMPEG_BIN, "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path),
                ],
                capture_output=True,
                timeout=45,
                check=False,
            )
        except FileNotFoundError:
            raise STTConfigurationError("ffmpeg_missing") from None
        except subprocess.TimeoutExpired:
            raise STTUnavailable("audio_conversion_timeout") from None
        if converted.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size == 0:
            raise STTUnsupported("audio_decode_failed")

        try:
            result = subprocess.run(
                [
                    WHISPER_BIN, "-m", str(MODEL_PATH), "-f", str(wav_path),
                    "-l", "ru", "-nt", "-np", "-t", "1", "-bs", "1",
                    "-otxt", "-of", str(output_base),
                ],
                capture_output=True,
                timeout=240,
                check=False,
            )
        except FileNotFoundError:
            raise STTConfigurationError("whisper_binary_missing") from None
        except subprocess.TimeoutExpired:
            raise STTUnavailable("transcription_timeout") from None
        if result.returncode != 0:
            raise STTUnavailable("transcription_failed")

        transcript_path = output_base.with_suffix(".txt")
        try:
            transcript = transcript_path.read_text(encoding="utf-8").strip()
        except OSError:
            raise STTUnavailable("transcript_output_missing") from None
        if not transcript:
            raise STTUnsupported("empty_transcription")
        return transcript

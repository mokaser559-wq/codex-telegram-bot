"""Local, free speech-to-text using whisper.cpp."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
import urllib.error
import urllib.request
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


def _normalize_assistant_name(transcript: str) -> str:
    """Correct known Whisper variants only when they start the transcript."""
    return re.sub(
        r"^(\s*)(?:Клот|Глот|Клодд{2,}|Клод)(?=\s|[,.:!?—–-]|$)",
        r"\1Клодд",
        transcript,
        count=1,
        flags=re.IGNORECASE,
    )



LOG = logging.getLogger("klodd.stt")


def _transcribe_with_groq(wav_path: Path, api_key: str) -> str:
    """Use Groq's higher-accuracy hosted Whisper; local Whisper remains fallback."""
    boundary = "----KloddSTT" + uuid.uuid4().hex
    fields = (
        ("model", "whisper-large-v3"),
        ("language", "ru"),
        ("response_format", "json"),
        ("temperature", "0"),
        ("prompt", "Русская разговорная речь. Возможные имена и термины: Клодд, таможенное дело, таможенному делу, ТН ВЭД, таможенное оформление."),
    )
    chunks = []
    for name, value in fields:
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ))
    chunks.extend((
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="voice.wav"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n",
        wav_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=b"".join(chunks),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
            "User-Agent": "Klodd-Telegram-Assistant",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise STTUnavailable("groq_transcription_unavailable") from exc
    except urllib.error.HTTPError as exc:
        # Includes free-tier rate limits; the local model takes over without
        # losing the message or requiring a paid plan.
        raise STTUnavailable("groq_transcription_http_error") from exc

    transcript = result.get("text") if isinstance(result, dict) else None
    if not isinstance(transcript, str) or not transcript.strip():
        raise STTUnavailable("groq_transcription_empty")
    return transcript.strip()


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

        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                return _normalize_assistant_name(_transcribe_with_groq(wav_path, groq_key))
            except STTUnavailable as exc:
                LOG.warning("Groq STT unavailable; using local Whisper fallback (%s)", str(exc))

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
        return _normalize_assistant_name(transcript)

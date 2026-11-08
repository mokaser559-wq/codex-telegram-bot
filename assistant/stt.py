"""Groq Whisper transcription for Telegram OGG voice messages."""

from __future__ import annotations

import json
import os
import re
import uuid
import urllib.error
import urllib.request
from pathlib import Path


MAX_AUDIO_BYTES = 20 * 1024 * 1024
GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_PROMPT = (
    "Русская разговорная речь. Возможные имена и термины: Клодд, "
    "таможенное дело, таможенному делу, таможенное оформление, ТН ВЭД."
)


class STTError(Exception):
    """Base class for speech recognition errors."""


class STTUnavailable(STTError):
    """Temporary service or network error; retry is appropriate."""


class STTUnsupported(STTError):
    """Invalid or unsupported audio; retry will not help."""


class STTConfigurationError(STTError):
    """Groq credentials are absent or rejected."""


def _normalize_assistant_name(transcript: str) -> str:
    return re.sub(
        r"(?i)(?<!\w)(?:клот|глот|клоуд|клод{1,})(?=\s|[,.:!?—–-]|$)",
        "Клодд",
        transcript,
        count=1,
    )


def _transcribe_with_groq(source: str | Path, api_key: str) -> str:
    """Upload OGG bytes directly as multipart audio to Groq."""
    path = Path(source)
    try:
        size = path.stat().st_size
        if size <= 0:
            raise STTUnsupported("empty_audio_file")
        if size > MAX_AUDIO_BYTES:
            raise STTUnsupported("audio_file_too_large")
        audio = path.read_bytes()
    except STTUnsupported:
        raise
    except OSError as exc:
        raise STTUnavailable("audio_file_unavailable") from exc

    boundary = "----KloddSTT" + uuid.uuid4().hex
    fields = (
        ("model", "whisper-large-v3"),
        ("language", "ru"),
        ("response_format", "json"),
        ("temperature", "0"),
        ("prompt", GROQ_PROMPT),
    )
    chunks = []
    for name, value in fields:
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"), b"\r\n",
        ))
    filename = path.name if path.suffix.lower() == ".ogg" else "voice.ogg"
    chunks.extend((
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: audio/ogg\r\n\r\n", audio, b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    request = urllib.request.Request(
        GROQ_TRANSCRIPTIONS_URL,
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
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code in (401, 403):
            raise STTConfigurationError("groq_authorization_failed") from None
        if exc.code in (408, 425, 429) or exc.code >= 500:
            raise STTUnavailable("groq_transcription_http_error") from None
        raise STTUnsupported("groq_rejected_audio") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise STTUnavailable("groq_transcription_unavailable") from exc
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise STTUnavailable("groq_invalid_response") from None

    transcript = result.get("text") if isinstance(result, dict) else None
    if not isinstance(transcript, str):
        raise STTUnavailable("groq_invalid_response")
    if not transcript.strip():
        raise STTUnsupported("empty_transcription")
    return _normalize_assistant_name(transcript.strip())


def transcribe_audio(source: str | Path) -> str:
    """Transcribe a Telegram OGG file with Groq Whisper Large v3."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise STTConfigurationError("groq_api_key_missing")
    return _transcribe_with_groq(source, api_key)

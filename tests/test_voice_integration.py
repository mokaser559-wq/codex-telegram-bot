import io
import json
import urllib.error
import uuid

import pytest

import assistant.providers as providers
import assistant.stt as stt
from assistant.storage import PostgresStore
import bridge


def test_groq_uploads_ogg_directly_with_required_options(tmp_path, monkeypatch):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS audio bytes")
    seen = {}

    def fake_urlopen(request, timeout):
        seen.update(url=request.full_url, body=request.data, timeout=timeout)
        return io.BytesIO(json.dumps({"text": "Клод, помоги с ТН ВЭД"}).encode())

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(stt.urllib.request, "urlopen", fake_urlopen)
    assert stt.transcribe_audio(path) == "Клодд, помоги с ТН ВЭД"
    assert b"whisper-large-v3" in seen["body"]
    assert b"language" in seen["body"] and b"ru" in seen["body"]
    assert b"temperature" in seen["body"] and b"0" in seen["body"]
    assert b"filename=\"voice.ogg\"" in seen["body"]
    assert b"Content-Type: audio/ogg" in seen["body"]
    assert seen["timeout"] == 90


@pytest.mark.parametrize("code", [401, 403])
def test_groq_auth_http_errors_are_configuration_errors(tmp_path, monkeypatch, code):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS")

    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, code, "denied", {}, io.BytesIO())

    monkeypatch.setattr(stt.urllib.request, "urlopen", fail)
    with pytest.raises(stt.STTConfigurationError):
        stt._transcribe_with_groq(path, "secret")


@pytest.mark.parametrize("code", [408, 425, 429, 500, 503])
def test_groq_rate_limit_and_server_errors_are_retryable(tmp_path, monkeypatch, code):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS")

    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, code, "temporary", {}, io.BytesIO())

    monkeypatch.setattr(stt.urllib.request, "urlopen", fail)
    with pytest.raises(stt.STTUnavailable):
        stt._transcribe_with_groq(path, "secret")


def test_invalid_audio_and_empty_transcription_are_permanent(tmp_path, monkeypatch):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS")

    def empty_response(request, timeout):
        return io.BytesIO(b'{"text":"  "}')

    monkeypatch.setattr(stt.urllib.request, "urlopen", empty_response)
    with pytest.raises(stt.STTUnsupported, match="empty_transcription"):
        stt._transcribe_with_groq(path, "key")
    path.write_bytes(b"")
    with pytest.raises(stt.STTUnsupported, match="empty_audio_file"):
        stt._transcribe_with_groq(path, "key")


def test_network_errors_are_retryable_and_auth_key_is_required(tmp_path, monkeypatch):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(stt.STTConfigurationError):
        stt.transcribe_audio(path)
    monkeypatch.setattr(
        stt.urllib.request, "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(stt.STTUnavailable):
        stt._transcribe_with_groq(path, "key")


def test_stt_has_no_local_whisper_or_ffmpeg_dependencies():
    import inspect

    source = inspect.getsource(stt)
    assert "subprocess" not in source
    assert "ffmpeg" not in source.lower()
    assert "whisper-cli" not in source


def test_correction_passes_history_and_falls_back_to_raw_text(monkeypatch):
    history = [{"role": "user", "text": "Учусь на таможенном деле."}]
    calls = []

    def capture(prompt, received_history, **kwargs):
        calls.append((prompt, received_history, kwargs))
        return "исправленный текст"

    monkeypatch.setattr(providers, "generate_answer", capture)
    assert providers.correct_transcription("черновой текст", history) == "исправленный текст"
    assert calls[0][1] == history
    assert calls[0][2]["temperature"] == 0.1
    assert "Не отвечай" in calls[0][2]["system_prompt"]

    monkeypatch.setattr(
        providers, "generate_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(providers.ProviderUnavailable("offline")),
    )
    assert providers.correct_transcription("непонятный текст") == "непонятный текст"


class _VoiceStore:
    def __init__(self, saved=True):
        self.saved = saved
        self.completed = []
        self.saved_transcriptions = []

    def history(self, *args):
        return [{"role": "user", "text": "Предыдущая фраза"}]

    def save_transcription(self, *args, **kwargs):
        self.saved_transcriptions.append((args, kwargs))
        return self.saved

    def complete_inbox(self, *args):
        self.completed.append(args)

    def ignore_inbox(self, *args):
        raise AssertionError("valid message must not be ignored")

    def permanent_inbox_error(self, *args):
        raise AssertionError(f"unexpected permanent error: {args}")

    def retry_inbox(self, *args):
        raise AssertionError(f"unexpected retry: {args}")


class _Item:
    update_id = 321
    attempts = 1
    generation = 4
    claim_token = uuid.uuid4()

    def __init__(self, payload):
        self.payload = payload


def _voice_update():
    return {
        "update_id": 321,
        "message": {
            "from": {"id": bridge.ALLOWED_USER_ID},
            "chat": {"id": 55},
            "voice": {"file_id": "file-id", "duration": 4, "file_size": 10},
        },
    }


def test_voice_message_uses_pipeline_saves_before_answer_and_passes_history(monkeypatch):
    store = _VoiceStore()
    history_seen = []
    monkeypatch.setattr(bridge, "_download_voice_file", lambda *args: None)
    monkeypatch.setattr(bridge, "transcribe_audio", lambda path: "raw words")
    monkeypatch.setattr(bridge, "correct_transcription", lambda raw, history: history_seen.append(history) or "corrected words")
    monkeypatch.setattr(bridge, "process_message", lambda text, history, **kwargs: "answer")
    bridge.process_inbox_item(store, _Item(_voice_update()))
    assert history_seen == [store.history(55, 321, bridge.MEMORY_LIMIT)]
    assert store.saved_transcriptions[0][0][2] == "corrected words"
    assert store.saved_transcriptions[0][1]["raw_transcript"] == "raw words"
    assert store.completed and store.completed[0][3] == "answer"


def test_stale_claim_stops_voice_pipeline(monkeypatch):
    store = _VoiceStore(saved=False)
    monkeypatch.setattr(bridge, "_download_voice_file", lambda *args: None)
    monkeypatch.setattr(bridge, "transcribe_audio", lambda path: "raw words")
    monkeypatch.setattr(bridge, "correct_transcription", lambda raw, history: raw)
    monkeypatch.setattr(bridge, "process_message", lambda *args, **kwargs: pytest.fail("stale claim continued"))
    bridge.process_inbox_item(store, _Item(_voice_update()))
    assert not store.completed


@pytest.mark.parametrize("active", [True, False])
def test_storage_transcription_fences_claim_and_never_overwrites_user_message(active):
    class Cursor:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params):
            self.calls.append((query, params))

        def fetchone(self):
            return (321,) if active and len(self.calls) == 1 else None

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return self.cursor_instance

    connection = Connection()
    store = PostgresStore(connect=lambda: connection)
    item = _Item(_voice_update())
    result = store.save_transcription(
        item.update_id, 55, "corrected", item.claim_token,
        item.generation, raw_transcript="raw",
    )
    statements = connection.cursor_instance.calls
    assert result is active
    assert "status = 'processing'" in statements[0][0]
    assert "claim_token = %s AND generation = %s" in statements[0][0]
    assert statements[0][1][:2] == ("corrected", "raw")
    if active:
        assert "ON CONFLICT (update_id, role) DO NOTHING" in statements[1][0]
        assert statements[1][1] == (55, 321, "corrected")
    else:
        assert len(statements) == 1


def test_cached_voice_transcription_skips_stt_but_checks_claim(monkeypatch):
    update = _voice_update()
    update["_klodd"] = {"transcription": "cached", "raw_transcription": "raw"}
    store = _VoiceStore()
    monkeypatch.setattr(bridge, "transcribe_audio", lambda *args: pytest.fail("cache missed"))
    monkeypatch.setattr(bridge, "process_message", lambda text, history, **kwargs: "answer")
    bridge.process_inbox_item(store, _Item(update))
    assert store.saved_transcriptions[0][0][2] == "cached"


@pytest.mark.parametrize("duration", [None, True, False, 0, -1, 301, "4", 4.5])
def test_voice_rejects_invalid_duration_before_downloading(duration, monkeypatch):
    update = _voice_update()
    update["message"]["voice"]["duration"] = duration
    monkeypatch.setattr(
        bridge, "_download_voice_file",
        lambda *args: pytest.fail("invalid duration must be rejected before download"),
    )
    with pytest.raises(stt.STTUnsupported, match="invalid_voice_duration"):
        bridge._voice_transcription(_VoiceStore(), _Item(update), update["message"], 55)


def test_ordinary_text_never_calls_stt(monkeypatch):
    store = _VoiceStore()
    update = {"update_id": 322, "message": {
        "from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 55}, "text": "ordinary"
    }}
    monkeypatch.setattr(bridge, "transcribe_audio", lambda *args: pytest.fail("text invoked STT"))
    monkeypatch.setattr(bridge, "process_message", lambda text, history, **kwargs: "answer")
    bridge.process_inbox_item(store, _Item(update))
    assert not store.saved_transcriptions
    assert store.completed


@pytest.mark.parametrize(
    "source",
    ["Клод", "Клодд", "Клоддд", "Клот", "Глот", "Клоуд"],
)
def test_assistant_name_normalization(source):
    assert stt._normalize_assistant_name(source + ", привет") == "Клодд, привет"


@pytest.mark.parametrize("remote_size", [True, False, -1, 1.5, "10"])
def test_telegram_rejects_invalid_remote_file_size(
    tmp_path, monkeypatch, remote_size
):
    monkeypatch.setattr(
        bridge, "telegram",
        lambda *args, **kwargs: {
            "file_path": "voice/file.ogg",
            "file_size": remote_size,
        },
    )

    def unexpected_download(*args, **kwargs):
        pytest.fail("Загрузка не должна начинаться при некорректном размере")

    monkeypatch.setattr(bridge.urllib.request, "urlopen", unexpected_download)

    with pytest.raises(bridge.STTUnsupported, match="voice_file_too_large"):
        bridge._download_voice_file(
            "test-file-id",
            tmp_path / "voice.ogg",
        )

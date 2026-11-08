import io

from assistant import stt


def test_groq_whisper_uses_russian_large_model_and_domain_hints(tmp_path, monkeypatch):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF-test-audio")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = request.data
        captured["timeout"] = timeout
        return io.BytesIO(b'{"text":"Клодд, помоги с таможенным делом."}')

    monkeypatch.setattr(stt.urllib.request, "urlopen", fake_urlopen)

    result = stt._transcribe_with_groq(audio, "test-secret")

    assert result == "Клодд, помоги с таможенным делом."
    assert captured["url"].endswith("/audio/transcriptions")
    assert captured["authorization"] == "Bearer test-secret"
    assert b"whisper-large-v3" in captured["body"]
    assert b"language" in captured["body"] and b"ru" in captured["body"]
    assert "таможенное дело".encode() in captured["body"]
    assert captured["timeout"] == 90

import assistant.providers as providers


def test_correct_transcription_uses_context_and_conservative_prompt(monkeypatch):
    captured = {}
    history = [
        {"role": "user", "text": "Я учусь в университете."},
        {"role": "assistant", "text": "Понял."},
    ]

    def fake_generate_answer(prompt, received_history, **kwargs):
        captured.update(prompt=prompt, history=received_history, **kwargs)
        return "Клод, помоги мне с учёбой."

    monkeypatch.setattr(providers, "generate_answer", fake_generate_answer)

    result = providers.correct_transcription("Клотом омоким мне с утёбой.", history)

    assert result == "Клод, помоги мне с учёбой."
    assert "Клотом омоким мне с утёбой." in captured["prompt"]
    assert captured["history"] == history
    assert captured["temperature"] == 0.1
    assert "Не добавляй факты" in captured["system_prompt"]
    assert "Верни только исправленный текст" in captured["system_prompt"]

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
    assert "Не добавляй" in captured["system_prompt"]
    assert "Верни только исправленный текст" in captured["system_prompt"]
    assert "имя личного ассистента" in captured["system_prompt"]
    assert "«Клодд»" in captured["system_prompt"]
    assert "обычное слово «клоун»" in captured["system_prompt"]



def test_correction_prompt_distinguishes_assistant_name_from_clown():
    captured = {}

    def fake_generate_answer(prompt, history, **kwargs):
        captured.update(kwargs)
        return "Сегодня я говорил про Клодда своему другу."

    monkeypatch = __import__("pytest").MonkeyPatch()
    try:
        monkeypatch.setattr(providers, "generate_answer", fake_generate_answer)
        result = providers.correct_transcription(
            "Сегодня я говорил про клоуна своему другу.",
            [{"role": "assistant", "text": "Я твой личный ассистент Клодд."}],
        )
    finally:
        monkeypatch.undo()

    assert result == "Сегодня я говорил про Клодда своему другу."
    prompt = captured["system_prompt"]
    assert "говорил про Клодда" in prompt
    assert "видео про Клодда" in prompt
    assert "кто такой клоун" in prompt
    assert "Если контекст допускает оба смысла" in prompt

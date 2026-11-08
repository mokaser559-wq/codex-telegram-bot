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
    assert "настоящее слово «клоун»" in captured["system_prompt"]



def test_correction_prompt_distinguishes_assistant_name_from_clown(monkeypatch):
    captured = {}

    def fake_generate_answer(prompt, history, **kwargs):
        captured.update(kwargs)
        return "Сегодня я говорил про Клодда своему другу."

    monkeypatch.setattr(providers, "generate_answer", fake_generate_answer)
    result = providers.correct_transcription(
        "Сегодня я говорил про клоуна своему другу.",
        [{"role": "assistant", "text": "Я твой личный ассистент Клодд."}],
    )

    assert result == "Сегодня я говорил про Клодда своему другу."
    prompt = captured["system_prompt"]
    assert "говорил про Клодда" in prompt
    assert "видео про Клодда" in prompt
    assert "кто такой клоун" in prompt
    assert "Если контекст допускает оба смысла" in prompt



def test_distinctive_kloount_misrecognition_restores_klodd(monkeypatch):
    monkeypatch.setattr(
        providers,
        "generate_answer",
        lambda *args, **kwargs: "Расскажи, кто такой клоун?",
    )

    result = providers.correct_transcription("Раскаживай кто такой клоунт")

    assert result == "Расскажи, кто такой Клодд?"


def test_real_clown_word_is_not_changed_by_name_fallback(monkeypatch):
    monkeypatch.setattr(
        providers,
        "generate_answer",
        lambda *args, **kwargs: "Расскажи, кто такой клоун?",
    )

    result = providers.correct_transcription("Расскажи, кто такой клоун")

    assert result == "Расскажи, кто такой клоун?"


def test_name_context_restores_klodd_in_cloud_like_spellings(monkeypatch):
    monkeypatch.setattr(
        providers,
        "generate_answer",
        lambda *args, **kwargs: "Сегодня я говорил про облако своему другу.",
    )
    history = [
        {"role": "user", "text": "Клодд, помоги мне с учёбой."},
        {"role": "assistant", "text": "Конечно, помогу."},
    ]

    result = providers.correct_transcription(
        "Сегодня я говорил про клоуда своему другу.", history
    )

    assert result == "Сегодня я говорил про Клодда своему другу."


def test_real_cloud_context_is_not_rewritten_as_assistant_name(monkeypatch):
    monkeypatch.setattr(
        providers,
        "generate_answer",
        lambda *args, **kwargs: "Я смотрел видео про облако, и это было смешно.",
    )
    history = [
        {"role": "user", "text": "Привет, Клодд!"},
        {"role": "user", "text": "Расскажи про облачные технологии и серверы."},
    ]

    result = providers.correct_transcription(
        "Я смотрел видео про клоуда, и это было смешно.", history
    )

    assert result == "Я смотрел видео про облако, и это было смешно."



def test_heavily_garbled_exam_phrase_is_reconstructed_conservatively(monkeypatch):
    monkeypatch.setattr(
        providers,
        "generate_answer",
        lambda *args, **kwargs: (
            "Клодд, я хочу подготовиться как за именем состав, "
            "план занятий на неделю и разобраться в самых сложных темах."
        ),
    )

    result = providers.correct_transcription(
        "Клодд, я хочу подготовиться как за именем состав, "
        "план занятий на неделю и разобраться в самых сложных темах."
    )

    assert result == (
        "Клодд, я хочу подготовиться к экзаменам, составить "
        "план занятий на неделю и разобраться в самых сложных темах."
    )


def test_garbled_phrase_recovery_rule_is_in_correction_prompt(monkeypatch):
    captured = {}

    def fake_generate_answer(prompt, history, **kwargs):
        captured.update(kwargs)
        return "Исправленный текст."

    monkeypatch.setattr(providers, "generate_answer", fake_generate_answer)
    providers.correct_transcription("Неразборчивая фраза.")

    prompt = captured["system_prompt"]
    assert "грамматически бессмысленный фрагмент" in prompt
    assert "к экзаменам, составить план занятий" in prompt
    assert "не достраивай отсутствующие мысли" in prompt


def test_garbled_exam_phrase_is_repaired_before_ai_call(monkeypatch):
    captured = {}

    def fake_generate_answer(prompt, history, **kwargs):
        captured["prompt"] = prompt
        return "План занятий на неделю."

    monkeypatch.setattr(providers, "generate_answer", fake_generate_answer)
    providers.correct_transcription(
        "Клодд, я хочу подготовиться как за именем состав, "
        "план занятий на неделю."
    )

    assert "подготовиться к экзаменам, составить" in captured["prompt"]
    assert "как за именем состав" not in captured["prompt"]


def test_garbled_exam_phrase_is_repaired_when_ai_provider_fails(monkeypatch):
    def fail_generate_answer(*args, **kwargs):
        raise providers.ProviderError("temporary provider failure")

    monkeypatch.setattr(providers, "generate_answer", fail_generate_answer)

    result = providers.correct_transcription(
        "Клодд, я хочу подготовиться как за именем состав, "
        "план занятий на неделю."
    )

    assert result == (
        "Клодд, я хочу подготовиться к экзаменам, составить "
        "план занятий на неделю."
    )



def test_customs_study_vocabulary_is_available_to_correction(monkeypatch):
    captured = {}

    def fake_generate_answer(prompt, history, **kwargs):
        captured.update(kwargs)
        return "Клодд, помоги подготовиться к экзаменам по таможенному делу."

    monkeypatch.setattr(providers, "generate_answer", fake_generate_answer)
    result = providers.correct_transcription(
        "Клодд помоги подготовиться к экзаминам по тому же мы делом."
    )

    assert result == "Клодд, помоги подготовиться к экзаменам по таможенному делу."
    assert "таможенное дело" in captured["system_prompt"]
    assert "ТН ВЭД" in captured["system_prompt"]
    assert "по звучанию" in captured["system_prompt"]


def test_final_exam_phrase_guard_uses_whitespace_regex(monkeypatch):
    monkeypatch.setattr(
        providers,
        "generate_answer",
        lambda *args, **kwargs: (
            "Клодд, хочу подготовиться как за именем состав, "
            "план занятий на неделю."
        ),
    )

    result = providers.correct_transcription(
        "Клодд, хочу подготовиться к экзаменам, составить план занятий."
    )

    assert "подготовиться к экзаменам, составить" in result
    assert "как за именем состав" not in result

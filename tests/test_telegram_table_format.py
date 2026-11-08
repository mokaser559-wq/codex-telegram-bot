from assistant.telegram_format import markdown_to_telegram_html


def test_markdown_table_is_rendered_as_readable_telegram_list():
    source = """| Что | Обычная программа | Искусственный интеллект |
|-----|-------------------|------------------------|
| Как работает | Выполняет инструкции | Обучается на данных |
| Пример | Калькулятор | Распознавание речи |"""

    rendered = markdown_to_telegram_html(source)

    assert "|" not in rendered
    assert "-----" not in rendered
    assert "<b>Как работает</b>: Выполняет инструкции" in rendered
    assert "<b>Пример</b>: Калькулятор" in rendered
    assert "Обучается на данных" in rendered
    assert "Распознавание речи" in rendered

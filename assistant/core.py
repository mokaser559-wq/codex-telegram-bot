from assistant.providers import generate_answer
from assistant.router import route_message
from assistant.search import (
    SearchError,
    format_search_context,
    format_source_links,
    needs_search,
    search_tavily,
)


def process_message(text, history=None, codex_handler=None, status_handler=None, requeue_handler=None):
    """Обрабатывает сообщение и возвращает текст ответа."""
    history = history or []
    route = route_message(text)

    if route == "local":
        if text == "/start":
            return "Привет! 😊 Я Клодд. Готов помочь."
        if text == "/help":
            return (
                "Я Клодд — твой личный AI-ассистент. "
                "Могу отвечать на вопросы и помогать с техническими задачами. "
                "Для прямого обращения к Codex используй /codex. "
                "Для повтора сохранённой записи: /retry inbox <id> или /retry outbox <id>."
            )
        if text == "/status":
            if status_handler is not None:
                return status_handler()
            return "Я на связи. Центральное ядро работает."
        if text.startswith("/retry "):
            parts = text.split()
            if len(parts) != 3 or parts[1] not in ("inbox", "outbox"):
                return "Формат: /retry inbox <update_id> или /retry outbox <outbox_id>."
            try:
                item_id = int(parts[2])
            except ValueError:
                return "ID должен быть целым числом."
            if requeue_handler is None:
                return "Повторная постановка пока недоступна."
            return requeue_handler(parts[1], item_id)

    if route == "codex":
        if codex_handler is None:
            return "Обработчик Codex пока не подключён."
        prompt = text.removeprefix("/codex").strip()
        if not prompt:
            return "Напиши задачу после команды /codex."
        return codex_handler(prompt, history)

    if needs_search(text):
        try:
            results = search_tavily(text)
        except SearchError:
            return "Сейчас не удалось выполнить поиск, поэтому я не могу надёжно проверить актуальные сведения. Попробуй повторить запрос позже."
        if not results:
            return "Поиск не нашёл подходящих источников, поэтому я не могу надёжно подтвердить актуальные сведения."
        context = format_search_context(results)
        answer = generate_answer(
            text, history, search_context=context,
            search_notice=(
                "Поиск выполнен. Отвечай только по найденным результатам. "
                "Не выдумывай даты или заголовки; используй точные URL из контекста. "
                "В конце перечисли источники, которые действительно использовал."
            ),
        )
        source_links = format_source_links(results)
        if source_links:
            answer = answer.rstrip() + "\n\n" + source_links
        return answer
    return generate_answer(text, history)

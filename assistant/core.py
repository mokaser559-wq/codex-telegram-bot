from assistant.providers import generate_answer
from assistant.router import route_message


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

    return generate_answer(text, history)

from assistant.providers import generate_answer
from assistant.router import route_message


def process_message(text, history=None, codex_handler=None):
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
                "Для прямого обращения к Codex используй /codex."
            )
        if text == "/status":
            return "Я на связи. Центральное ядро работает."

    if route == "codex":
        if codex_handler is None:
            return "Обработчик Codex пока не подключён."
        prompt = text.removeprefix("/codex").strip()
        if not prompt:
            return "Напиши задачу после команды /codex."
        return codex_handler(prompt, history)

    return generate_answer(text, history)

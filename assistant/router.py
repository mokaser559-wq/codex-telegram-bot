import re


CODEX_PATTERNS = (
    r"\bнапиши\s+(код|скрипт|программу|функцию)",
    r"\bсоздай\s+(файл|скрипт|программу|проект)",
    r"\bисправь\s+(код|ошибку|баг|файл|скрипт)",
    r"\bизмени\s+(код|файл|скрипт|проект)",
    r"\bоткрой\s+(файл|папку|терминал|браузер)",
    r"\bзапусти\s+(проект|скрипт|программу|тесты|команду)",
    r"\bустанови\s+(пакет|библиотеку|зависимости)",
    r"\bпроанализируй\s+(проект|репозиторий|код|файлы)",
    r"\bработай\s+с\s+(файлами|проектом|репозиторием)",
    r"\bотлад",
    r"\bвыполни\s+команду\s+в\s+(терминале|консоли)",
    r"\bна\s+моём\s+компьютере\b",
    r"\bна\s+моем\s+компьютере\b",
    r"\bна\s+рабочем\s+столе\b",
)

IDENTITY_PATTERNS = (
    r"\bкто ты\b", r"\bчто ты за", r"\bтвоя модель\b", r"\bкакая у тебя модель\b",
    r"\bназови свою модель\b", r"\bчерез какой ai\b", r"\bкакой ты ai\b",
    r"\bкак ты работаешь\b", r"\bтвои возможности\b",
)


def route_message(text):
    """Возвращает тип обработки: local, chat или codex."""
    text = text.strip()

    if text in ("/start", "/help", "/status") or text.startswith("/retry "):
        return "local"

    if text == "/codex" or text.startswith("/codex "):
        return "codex"

    normalized = text.lower()
    # Questions about Klodd's identity and active model always go through
    # the configured primary AI, even if future technical keywords overlap.
    if any(re.search(pattern, normalized) for pattern in IDENTITY_PATTERNS):
        return "chat"
    if any(re.search(pattern, normalized) for pattern in CODEX_PATTERNS):
        return "codex"

    return "chat"

import re

from assistant.providers import generate_answer
from assistant.router import route_message
from assistant.search import (
    SearchError,
    filter_search_results,
    filter_ozon_product_results,
    verify_ozon_product_results,
    extract_ozon_budget,
    format_ozon_search_context,
    format_ozon_debug_report,
    _is_ozon_product_query,
    format_search_context,
    format_source_links,
    needs_search,
    search_tavily,
)


def _filter_ozon_budget_results(results, budget):
    """Keep only cards with a live, explicit Ozon price and in-stock state."""
    filtered = []
    for item in results:
        price = item.get("_ozon_price_rub")
        if item.get("_ozon_live_verified") is not True:
            continue
        if item.get("_ozon_live_available") is not True:
            continue
        if not isinstance(price, (int, float)) or isinstance(price, bool) or price <= 0:
            continue
        if budget is not None and price > budget:
            continue
        filtered.append(item)
    return filtered


def process_message(text, history=None, codex_handler=None, status_handler=None, requeue_handler=None):
    """Обрабатывает сообщение и возвращает текст ответа."""
    history = history or []
    if text == "/ozon_debug":
        try:
            diagnostic_results = search_tavily(
                "Найди на Ozon беспроводные наушники Anker Soundcore"
            )
            return format_ozon_debug_report(diagnostic_results)
        except SearchError:
            return "Не удалось получить диагностический ответ Tavily. Ключ и настройки не изменялись."
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
            raw_results = search_tavily(text)
            is_ozon = _is_ozon_product_query(text)
            if is_ozon:
                budget = extract_ozon_budget(text)
                results = _filter_ozon_budget_results(
                    verify_ozon_product_results(raw_results), budget
                )
            else:
                results = filter_search_results(raw_results)
        except SearchError:
            return "Сейчас не удалось выполнить поиск, поэтому я не могу надёжно проверить актуальные сведения. Попробуй повторить запрос позже."
        if not results:
            if is_ozon:
                budget = extract_ozon_budget(text)
                if budget is not None:
                    return (
                        f"Не удалось подтвердить на карточках Ozon товары до {budget} ₽ с актуальной ценой и наличием. "
                        "Я не буду рекомендовать товары по устаревшим поисковым ценам или без подтверждённого наличия. "
                        "Возможно, Ozon временно не отдаёт данные карточек."
                    )
                return (
                    "Не удалось подтвердить актуальную цену и наличие товаров на карточках Ozon. "
                    "Поисковая выдача сама по себе не считается подтверждением. Попробуй повторить позже."
                )
            return (
                "Поиск не нашёл прямых ссылок на отдельные публикации. "
                "Я не хочу выдавать страницы разделов или главные страницы за источники."
            )
        context = format_ozon_search_context(results) if is_ozon else format_search_context(results)
        answer = generate_answer(
            text, history, search_context=context,
            search_notice=(
                "Поиск выполнен. Отвечай только по найденным результатам. "
                + ("В контексте оставлены только товары, у которых цена и наличие проверены непосредственно по актуальной карточке Ozon и цена не выше бюджета. Перечисляй только эти товары, не добавляй товары по памяти или из прошлых сообщений. " if is_ozon and budget is not None else "")
                + "Не выдумывай даты или заголовки, используй только факты, "
                "которые подтверждаются фрагментами. Не создавай отдельный "
                "список источников в конце ответа — ссылки будут добавлены автоматически."
            ),
        )
        source_links = format_source_links(
            results, answer,
            heading="Карточки товаров на Ozon" if is_ozon else "Ссылки на найденные публикации",
        )
        if source_links:
            answer = answer.rstrip() + "\n\n" + source_links
        return answer
    return generate_answer(text, history)

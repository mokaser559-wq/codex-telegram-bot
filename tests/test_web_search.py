import io
import json
import urllib.error

import pytest

import assistant.core as core
import assistant.providers as providers
import assistant.search as search


def test_format_search_context_contains_publication_date_and_exact_url():
    formatted = search.format_search_context([
        {
            "title": "Example",
            "url": "https://example.test/article",
            "published_date": "2026-11-10",
            "content": "Snippet",
        }
    ], today="2026-11-11")
    assert "Дата поиска: 2026-11-11" in formatted
    assert "Заголовок: Example" in formatted
    assert "Дата публикации: 2026-11-10" in formatted
    assert "URL: https://example.test/article" in formatted
    assert "Фрагмент: Snippet" in formatted


def test_missing_publication_date_is_explicit():
    formatted = search.format_search_context([
        {"title": "Example", "url": "https://example.test/article", "content": "Snippet"}
    ], today="2026-11-11")
    assert "Дата публикации: не указана" in formatted


def test_listing_and_homepage_urls_are_filtered_before_model_context():
    results = [
        {"title": "Tag page", "url": "https://iz.ru/tag/iskusstvennyi-intellekt", "content": "listing"},
        {"title": "Topic page", "url": "https://example.test/topics/ai", "content": "listing"},
        {"title": "Homepage", "url": "https://example.test/", "content": "home"},
        {"title": "News index", "url": "https://example.test/news", "content": "index"},
        {"title": "Article", "url": "https://example.test/news/2026/11/article-title", "content": "article"},
    ]
    filtered = search.filter_search_results(results)
    assert [item["title"] for item in filtered] == ["Article"]
    context = search.format_search_context(results, today="2026-11-11")
    assert "Article" in context
    assert "Tag page" not in context
    assert "Topic page" not in context
    assert "Homepage" not in context


def test_nested_article_paths_are_not_rejected_as_listing_pages():
    results = [
        {"title": "News article", "url": "https://example.test/news/ai-breakthrough"},
        {"title": "Topic article", "url": "https://example.test/topics/ai/new-model"},
        {"title": "Category article", "url": "https://example.test/category/ai/report-2026"},
        {"title": "Tag index", "url": "https://example.test/tag/ai"},
        {"title": "Category index", "url": "https://example.test/category/ai"},
        {"title": "News index", "url": "https://example.test/news"},
    ]
    assert [item["title"] for item in search.filter_search_results(results)] == [
        "News article", "Topic article", "Category article"
    ]


def test_filter_removes_duplicate_urls_with_tracking_and_trailing_slash():
    results = [
        {"title": "First", "url": "https://EXAMPLE.test/story/?utm_source=feed"},
        {"title": "Duplicate", "url": "https://example.test/story#section"},
    ]
    assert [item["title"] for item in search.filter_search_results(results)] == ["First"]


def test_invalid_http_hosts_and_ports_are_rejected():
    assert search._valid_http_url("https://example.test:bad/article") == ""
    assert search._valid_http_url("https://user@example.test/article") == ""
    assert search._valid_http_url("https:///missing-host") == ""



def test_news_filter_rejects_social_video_and_theme_pages():
    results = [
        {"title": "YouTube short", "url": "https://www.youtube.com/shorts/abc"},
        {"title": "Facebook video", "url": "https://www.facebook.com/page/videos/123"},
        {"title": "Kommersant topic", "url": "https://www.kommersant.ru/theme/2912"},
        {"title": "Actual article", "url": "https://example.test/news/ai-model-launch"},
    ]
    assert [item["title"] for item in search.filter_search_results(results)] == ["Actual article"]


def test_tavily_news_query_uses_news_topic_and_advanced_depth(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"results":[]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    search.search_tavily("Найди последние новости об искусственном интеллекте")
    assert captured["topic"] == "news"
    assert captured["search_depth"] == "advanced"
    assert captured["include_published_date"] is True
    assert captured["max_results"] == 10
    assert "youtube.com" in captured["exclude_domains"]
    assert "facebook.com" in captured["exclude_domains"]


def test_general_tavily_query_keeps_general_search_settings(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"results":[]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    search.search_tavily("Объясни принцип работы нейронной сети")
    assert "topic" not in captured
    assert "search_depth" not in captured
    assert "exclude_domains" not in captured



def test_explicit_month_range_is_sent_to_tavily_and_filters_out_of_range_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(
        search, "requested_date_range",
        lambda query: {"start_date": "2026-11-01", "end_date": "2026-11-12"},
    )
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"results": [
                {"title": "September", "url": "https://example.test/september", "published_date": "2026-09-30"},
                {"title": "November", "url": "https://example.test/november", "published_date": "2026-11-10"},
                {"title": "Unknown date", "url": "https://example.test/unknown"},
            ]}).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    results = search.search_tavily("Новости ИИ за ноябрь 2026")
    assert captured["start_date"] == "2026-11-01"
    assert captured["end_date"] == "2026-11-12"
    assert [item["title"] for item in results] == ["November", "Unknown date"]


def test_requested_date_range_understands_russian_month_names():
    assert search.requested_date_range(
        "Новости ИИ за ноябрь 2026 года", today=search.date(2026, 11, 11)
    ) == {"start_date": "2026-11-01", "end_date": "2026-11-12"}
    assert search.requested_date_range(
        "Новости ИИ за сентябрь 2026", today=search.date(2026, 11, 11)
    ) == {"start_date": "2026-09-01", "end_date": "2026-10-01"}


def test_filter_handles_non_list_payload_safely():
    assert search.filter_search_results(None) == []
    assert search.filter_search_results({"url": "https://example.test/story"}) == []


def test_source_links_use_exact_article_urls_skip_invalid_and_deduplicate():
    links = search.format_source_links([
        {"title": "Article A", "url": "https://example.test/a", "published_date": "2026-11-10"},
        {"title": "Bad URL", "url": "javascript:alert(1)"},
        {"title": "Listing", "url": "https://example.test/tag/ai"},
        {"title": "Article B", "url": "https://example.test/b"},
        {"title": "Duplicate", "url": "https://example.test/a"},
        {"title": "No URL"},
    ])
    assert links == (
        "Ссылки на найденные публикации:\n"
        "- Article A — 2026-11-10: https://example.test/a\n"
        "- Article B: https://example.test/b"
    )
    assert "javascript:" not in links
    assert "/tag/" not in links


def test_source_links_do_not_repeat_urls_already_in_answer():
    answer = "Новость подтверждена: https://example.test/a"
    links = search.format_source_links([
        {"title": "Article A", "url": "https://example.test/a"},
        {"title": "Article B", "url": "https://example.test/b"},
    ], answer)
    assert links == (
        "Ссылки на найденные публикации:\n"
        "- Article B: https://example.test/b"
    )


def test_source_links_sanitize_multiline_titles_and_reject_whitespace_urls():
    links = search.format_source_links([
        {"title": "Article\n- Fake item", "url": "https://example.test/a"},
        {"title": "Bad URL", "url": "https://example.test/a b"},
    ])
    assert links == (
        "Ссылки на найденные публикации:\n"
        "- Article - Fake item: https://example.test/a"
    )


def test_source_links_empty_when_no_valid_urls():
    assert search.format_source_links([
        {"title": "No URL", "url": "not-a-url"},
        {"title": "Tag", "url": "https://example.test/tag/ai"},
    ]) == ""


def test_ordinary_message_and_stt_correction_do_not_search(monkeypatch):
    monkeypatch.setattr(core, "search_tavily", lambda query: pytest.fail("unexpected search"))
    monkeypatch.setattr(core, "generate_answer", lambda text, history: "ok")
    assert core.process_message("Привет, как дела?") == "ok"
    assert providers.correct_transcription("Привет, как дела?") == "Привет, как дела?"


def test_search_route_preserves_history_and_appends_deterministic_sources(monkeypatch):
    history = [{"role": "user", "text": "Предыдущий вопрос"}]
    calls = []
    results = [
        {
            "title": "Official article",
            "url": "https://openai.com/index/example/",
            "published_date": "2026-11-10",
            "content": "Current fact",
        },
        {
            "title": "Tag page",
            "url": "https://example.test/tag/ai",
            "content": "Not an article",
        },
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    monkeypatch.setattr(core, "generate_answer", lambda *args, **kwargs: calls.append((args, kwargs)) or "Ответ")
    answer = core.process_message("Какие последние новости сегодня?", history)
    args, kwargs = calls[0]
    assert args == ("Какие последние новости сегодня?", history)
    assert "Current fact" in kwargs["search_context"]
    assert "Not an article" not in kwargs["search_context"]
    assert "2026-11-10" in kwargs["search_context"]
    assert "отдельный список источников" in kwargs["search_notice"]
    assert answer.endswith(
        "Ссылки на найденные публикации:\n"
        "- Official article — 2026-11-10: https://openai.com/index/example/"
    )


@pytest.mark.parametrize("status", [434, 500])
def test_tavily_http_errors_are_sanitized(monkeypatch, status):
    monkeypatch.setenv("TAVILY_API_KEY", "private-test-key")

    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, status, "failure", {}, None)

    monkeypatch.setattr(search.urllib.request, "urlopen", fail)
    with pytest.raises(search.SearchError) as error:
        search.search_tavily("query")
    assert str(status) in str(error.value)
    assert "private-test-key" not in str(error.value)


@pytest.mark.parametrize("provider", providers.PROVIDERS, ids=lambda p: p["name"])
def test_providers_receive_search_context_and_system_rule(monkeypatch, provider):
    monkeypatch.setenv(provider["key_env"], "fake-key")
    other = next(p for p in providers.PROVIDERS if p != provider)
    monkeypatch.delenv(other["key_env"], raising=False)
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"choices":[{"message":{"content":"answer"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    history = [{"role": "user", "text": "Earlier"}]
    providers.generate_answer(
        "Question", history, search_context="Дата поиска: today\nФрагмент: result",
        search_notice="Search notice",
    )
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert search.SEARCH_SYSTEM_RULE in messages[0]["content"]
    assert "точный URL" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Earlier"}
    assert "Search notice" in messages[-1]["content"]
    assert "result" in messages[-1]["content"]



def test_ozon_product_query_uses_domain_restricted_advanced_search(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"results":[]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    search.search_tavily("Найди на Ozon беспроводные наушники до 5000 рублей")
    assert captured["query"].startswith("site:ozon.ru/product/ ")
    assert captured["include_domains"] == ["ozon.ru"]
    assert captured["search_depth"] == "advanced"
    assert captured["max_results"] == 6
    assert captured["include_raw_content"] == "markdown"
    assert "topic" not in captured


def test_ozon_filter_keeps_unique_cards_and_normalizes_review_urls():
    results = [
        {"title": "Product", "url": "https://www.ozon.ru/product/headphones-123456/"},
        {"title": "Duplicate", "url": "https://ozon.ru/product/headphones-123456/?utm_source=search"},
        {"title": "Category", "url": "https://www.ozon.ru/category/headphones/"},
        {"title": "Search", "url": "https://www.ozon.ru/search/?text=headphones"},
        {"title": "Other marketplace", "url": "https://www.wildberries.ru/catalog/123/detail.aspx"},
        {"title": "Reviews", "url": "https://www.ozon.ru/product/other-headphones-789/reviews/"},
        {"title": "Homepage", "url": "https://www.ozon.ru/"},
    ]
    filtered = search.filter_ozon_product_results(results)
    assert [item["title"] for item in filtered] == ["Product", "Reviews"]
    assert filtered[1]["url"] == "https://www.ozon.ru/product/other-headphones-789/"


def test_ozon_search_route_uses_product_filter_and_card_heading(monkeypatch):
    results = [
        {
            "title": "Наушники Example",
            "url": "https://www.ozon.ru/product/headphones-123456/",
            "content": "Цена 3 990 ₽, Bluetooth 5.3",
        },
        {
            "title": "Category",
            "url": "https://www.ozon.ru/category/headphones/",
            "content": "not a product card",
        },
    ]
    calls = []
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    monkeypatch.setattr(
        core, "generate_answer",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "Подходящий товар",
    )
    answer = core.process_message("Найди на Ozon беспроводные наушники")
    assert "Цена 3 990 ₽" in calls[0][1]["search_context"]
    assert "not a product card" not in calls[0][1]["search_context"]
    assert answer.endswith(
        "Карточки товаров на Ozon:\n"
        "- Наушники Example: https://www.ozon.ru/product/headphones-123456/"
    )



def test_ozon_price_is_extracted_only_when_unambiguous():
    item = {"content": "Цена 3 990 ₽"}
    assert search._extract_ozon_price(item) == (3990, "примерная цена из поискового фрагмента Tavily", False)
    assert search._extract_ozon_price({"content": "Цена 3 990 ₽, старая цена 4 990 ₽"}) == (3990, "примерная цена из поискового фрагмента Tavily", False)
    assert search._extract_ozon_price({"raw_content": "Стоимость: 2 799 руб."}) == (
        2799, "извлечённое содержимое карточки", True
    )
    assert search._extract_ozon_price({"raw_content": "Доставка 299 ₽"}) == (None, "", False)


@pytest.mark.parametrize(("query", "expected"), [
    ("Найди на Ozon наушники до 5000 рублей", 5000),
    ("Ozon, бюджетом до 12 000 ₽", 12000),
    ("Подбери на Ozon не дороже 3500", 3500),
    ("Найди на Ozon наушники", None),
])
def test_extract_ozon_budget(query, expected):
    assert search.extract_ozon_budget(query) == expected


def test_ozon_accepts_single_unlabeled_price_from_card_content():
    assert search._extract_ozon_price({
        "raw_content": "Anker Soundcore P42i\n4 502 ₽\n-41%"
    }) == (4502, "единственная цена в содержимом карточки", True)


def test_ozon_ignores_struck_old_price_when_current_price_is_clear():
    assert search._extract_ozon_price({
        "raw_content": "4 502 ₽ ~~7 747 ₽~~ -41%"
    }) == (4502, "единственная цена в содержимом карточки", True)
    assert search._extract_ozon_price({
        "raw_content": "4 502 ₽ <del>7 747 ₽</del> -41%"
    }) == (4502, "единственная цена в содержимом карточки", True)


@pytest.mark.parametrize("content", [
    "4 502 ₽\n7 747 ₽",
    "Доставка 299 ₽",
])
def test_ozon_does_not_guess_between_unlabeled_or_delivery_prices(content):
    assert search._extract_ozon_price({"raw_content": content}) == (None, "", False)


@pytest.mark.parametrize("content", [
    "Цена 4 502 ₽\nДоставка 299 ₽",
    "Цена 4 502 ₽\nПлатёж 375 ₽/мес",
])
def test_ozon_keeps_labeled_product_price_separate_from_delivery_and_installments(content):
    assert search._extract_ozon_price({"raw_content": content}) == (
        4502, "извлечённое содержимое карточки", True
    )

def test_ozon_enrichment_keeps_unknown_price_explicit():
    items = search.enrich_ozon_product_results([
        {"title": "Known", "url": "https://ozon.ru/product/a-123/", "content": "Цена 4 500 ₽", "raw_content": "Цена: 4 500 ₽"},
        {"title": "Unknown", "url": "https://ozon.ru/product/b-456/", "content": "Беспроводные наушники"},
        {"title": "Ambiguous", "url": "https://ozon.ru/product/c-789/", "content": "3 900 ₽, старая цена 5 900 ₽"},
        {"title": "Snippet", "url": "https://ozon.ru/product/d-012/", "content": "Цена 3 900 ₽"},
    ])
    assert [item["_ozon_price_rub"] for item in items] == [4500, None, 3900, 3900]
    assert [item["_ozon_price_verified"] for item in items] == [True, False, False, False]
    context = search.format_ozon_search_context(items, today="2026-11-12")
    assert "Цена: 4500 ₽" in context
    assert "примерно 3900 ₽ (поисковый фрагмент Tavily)" in context
    assert context.count("не удалось однозначно подтвердить") == 1
    assert "примерно 3900 ₽ (поисковый фрагмент Tavily)" in context


def test_ozon_snippet_price_is_presented_as_approximate_not_exact():
    item = search.enrich_ozon_product_results([{
        "title": "Anker headphones",
        "url": "https://ozon.ru/product/anker-headphones-123/",
        "content": "Anker headphones, цена 4 700 ₽, старая цена 6 000 ₽",
    }])[0]
    assert item["_ozon_price_rub"] == 4700
    assert item["_ozon_price_verified"] is False
    context = search.format_ozon_search_context([item])
    assert "примерно 4700 ₽" in context
    assert "подтверждена по содержимому карточки" not in context


def test_ozon_budget_route_keeps_unknown_separately_and_excludes_over_budget(monkeypatch):
    results = [
        {"title": "Under budget", "url": "https://ozon.ru/product/a-123/", "content": "Цена 4 500 ₽", "raw_content": "Цена: 4 500 ₽"},
        {"title": "Over budget", "url": "https://ozon.ru/product/b-456/", "content": "Цена 6 500 ₽", "raw_content": "Цена: 6 500 ₽"},
        {"title": "Unknown", "url": "https://ozon.ru/product/c-789/", "content": "Наушники Bluetooth"},
    ]
    calls = []
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    monkeypatch.setattr(
        core, "generate_answer",
        lambda *args, **kwargs: calls.append(kwargs) or "Подходящий товар за 4500 ₽",
    )
    answer = core.process_message("Найди на Ozon наушники до 5000 рублей")
    assert "Under budget" in calls[0]["search_context"]
    assert "Over budget" not in calls[0]["search_context"]
    assert "Unknown" not in calls[0]["search_context"]
    assert "не удалось однозначно подтвердить" not in calls[0]["search_context"]
    assert "Карточки товаров на Ozon" in answer


def test_ozon_budget_route_excludes_unknown_and_over_budget_cards(monkeypatch):
    results = [
        {"title": "Unknown", "url": "https://ozon.ru/product/c-789/", "content": "Наушники Bluetooth"},
        {"title": "Over budget", "url": "https://ozon.ru/product/b-456/", "content": "Цена 6 500 ₽"},
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    calls = []
    monkeypatch.setattr(core, "generate_answer", lambda *args, **kwargs: calls.append(kwargs) or "Не должно вызываться.")
    answer = core.process_message("Найди на Ozon наушники до 5000 рублей")
    assert calls == []
    assert "Не нашёл на Ozon товаров" in answer
    assert "ozon.ru/product/" not in answer


@pytest.mark.parametrize("status_text", [
    "Товар закончился",
    "Нет в наличии",
    "Узнать о поступлении",
    "Sold out",
])
def test_ozon_budget_filter_excludes_unavailable_products(status_text):
    items = [
        {"title": "Available", "_ozon_price_rub": 4500, "content": "В наличии"},
        {"title": "Unavailable", "_ozon_price_rub": 3900, "content": status_text},
    ]
    assert [item["title"] for item in core._filter_ozon_budget_results(items, 5000)] == ["Available"]


def test_ozon_budget_filter_requires_price_and_respects_budget():
    items = [
        {"title": "Approximate within budget", "_ozon_price_rub": 4990},
        {"title": "Over budget", "_ozon_price_rub": 5001},
        {"title": "Unknown price", "_ozon_price_rub": None},
    ]
    assert [item["title"] for item in core._filter_ozon_budget_results(items, 5000)] == [
        "Approximate within budget"
    ]


def test_ozon_budget_route_returns_no_links_when_all_cards_are_unknown_or_unavailable(monkeypatch):
    results = [
        {"title": "Unknown", "url": "https://ozon.ru/product/a-123/", "content": "Наушники"},
        {"title": "Sold out", "url": "https://ozon.ru/product/b-456/", "content": "Товар закончился", "raw_content": "Цена 4 500 ₽"},
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    answer = core.process_message("Найди на Ozon наушники до 5000 рублей")
    assert "Не нашёл на Ozon товаров" in answer
    assert "ozon.ru/product/" not in answer



def test_ozon_search_combines_broad_and_direct_product_queries(monkeypatch):
    responses = [
        {"results": [
            {"title": "Ozon category", "url": "https://www.ozon.ru/category/headphones/"},
            {"title": "Anker Soundcore P42i", "url": "https://www.ozon.ru/product/anker-soundcore-p42i-123/"},
        ]},
        {"results": [
            {"title": "Anker Soundcore P20i", "url": "https://www.ozon.ru/product/anker-soundcore-p20i-456/"},
            {"title": "Anker Soundcore P42i duplicate", "url": "https://www.ozon.ru/product/anker-soundcore-p42i-123/"},
        ]},
    ]
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        return io.BytesIO(json.dumps(responses.pop(0)).encode("utf-8"))

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)

    results = search.search_tavily("Найди на Ozon беспроводные наушники Anker Soundcore до 5000 рублей")

    assert len(requests) == 2
    assert "site:ozon.ru " in requests[0]["query"]
    assert "site:ozon.ru/product/" in requests[1]["query"]
    assert [item["title"] for item in results] == [
        "Ozon category",
        "Anker Soundcore P42i",
        "Anker Soundcore P20i",
    ]

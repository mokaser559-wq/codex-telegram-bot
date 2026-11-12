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
    assert "5000" not in captured["query"]
    assert "беспроводные наушники" in captured["query"]
    assert captured["include_domains"] == ["ozon.ru"]
    assert captured["search_depth"] == "advanced"
    assert captured["max_results"] == 10
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


def test_ozon_popularity_extraction_and_ranking_uses_purchases_then_reviews():
    items = [
        {"title": "Many reviews", "url": "https://ozon.ru/product/reviews-111/",
         "content": "4,8 · 2 400 отзывов"},
        {"title": "Most purchases", "url": "https://ozon.ru/product/purchases-222/",
         "content": "Купили более 10 тыс. раз · 4,7 · 800 отзывов"},
        {"title": "Fewer purchases", "url": "https://ozon.ru/product/fewer-333/",
         "content": "Купили 1 200 раз · 4,9 · 300 отзывов"},
        {"title": "Unknown", "url": "https://ozon.ru/product/unknown-444/"},
        {"title": "Duplicate", "url": "https://ozon.ru/product/purchases-222/?utm_source=x",
         "content": "Купили 99 тыс. раз"},
    ]
    ranked = search.rank_ozon_product_results(items, limit=4)
    assert [item["title"] for item in ranked] == [
        "Most purchases", "Fewer purchases", "Many reviews", "Unknown"
    ]
    assert ranked[0]["_ozon_purchase_count"] == 10000
    assert ranked[0]["_ozon_review_count"] == 800
    assert ranked[0]["_ozon_rating"] == 4.7
    assert ranked[2]["_ozon_review_count"] == 2400


def test_ozon_shortlist_reports_metrics_only_when_found():
    answer = search.format_ozon_product_shortlist(search.rank_ozon_product_results([{
        "title": "Anker Soundcore Space One",
        "url": "https://ozon.ru/product/space-one-123/",
        "content": "Купили 3 500 раз · 4,8 · 1 200 отзывов",
    }]))
    assert "Space One" in answer
    assert "Покупки: 3500" in answer
    assert "Отзывы: 1200" in answer
    assert "Рейтинг: 4,8/5" in answer
    assert "Наличие" in answer
    assert "https://www.ozon.ru/product/space-one-123/" in answer


def test_ozon_shortlist_does_not_claim_popularity_when_metrics_are_missing():
    answer = search.format_ozon_product_shortlist(search.rank_ozon_product_results([{
        "title": "Anker Soundcore Space One",
        "url": "https://ozon.ru/product/space-one-123/",
    }]))
    assert "популярность этих товаров подтвердить не удалось" in answer


def test_ozon_search_route_only_returns_cards_with_explicit_stock_signal(monkeypatch):
    results = [
        {"title": "Anker Soundcore Space One", "url": "https://www.ozon.ru/product/space-one-123/",
         "content": "В наличии · 4,8 · 1200 отзывов"},
        {"title": "Anker Soundcore Q45", "url": "https://www.ozon.ru/product/q45-456/",
         "content": "Товар закончился · 4,9 · 900 отзывов"},
        {"title": "Anker Soundcore P40i", "url": "https://www.ozon.ru/product/p40i-789/",
         "content": "Нет в наличии"},
        {"title": "Unrelated phone case", "url": "https://www.ozon.ru/product/case-012/",
         "content": "В наличии"},
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    answer = core.process_message("Найди на Ozon наушники Anker Soundcore")
    assert "Space One" in answer
    assert "Q45" not in answer
    assert "P40i" not in answer
    assert "case-012" not in answer
    assert "Цена" not in answer


def test_ozon_search_route_returns_unverified_cards_when_stock_is_not_confirmed(monkeypatch):
    results = [
        {"title": "Anker Soundcore C30i", "url": "https://www.ozon.ru/product/c30i-123/"},
        {"title": "Anker Soundcore Space One", "url": "https://www.ozon.ru/product/space-one-456/",
         "content": "Товар закончился"},
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    answer = core.process_message("Найди на Ozon наушники Anker Soundcore")
    assert "наличие не подтверждено" in answer
    assert "C30i" in answer
    assert "Space One" in answer




@pytest.mark.parametrize(("query", "expected"), [
    ("Найди на Ozon беспроводные наушники Anker Soundcore до 5000 рублей", "Найди на Ozon беспроводные наушники Anker Soundcore"),
    ("Ozon, бюджетом до 12 000 ₽, смартфоны", "Ozon, смартфоны"),
    ("Подбери на Ozon не дороже 3500", "Подбери на Ozon"),
    ("Найди на Ozon наушники", "Найди на Ozon наушники"),
])
def test_remove_ozon_budget_constraint(query, expected):
    assert search.remove_ozon_budget_constraint(query) == expected

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



def test_ozon_route_shows_unverified_cards_when_stock_signal_is_missing(monkeypatch):
    items = [
        {"title": "Anker Soundcore Space One", "url": "https://www.ozon.ru/product/space-one-123/"},
        {"title": "Anker Soundcore Q45", "url": "https://www.ozon.ru/product/q45-456/", "content": "Товар закончился"},
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: items)
    answer = core.process_message("Найди на Ozon наушники Anker Soundcore")
    assert "наличие не подтверждено" in answer
    assert "Space One" in answer and "Q45" in answer


def test_ozon_route_uses_category_fallback_when_no_product_cards(monkeypatch):
    items = [
        {"title": "Увлажняющие кремы для лица", "url": "https://www.ozon.ru/category/kremy-dlya-lica-123/"},
        {"title": "Косметика", "url": "https://www.ozon.ru/category/kosmetika-456/"},
        {"title": "Beauty blog", "url": "https://example.com/blog"},
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: items)
    answer = core.process_message("Найди на Ozon увлажняющий крем для лица некомедогенный")
    assert "разделы Ozon" in answer
    assert "category/kremy-dlya-lica-123" in answer
    assert "example.com" not in answer


def test_ozon_search_includes_configured_delivery_city(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setenv("OZON_DELIVERY_CITY", "Санкт-Петербург")
    captured = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"results":[]}'

    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data)["query"])
        return Response()

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    search.search_tavily("Найди на Ozon наушники до 5000 рублей")
    assert len(captured) == 2
    assert all("доставка в Санкт-Петербург" in query for query in captured)
    assert all("5000" not in query for query in captured)


def test_ozon_delivery_city_can_be_configured(monkeypatch):
    monkeypatch.setenv("OZON_DELIVERY_CITY", "Казань")
    assert search.ozon_delivery_city() == "Казань"


def test_ozon_category_query_ignores_marketplace_and_typo_terms():
    results = [{
        "title": "Увлажняющий крем для лица",
        "url": "https://www.ozon.ru/product/face-cream-123/",
    }]
    assert search.filter_ozon_results_for_query(
        results, "Найди увлажняющий крем для лица некамедогенный на озоне, дешёвый но качественный"
    ) == results

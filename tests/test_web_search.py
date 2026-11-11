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

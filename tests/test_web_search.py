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


def test_source_links_use_exact_tavily_urls_and_skip_invalid_urls():
    links = search.format_source_links([
        {
            "title": "Article A",
            "url": "https://example.test/a",
            "published_date": "2026-11-10",
        },
        {"title": "Bad URL", "url": "javascript:alert(1)"},
        {"title": "Article B", "url": "https://example.test/b"},
        {"title": "No URL"},
    ])
    assert links == (
        "Ссылки на найденные материалы:\n"
        "- Article A — 2026-11-10: https://example.test/a\n"
        "- Article B: https://example.test/b"
    )
    assert "javascript:" not in links


def test_source_links_empty_when_no_valid_urls():
    assert search.format_source_links([
        {"title": "No URL", "url": "not-a-url"}
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
        }
    ]
    monkeypatch.setattr(core, "search_tavily", lambda query: results)
    monkeypatch.setattr(core, "generate_answer", lambda *args, **kwargs: calls.append((args, kwargs)) or "Ответ")
    answer = core.process_message("Какие последние новости сегодня?", history)
    args, kwargs = calls[0]
    assert args == ("Какие последние новости сегодня?", history)
    assert "Current fact" in kwargs["search_context"]
    assert "2026-11-10" in kwargs["search_context"]
    assert kwargs["search_notice"]
    assert answer.endswith(
        "Ссылки на найденные материалы:\n"
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

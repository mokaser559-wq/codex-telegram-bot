import html
from html.parser import HTMLParser

import pytest

import bridge
from assistant.telegram_format import markdown_to_telegram_html, split_telegram_html


@pytest.mark.parametrize(
    "source",
    (
        "```python\nстрока < &\nвторая строка",
        "```\nкод & < >\n",
        "до блока\n```python\n<исходный текст> &\nбез закрытия\n",
    ),
)
def test_unclosed_fenced_code_preserves_every_source_character(source):
    rendered = markdown_to_telegram_html(source)

    assert html.unescape(rendered) == source


def test_markdown_styles_headings_and_lists_render_as_telegram_html():
    source = (
        "# Заголовок\n"
        "**жирный**, *курсив*, ~~зачёркнутый~~\n"
        "- первый\n"
        "- второй\n"
        "1. один\n"
        "2. два"
    )

    rendered = markdown_to_telegram_html(source)

    assert rendered == (
        "<b>Заголовок</b>\n"
        "<b>жирный</b>, <i>курсив</i>, <s>зачёркнутый</s>\n"
        "• первый\n"
        "• второй\n"
        "1. один\n"
        "2. два"
    )


def test_markdown_code_and_links_are_escaped_and_rendered():
    source = '`<inline & "code">`\n```python\nx < 2 && y > 1\n```\n[ссылка](https://example.com/?a=1&b=2)'

    rendered = markdown_to_telegram_html(source)

    assert rendered == (
        "<code>&lt;inline &amp; &quot;code&quot;&gt;</code>\n"
        "<pre><code>x &lt; 2 &amp;&amp; y &gt; 1</code></pre>\n"
        '<a href="https://example.com/?a=1&amp;b=2">ссылка</a>'
    )


def test_plain_text_html_symbols_and_quotes_are_always_escaped():
    source = '5 < 7 & 8 > 3; "кавычки" и <b>тег</b>'

    assert markdown_to_telegram_html(source) == (
        "5 &lt; 7 &amp; 8 &gt; 3; &quot;кавычки&quot; и "
        "&lt;b&gt;тег&lt;/b&gt;"
    )


@pytest.mark.parametrize(
    "source",
    (
        "[опасная ссылка](javascript:alert(1))",
        "**незакрытый markdown",
        "```python\n<b>не html",
    ),
)
def test_malformed_or_unsafe_markdown_never_emits_untrusted_html(source):
    rendered = markdown_to_telegram_html(source)

    assert "<a href=\"javascript:" not in rendered
    assert "<b>не html" not in rendered
    if source.startswith("[опасная"):
        assert "javascript:" in rendered
        assert "<a href=" not in rendered
    elif source.startswith("**"):
        assert rendered == source
    else:
        assert rendered == "```python\n&lt;b&gt;не html"


def test_send_falls_back_to_plain_text_when_telegram_rejects_html(monkeypatch):
    calls = []

    def fake_telegram(method, data):
        calls.append((method, data))
        if len(calls) == 1:
            raise bridge.TelegramPermanentError(400)

    monkeypatch.setattr(bridge, "telegram", fake_telegram)

    bridge.send(22, "**ответ** & текст")

    assert calls == [
        ("sendMessage", {"chat_id": 22, "text": "<b>ответ</b> &amp; текст", "parse_mode": "HTML"}),
        ("sendMessage", {"chat_id": 22, "text": "ответ & текст"}),
    ]


def test_send_does_not_fallback_for_non_formatting_telegram_errors(monkeypatch):
    calls = []

    def fail(method, data):
        calls.append(data)
        raise bridge.TelegramPermanentError(403)

    monkeypatch.setattr(bridge, "telegram", fail)

    with pytest.raises(bridge.TelegramPermanentError) as error:
        bridge.send(22, "ответ")

    assert error.value.code == 403
    assert len(calls) == 1


def test_send_fallback_sends_plain_visible_text_for_each_long_chunk(monkeypatch):
    fallback = []

    def reject_html(method, data):
        if data.get("parse_mode") == "HTML":
            raise bridge.TelegramPermanentError(400)
        fallback.append(data["text"])

    monkeypatch.setattr(bridge, "telegram", reject_html)
    expected = "текст & " * 699 + "текст &"
    source = "**" + expected + "**"

    bridge.send(22, source)

    assert len(fallback) > 1
    assert "".join(fallback) == expected
    assert all("<b>" not in chunk and "&amp;" not in chunk for chunk in fallback)


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.tags = []
        self.unbalanced = False

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_endtag(self, tag):
        if not self.tags or self.tags[-1] != tag:
            self.unbalanced = True
            return
        self.tags.pop()


def _parse_visible_text(fragment):
    parser = _VisibleText()
    parser.feed(fragment)
    parser.close()
    return "".join(parser.parts)


def _assert_balanced_html(fragment):
    parser = _VisibleText()
    parser.feed(fragment)
    parser.close()
    assert not parser.unbalanced
    assert parser.tags == []
    return "".join(parser.parts)


def test_split_telegram_html_reopens_nested_tags_and_preserves_entities():
    rendered = "<b><i>" + "😀&amp;" * 1600 + "</i></b>"

    chunks = split_telegram_html(rendered)

    assert len(chunks) > 1
    assert "".join(_assert_balanced_html(fragment) for fragment, _ in chunks) == "😀&" * 1600
    assert "".join(plain for _, plain in chunks) == "😀&" * 1600
    assert all(len(plain.encode("utf-16-le")) // 2 <= 4000 for _, plain in chunks)
    assert all(fragment.startswith("<b><i>") and fragment.endswith("</i></b>") for fragment, _ in chunks)


def test_split_telegram_html_repairs_malformed_nesting_without_losing_text():
    chunks = split_telegram_html("<b><i>ab</b>cd</i>", max_length=2)

    assert all(_assert_balanced_html(fragment) == plain for fragment, plain in chunks)
    assert "".join(plain for _, plain in chunks) == "abcd</i>"


def test_split_telegram_html_escapes_tags_outside_telegram_allowlist():
    source = '<script>abc</script><b onclick="x">def</b><B>up</B>&raw'
    chunks = split_telegram_html(source)

    assert all(_assert_balanced_html(fragment) == plain for fragment, plain in chunks)
    assert "".join(plain for _, plain in chunks) == source
    assert all(
        "<script>" not in fragment and '<b onclick="x">' not in fragment and "<B>" not in fragment
        for fragment, _ in chunks
    )


def test_split_telegram_html_handles_known_and_unknown_entities_without_loss():
    chunks = split_telegram_html("A&amp;&#x1F600;&bogus;&#99999999;", max_length=3)

    assert all(_assert_balanced_html(fragment) == plain for fragment, plain in chunks)
    assert "".join(plain for _, plain in chunks) == "A&😀&bogus;&#99999999;"
    assert all("&bogus;" not in fragment for fragment, _ in chunks)
    assert all("&#99999999;" not in fragment for fragment, _ in chunks)


def test_split_telegram_html_empty_input_returns_no_messages():
    assert split_telegram_html("") == []


def test_split_telegram_html_long_plain_text_respects_utf16_limit():
    text = "a" * 3999 + "😀" + "b" * 3999
    chunks = split_telegram_html(text)

    assert len(chunks) == 3
    assert "".join(plain for _, plain in chunks) == text
    assert all(len(plain.encode("utf-16-le")) // 2 <= 4000 for _, plain in chunks)
    assert all(_assert_balanced_html(fragment) == plain for fragment, plain in chunks)


@pytest.mark.parametrize(
    ("source", "opening", "closing", "expected"),
    (
        ("**" + "ж" * 4100 + "**", "<b>", "</b>", "ж" * 4100),
        ("[" + "с" * 4100 + "](https://example.com)", '<a href="https://example.com">', "</a>", "с" * 4100),
        ("```python\n" + "&" * 4100 + "\n```", "<pre><code>", "</code></pre>", "&" * 4100),
    ),
)
def test_send_splits_after_formatting_without_breaking_html(
    monkeypatch, source, opening, closing, expected,
):
    calls = []
    monkeypatch.setattr(bridge, "telegram", lambda method, data: calls.append(data))

    bridge.send(22, source)

    assert len(calls) == 2
    visible_parts = []
    for call in calls:
        fragment = call["text"]
        assert call["parse_mode"] == "HTML"
        assert fragment.startswith(opening)
        assert fragment.endswith(closing)
        visible = _parse_visible_text(fragment)
        assert len(visible.encode("utf-16-le")) // 2 <= 4000
        visible_parts.append(visible)
    assert "".join(visible_parts) == expected

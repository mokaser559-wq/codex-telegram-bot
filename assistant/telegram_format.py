"""Small, escaped Markdown subset renderer for Telegram HTML messages."""

from __future__ import annotations

import html
import re
from urllib.parse import urlsplit


_FENCE = re.compile(r"^\s*```.*$")
_FENCE_END = re.compile(r"^\s*```\s*$")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
_CODE = re.compile(r"(`+)(.+?)\1")
_STRONG = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__")
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_ITALIC_STAR = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_ITALIC_UNDERSCORE = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")
_HTML_TOKEN = re.compile(
    r"<[^<>]*>|&[^;\s<>&]*;|.",
    re.DOTALL,
)
_HTML_TAG = re.compile(r"<(?P<closing>/)?(?P<name>[a-z]+)(?P<attrs>[^<>]*)>")
_HTML_ENTITY = re.compile(r"&(?:lt|gt|amp|quot|#\d+|#x[0-9a-fA-F]+);\Z")
_SIMPLE_TAGS = {"b", "i", "s", "code", "pre"}
_ANCHOR_HREF = re.compile(r'\s+href="([^"]*)"\s*\Z')
_NUMERIC_ENTITY = re.compile(r"&#(?P<decimal>\d+|[xX][0-9a-fA-F]+);\Z")


def _safe_link(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _decode_telegram_entity(entity: str) -> str | None:
    if entity in ("&lt;", "&gt;", "&amp;", "&quot;"):
        return html.unescape(entity)
    match = _NUMERIC_ENTITY.fullmatch(entity)
    if not match:
        return None
    digits = match.group("decimal")
    try:
        codepoint = int(digits[1:], 16) if digits[:1].lower() == "x" else int(digits)
    except ValueError:
        return None
    if codepoint == 0 or codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
        return None
    return chr(codepoint)


def _inline(text: str, depth: int = 0) -> str:
    if depth > 8:
        return html.escape(text)

    output = []
    index = 0
    while index < len(text):
        link = _LINK.match(text, index)
        if link and _safe_link(link.group(2)):
            label, url = link.groups()
            output.append(f'<a href="{html.escape(url, quote=True)}">{_inline(label, depth + 1)}</a>')
            index = link.end()
            continue

        code = _CODE.match(text, index)
        if code:
            output.append(f"<code>{html.escape(code.group(2))}</code>")
            index = code.end()
            continue

        matched = False
        for pattern, tag in (
            (_STRONG, "b"),
            (_STRIKE, "s"),
            (_ITALIC_STAR, "i"),
            (_ITALIC_UNDERSCORE, "i"),
        ):
            match = pattern.match(text, index)
            if match:
                content = next(group for group in match.groups() if group is not None)
                output.append(f"<{tag}>{_inline(content, depth + 1)}</{tag}>")
                index = match.end()
                matched = True
                break
        if matched:
            continue

        output.append(html.escape(text[index]))
        index += 1
    return "".join(output)


def markdown_to_telegram_html(text: str) -> str:
    """Render common Markdown constructs; escape all unrecognized text."""
    lines = text.split("\n")
    rendered = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _FENCE.match(line):
            end = index + 1
            while end < len(lines) and not _FENCE_END.match(lines[end]):
                end += 1
            if end == len(lines):
                # Incomplete fenced blocks are data, not trusted HTML.
                rendered.extend(html.escape(part) for part in lines[index:])
                break
            code = "\n".join(lines[index + 1:end])
            rendered.append(f"<pre><code>{html.escape(code)}</code></pre>")
            index = end + 1
            continue

        heading = _HEADING.match(line)
        if heading:
            rendered.append(f"<b>{_inline(heading.group(2))}</b>")
        else:
            list_item = _LIST_ITEM.match(line)
            if list_item:
                marker = line.lstrip().split(None, 1)[0]
                prefix = "• " if marker in ("-", "*", "+") else f"{marker} "
                rendered.append(html.escape(prefix) + _inline(list_item.group(1)))
            else:
                rendered.append(_inline(line))
        index += 1
    return "\n".join(rendered)


def split_telegram_html(rendered: str, max_length: int = 4000) -> list[tuple[str, str]]:
    """Split rendered HTML without cutting entities or tags.

    The returned plain text is the visible content for safe API fallback.
    Telegram measures message text after entity parsing; count UTF-16 code
    units conservatively and leave room below the 4096-unit API limit.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    chunks: list[tuple[str, str]] = []
    html_parts: list[str] = []
    plain_parts: list[str] = []
    open_tags: list[tuple[str, str]] = []
    visible_length = 0

    def finish_chunk() -> None:
        if not plain_parts:
            return
        html_parts.extend(f"</{tag}>" for tag, _ in reversed(open_tags))
        chunks.append(("".join(html_parts), "".join(plain_parts)))

    def append_visible(token: str, *, escaped: bool = False, decoded: str | None = None) -> None:
        nonlocal html_parts, plain_parts, visible_length
        if escaped:
            visible = token
        elif decoded is not None:
            visible = decoded
        else:
            visible = html.unescape(token)
        token_length = len(visible.encode("utf-16-le")) // 2
        if visible_length and visible_length + token_length > max_length:
            finish_chunk()
            html_parts = [opening for _, opening in open_tags]
            plain_parts = []
            visible_length = 0
        html_parts.append(token if not escaped else html.escape(token, quote=False))
        plain_parts.append(visible)
        visible_length += token_length

    for token in _HTML_TOKEN.findall(rendered):
        tag_match = _HTML_TAG.fullmatch(token)
        if tag_match:
            tag = tag_match.group("name")
            closing = bool(tag_match.group("closing"))
            attrs = tag_match.group("attrs")
            allowed = (tag in _SIMPLE_TAGS or tag == "a") and not attrs if closing else (
                tag in _SIMPLE_TAGS and not attrs
            )
            if tag == "a" and not closing:
                href_match = _ANCHOR_HREF.fullmatch(attrs)
                allowed = bool(href_match and _safe_link(html.unescape(href_match.group(1))))

            if allowed and closing:
                matching_index = next(
                    (idx for idx in range(len(open_tags) - 1, -1, -1)
                     if open_tags[idx][0] == tag),
                    None,
                )
                if matching_index is None:
                    # Keep unmatched closing tags as escaped literal content.
                    append_visible(token, escaped=True)
                    continue
                while len(open_tags) - 1 > matching_index:
                    nested_tag, _ = open_tags.pop()
                    html_parts.append(f"</{nested_tag}>")
                html_parts.append(token)
                open_tags.pop()
                continue

            if allowed:
                html_parts.append(token)
                open_tags.append((tag, token))
                continue

            # Unknown tags and unsupported attributes are literal text.
            for char in token:
                append_visible(char, escaped=True)
            continue

        if token.startswith("&") and token.endswith(";"):
            decoded = _decode_telegram_entity(token) if _HTML_ENTITY.fullmatch(token) else None
            if decoded is not None:
                append_visible(token, decoded=decoded)
            else:
                # Unknown entity names are ordinary text, escaped as an ampersand.
                for char in token:
                    append_visible(char, escaped=True)
            continue

        if token.startswith("<") and token.endswith(">"):
            for char in token:
                append_visible(char, escaped=True)
        else:
            append_visible(token, escaped=True)

    if plain_parts:
        finish_chunk()
    return chunks

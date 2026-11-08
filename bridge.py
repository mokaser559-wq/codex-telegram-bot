import hmac
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Condition

from assistant.core import process_message
from assistant.providers import (
    NoProviderConfigured,
    ProviderAuthenticationError,
    ProviderError,
    ProviderUnavailable,
    configured_provider_names,
)
from assistant.router import route_message
from assistant.stt import STTConfigurationError, STTUnavailable, STTUnsupported, transcribe_audio
from assistant.storage import (
    PostgresStore,
    StorageConfigurationError,
    StorageUnavailable,
    HistoryImportError,
    retry_delay,
)
from assistant.telegram_format import markdown_to_telegram_html, split_telegram_html

TOKEN_FILE = Path(os.environ.get(
    "BOT_TOKEN_FILE",
    str(Path.home() / ".local/share/codex-telegram/bot_token"),
))
ALLOWED_USER_ID = 1193680242
_LOCAL_CODEX = "/home/rozzen/.local/bin/codex"
CODEX = _LOCAL_CODEX if os.path.exists(_LOCAL_CODEX) else (shutil.which("codex") or "codex")
SANDBOX = os.environ.get("CODEX_SANDBOX", "workspace-write")
API = None
TIMEOUT = 300
MEMORY_LIMIT = 10
GET_UPDATES_TIMEOUT = 5
GET_UPDATES_HTTP_TIMEOUT = 8
WEBHOOK_MAX_BODY_BYTES = 1_048_576
WEBHOOK_READ_TIMEOUT_SECONDS = 10
HTTP_MAX_CONCURRENT_REQUESTS = 16
HTTP_SHUTDOWN_WAIT_SECONDS = 12
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("klodd")


def persist_webhook_request(store, provided_secret, webhook_secret, content_length, raw_body):
    """Validate and durably persist one HTTP webhook request."""
    if not hmac.compare_digest(provided_secret.encode("utf-8"), webhook_secret.encode("utf-8")):
        return 403
    if content_length < 0 or content_length > WEBHOOK_MAX_BODY_BYTES:
        return 413 if content_length > WEBHOOK_MAX_BODY_BYTES else 400
    if len(raw_body) != content_length:
        return 400
    try:
        update = json.loads(raw_body)
        if not isinstance(update, dict):
            return 400
        update_id = update.get("update_id")
        if isinstance(update_id, bool) or not isinstance(update_id, int) or not 0 <= update_id <= 9_223_372_036_854_775_807:
            return 400
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return 400
    try:
        store.persist_webhook_update(update)
    except (StorageUnavailable, StorageConfigurationError):
        return 503
    return 200


def _secret_matches(provided_secret, webhook_secret):
    return hmac.compare_digest(provided_secret.encode("utf-8"), webhook_secret.encode("utf-8"))

SYSTEM_PROMPT = (
    "Ты — дружелюбный личный помощник пользователя. Общайся естественно, тепло и просто, как хороший приятель: без канцелярита, шаблонных фраз и лишней официальности. На обычные сообщения отвечай по-человечески. В технических задачах будь компетентным, объясняй понятным языком и выполняй просьбу по существу. Не утверждай, что выполнил действие, если не выполнил его. Отвечай на языке пользователя. Если вопрос требует актуальных сведений, поиска фактов, новостей, сравнения вариантов или проверки информации, обязательно используй инструмент Tavily MCP для поиска в интернете. Не ограничивайся собственными знаниями. Основывай ответ на найденных источниках, добавляй прямые ссылки и кратко объясняй вывод. Обычно достаточно одного поискового вызова и 3 наиболее релевантных результатов; повторяй поиск только если результатов недостаточно. Не используй поиск для обычной беседы и задач, где он не нужен. Отвечай кратко, без длинных вступлений и ненужных рассуждений. Не выдумывай результаты поиска и ссылки."
)


class Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def start_http_server(store, webhook_secret=None, webhook_path="/telegram/webhook"):
    port = os.environ.get("PORT")
    if not port:
        if webhook_secret:
            raise StorageConfigurationError("PORT обязателен в webhook-режиме")
        return None
    mode = "webhook" if webhook_secret else "polling"
    class Health(Ping):
        def setup(self):
            super().setup()
            self.connection.settimeout(WEBHOOK_READ_TIMEOUT_SECONDS)

        def do_POST(self):
            if not webhook_secret or self.path != webhook_path:
                self._respond(404, b"not found")
                return
            provided_secret = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if not _secret_matches(provided_secret, webhook_secret):
                self._respond(403, b"forbidden")
                return
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length)
            except (TypeError, ValueError):
                self._respond(400, b"invalid content length")
                return
            if length < 0 or length > WEBHOOK_MAX_BODY_BYTES:
                status = 413 if length > WEBHOOK_MAX_BODY_BYTES else 400
                self._respond(status, b"invalid content length")
                return
            try:
                raw = self._read_body_with_deadline(length)
            except (OSError, TimeoutError):
                self._respond(400, b"invalid body")
                return
            status = persist_webhook_request(
                store,
                provided_secret,
                webhook_secret,
                length,
                raw,
            )
            self._respond(status, b"ok" if status == 200 else b"invalid or unavailable")

        def _read_body_with_deadline(self, length):
            deadline = time.monotonic() + WEBHOOK_READ_TIMEOUT_SECONDS
            chunks = []
            remaining = length
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise TimeoutError("request body deadline exceeded")
                self.connection.settimeout(timeout)
                chunk = self.rfile.read1(min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def _respond(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if mode == "polling":
                return Ping.do_GET(self)
            try:
                result = store.health(mode=mode)
            except (StorageUnavailable, StorageConfigurationError):
                result = {"ok": False, "database": False, "polling": False}
            body = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(200 if result.get("ok") else 503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_HEAD = do_GET

    class BoundedHTTPServer(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

        def __init__(self, address, handler):
            self._request_slots = BoundedSemaphore(HTTP_MAX_CONCURRENT_REQUESTS)
            self._active_condition = Condition()
            self._active_requests = 0
            super().__init__(address, handler)

        def process_request(self, request, client_address):
            if not self._request_slots.acquire(blocking=False):
                try:
                    request.settimeout(1)
                    request.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\n"
                        b"Connection: close\r\nContent-Length: 0\r\n\r\n"
                    )
                except OSError:
                    pass
                finally:
                    self.shutdown_request(request)
                return
            with self._active_condition:
                self._active_requests += 1
            try:
                super().process_request(request, client_address)
            except Exception:
                with self._active_condition:
                    self._active_requests -= 1
                    self._active_condition.notify_all()
                self._request_slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                with self._active_condition:
                    self._active_requests -= 1
                    self._active_condition.notify_all()
                self._request_slots.release()

        def wait_for_requests(self, timeout=None):
            if timeout is None:
                timeout = HTTP_SHUTDOWN_WAIT_SECONDS
            with self._active_condition:
                return self._active_condition.wait_for(lambda: self._active_requests == 0, timeout)

    server = BoundedHTTPServer(("0.0.0.0", int(port)), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    LOG.info("HTTP endpoint started")
    return server


def shutdown_http_server(server):
    server.shutdown()
    server.wait_for_requests()
    server.server_close()


class TelegramTransientError(Exception):
    def __init__(self, code: int | None = None, retry_after: int | None = None):
        super().__init__("telegram_transient")
        self.code = code
        self.retry_after = retry_after


class TelegramPermanentError(Exception):
    def __init__(self, code: int | None = None):
        super().__init__("telegram_permanent")
        self.code = code


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _http_retry_after(error):
    headers = getattr(error, "headers", None)
    retry_after = _positive_int(headers.get("Retry-After")) if headers else None
    if retry_after is not None:
        return retry_after
    try:
        body = json.loads(error.read())
        parameters = body.get("parameters") or {}
        return _positive_int(parameters.get("retry_after"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def telegram(method, data=None, timeout=40):
    body = urllib.parse.urlencode(data or {}).encode()
    req = urllib.request.Request(API + "/" + method, data=body)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise TelegramTransientError(exc.code, _http_retry_after(exc)) from None
        if exc.code in (408, 425) or exc.code >= 500:
            raise TelegramTransientError(exc.code) from None
        raise TelegramPermanentError(exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise TelegramTransientError() from None
    if not isinstance(result, dict):
        raise TelegramTransientError()
    if not result.get("ok"):
        code = result.get("error_code")
        if code == 429 or code in (408, 425) or (isinstance(code, int) and code >= 500):
            parameters = result.get("parameters") or {}
            raise TelegramTransientError(code, _positive_int(parameters.get("retry_after")))
        raise TelegramPermanentError(code if isinstance(code, int) else None)
    return result["result"]


MAX_VOICE_BYTES = 20 * 1024 * 1024
MAX_VOICE_DURATION_SECONDS = 300


def _download_voice_file(file_id, destination, declared_size=None):
    if declared_size is not None and (not isinstance(declared_size, int) or declared_size < 0 or declared_size > MAX_VOICE_BYTES):
        raise STTUnsupported("voice_file_too_large")
    try:
        file_info = telegram("getFile", {"file_id": file_id}, timeout=20)
    except TelegramTransientError:
        raise STTUnavailable("telegram_get_file_unavailable") from None
    except TelegramPermanentError:
        raise STTUnsupported("telegram_file_unavailable") from None
    file_path = file_info.get("file_path") if isinstance(file_info, dict) else None
    remote_size = file_info.get("file_size") if isinstance(file_info, dict) else None
    if not isinstance(file_path, str) or not file_path or ".." in Path(file_path).parts:
        raise STTUnsupported("invalid_telegram_file_path")
    if isinstance(remote_size, int) and remote_size > MAX_VOICE_BYTES:
        raise STTUnsupported("voice_file_too_large")

    file_api = API.replace("https://api.telegram.org/bot", "https://api.telegram.org/file/bot", 1)
    url = file_api + "/" + urllib.parse.quote(file_path, safe="/")
    request = urllib.request.Request(url)
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=30) as response, open(destination, "wb") as output:
            length = _positive_int(response.headers.get("Content-Length"))
            if length is not None and length > MAX_VOICE_BYTES:
                raise STTUnsupported("voice_file_too_large")
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_VOICE_BYTES:
                    raise STTUnsupported("voice_file_too_large")
                output.write(chunk)
    except STTUnsupported:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 429 or exc.code in (408, 425) or exc.code >= 500:
            raise STTUnavailable("telegram_download_unavailable") from None
        raise STTUnsupported("telegram_download_rejected") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise STTUnavailable("telegram_download_unavailable") from None
    if total == 0:
        raise STTUnsupported("empty_voice_file")


def _voice_transcription(store, item, message, chat_id):
    internal = item.payload.get("_klodd") or {}
    cached = internal.get("transcription") if isinstance(internal, dict) else None
    if isinstance(cached, str) and cached.strip():
        return cached.strip()

    voice = message.get("voice")
    if not isinstance(voice, dict):
        raise STTUnsupported("voice_metadata_missing")
    duration = voice.get("duration")
    if isinstance(duration, int) and duration > MAX_VOICE_DURATION_SECONDS:
        raise STTUnsupported("voice_too_long")
    declared_size = voice.get("file_size")
    if isinstance(declared_size, int) and declared_size > MAX_VOICE_BYTES:
        raise STTUnsupported("voice_file_too_large")
    file_id = voice.get("file_id")
    if not isinstance(file_id, str) or not file_id:
        raise STTUnsupported("voice_file_id_missing")

    with tempfile.TemporaryDirectory(prefix="klodd-voice-") as directory:
        audio_path = Path(directory) / "voice.ogg"
        _download_voice_file(file_id, audio_path, voice.get("file_size"))
        transcript = transcribe_audio(audio_path)
    if not store.save_transcription(
        item.update_id, chat_id, transcript, item.claim_token, item.generation,
    ):
        return None
    return transcript


def send(chat_id, text):
    text = text.strip() or "(Codex не вернул текстовый ответ)"
    rendered = markdown_to_telegram_html(text)
    for chunk, plain_text in split_telegram_html(rendered):
        data = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
        }
        try:
            telegram("sendMessage", data)
        except TelegramPermanentError as exc:
            if exc.code != 400:
                raise
            # Preserve visible content if Telegram rejects an entity/tag.
            telegram("sendMessage", {"chat_id": chat_id, "text": plain_text})


def ask_codex(prompt, history=None):
    history = history or []
    context = ""
    if history:
        context = "\n\nНедавняя история диалога (от старых сообщений к новым):\n" + "\n".join(
            ("Пользователь: " if item["role"] == "user" else "Ты: ") + item["text"]
            for item in history
        ) + "\n\nПродолжай разговор с учётом этого контекста.\n"
    prompt = context + "\nНовое сообщение пользователя:\n" + prompt
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as output:
        result = subprocess.run(
            [
                CODEX, "exec",
                "--skip-git-repo-check",
                "--sandbox", SANDBOX,
                "--ephemeral",
                "--output-last-message", output.name,
                SYSTEM_PROMPT + "\n\nЗадание пользователя:\n" + prompt
            ],
            cwd=str(Path.home()),
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        output.seek(0)
        answer = output.read().strip()
        if result.returncode != 0:
            # stdout/stderr can include prompts, paths, environment details, or
            # credentials emitted by a subprocess. Keep diagnostics to metadata.
            LOG.error("Codex execution failed (returncode=%d)", result.returncode)
            return "Не удалось выполнить техническую задачу. Попробуй ещё раз позже."
        return answer or "Codex не вернул ответ. Попробуй ещё раз."


def main():
    global API
    if not TOKEN_FILE.is_file():
        raise SystemExit("Не найден файл bot_token")
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("Файл bot_token пуст")
    API = "https://api.telegram.org/bot" + token
    os.chmod(TOKEN_FILE, 0o600)
    store = PostgresStore(allowed_user_id=ALLOWED_USER_ID)
    migrations_path = Path(__file__).parent / "migrations"
    stop = threading.Event()
    fatal = threading.Event()
    _install_shutdown_handlers(stop)
    lock_lost = None
    mode = os.environ.get("TELEGRAM_MODE", "polling").strip().lower()
    if mode not in ("polling", "webhook"):
        raise StorageConfigurationError("TELEGRAM_MODE должен быть polling или webhook")
    webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
    webhook_path = os.environ.get("TELEGRAM_WEBHOOK_PATH", "/telegram/webhook")
    if mode == "webhook" and not webhook_secret:
        raise StorageConfigurationError("TELEGRAM_WEBHOOK_SECRET обязателен в webhook-режиме")
    if not webhook_path.startswith("/"):
        raise StorageConfigurationError("TELEGRAM_WEBHOOK_PATH должен начинаться с /")

    def initialize_runtime():
        store.initialize(migrations_path)
        store.recover_stale()
        store.begin_runtime()
        try:
            import_result = store.import_legacy_history(
                os.environ.get(
                    "LEGACY_MEMORY_FILE",
                    str(Path.home() / ".local/share/codex-telegram/chat_memory.json"),
                )
            )
            if import_result not in ("absent", "already_imported"):
                LOG.info("Legacy history import completed")
        except HistoryImportError as exc:
            LOG.error("Legacy history import failed (%s); source file kept", type(exc).__name__)
        LOG.info("Storage initialized; AI providers configured: %s", ",".join(configured_provider_names()) or "none")

    def run_worker_runtime(server=None, poller=None):
        worker = threading.Thread(target=worker_loop, args=(store, stop, fatal, mode), name="message-worker")
        worker.start()
        LOG.info("%s and message worker started", mode.capitalize())
        try:
            while worker.is_alive() and (poller is None or poller.is_alive()):
                if stop.wait(1):
                    break
        finally:
            stop.set()
            if server is not None:
                shutdown_http_server(server)
            if poller is not None:
                poller.join()
            worker.join()

    try:
        if mode == "polling":
            server = start_http_server(store)
            with store.polling_lock(stop) as lock_lost:
                initialize_runtime()
                poller = threading.Thread(target=poll_loop, args=(store, stop, fatal), name="telegram-poller")
                poller.start()
                run_worker_runtime(server=server, poller=poller)
        else:
            initialize_runtime()
            server = start_http_server(store, webhook_secret, webhook_path)
            run_worker_runtime(server=server)
    except KeyboardInterrupt:
        LOG.info("Shutdown requested")
    finally:
        stop.set()
    if fatal.is_set() or (lock_lost is not None and lock_lost.is_set()):
        if lock_lost is not None and lock_lost.is_set() and not fatal.is_set():
            LOG.critical("Polling stopped because the PostgreSQL advisory-lock connection was lost")
        raise SystemExit(1)


def _install_shutdown_handlers(stop):
    def request_shutdown(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)


def _stop_and_join_runtime(stop, poller, worker):
    stop.set()
    # Keep the advisory lock until both threads have stopped. The HTTP calls
    # and Codex have their own timeouts; an active worker must not outlive it.
    poller.join()
    worker.join()


def poll_loop(store, stop, fatal=None):
    failures = 0
    while not stop.is_set():
        try:
            if not store.verify_polling_lock():
                LOG.critical("Polling stopped because advisory-lock ownership could not be confirmed")
                if fatal is not None:
                    fatal.set()
                stop.set()
                break
            params = {"timeout": GET_UPDATES_TIMEOUT}
            offset = store.offset()
            if offset is not None:
                params["offset"] = offset
            if stop.is_set():
                break
            if not store.verify_polling_lock():
                if fatal is not None:
                    fatal.set()
                stop.set()
                break
            updates = telegram("getUpdates", params, timeout=GET_UPDATES_HTTP_TIMEOUT)
            # Recheck after the network wait: a response received after lock
            # loss must not be persisted or used to advance the offset.
            if stop.is_set():
                break
            if not store.verify_polling_lock():
                if fatal is not None:
                    fatal.set()
                stop.set()
                break
            if stop.is_set():
                break
            store.persist_updates(updates)
            store.record_poll(True)
            failures = 0
        except (StorageUnavailable, TelegramTransientError) as exc:
            failures += 1
            LOG.warning("Polling temporarily unavailable (%s, code=%s)", type(exc).__name__, getattr(exc, "code", None))
            try:
                store.record_poll(False, type(exc).__name__)
            except StorageUnavailable:
                pass
            except StorageConfigurationError as db_exc:
                LOG.critical("Polling stopped on permanent database error (%s)", str(db_exc))
                if fatal is not None:
                    fatal.set()
                stop.set()
            delay = getattr(exc, "retry_after", None) or retry_delay(failures)
            stop.wait(delay)
        except TelegramPermanentError as exc:
            LOG.error("Polling stopped on permanent Telegram error (code=%s)", exc.code)
            try:
                store.record_poll(False, "telegram_permanent_" + str(exc.code or "unknown"))
            except (StorageUnavailable, StorageConfigurationError):
                pass
            if fatal is not None:
                fatal.set()
            stop.set()
        except StorageConfigurationError as exc:
            LOG.critical("Polling stopped on permanent database error (%s)", str(exc))
            if fatal is not None:
                fatal.set()
            stop.set()
        except Exception as exc:
            failures += 1
            LOG.error("Polling failed (%s)", type(exc).__name__)
            stop.wait(retry_delay(failures))


def _status_text(store, mode="polling"):
    try:
        state = store.health(mode=mode)
    except (StorageUnavailable, StorageConfigurationError):
        return "База состояния временно недоступна."
    lines = ["База данных: доступна."]
    if mode == "polling":
        lines.append(f"Polling: {'работает' if state['polling'] else 'нет свежего успешного запроса'}.")
    else:
        lines.append("Режим: webhook.")
    lines.extend((
        f"Worker: {'работает' if state['worker'] else 'нет свежего прогресса'}.",
        f"Зависшие очереди: inbox={'да' if state['inbox_stuck'] else 'нет'}, outbox={'да' if state['outbox_stuck'] else 'нет'}.",
        f"В очереди: {state['inbox_pending']} сообщений, {state['outbox_pending']} ответов.",
        f"На разборе: {state['inbox_dead_letter']} сообщений, {state['outbox_dead_letter']} ответов.",
    ))
    return "\n".join(lines)


def worker_loop(store, stop, fatal=None, mode="polling"):
    heartbeat_at = 0.0
    recovery_at = 0.0
    while not stop.is_set():
        try:
            now = time.monotonic()
            if now - heartbeat_at >= 15:
                store.record_worker_heartbeat()
                heartbeat_at = now
            if now - recovery_at >= 15:
                store.recover_stale()
                recovery_at = now
            item = store.claim_inbox()
            if item is not None:
                process_inbox_item(store, item, mode=mode)
                continue
            outgoing = store.claim_outbox()
            if outgoing is not None:
                try:
                    deliver_outbox_item(store, outgoing)
                except (TelegramTransientError, TelegramPermanentError) as exc:
                    store.retry_outbox(
                        outgoing.id,
                        outgoing.attempts,
                        "telegram_" + str(exc.code or "network"),
                        outgoing.claim_token,
                        retry_after=getattr(exc, "retry_after", None),
                        permanent=isinstance(exc, TelegramPermanentError),
                    )
                    LOG.warning("Outbox delivery failed (%s)", type(exc).__name__)
                except StorageUnavailable:
                    LOG.warning("Outbox state could not be saved; stale-send recovery will retry it")
                continue
        except StorageUnavailable as exc:
            LOG.warning("Worker storage unavailable (%s)", type(exc).__name__)
        except StorageConfigurationError as exc:
            LOG.critical("Worker stopped on permanent database error (%s)", str(exc))
            if fatal is not None:
                fatal.set()
            stop.set()
        except Exception as exc:
            LOG.error("Worker failed (%s)", type(exc).__name__)
        stop.wait(1)


def deliver_outbox_item(store, outgoing):
    """Send only while PostgreSQL fences this claim against stale recovery."""
    result = store.deliver_outbox(outgoing, send)
    if not result:
        LOG.info("Skipping superseded outbox item %s", outgoing.id)
    return result


def process_inbox_item(store, item, mode="polling"):
    update = item.payload
    message = update.get("message") or {}
    sender = message.get("from") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if sender.get("id") != ALLOWED_USER_ID or not chat_id:
        store.ignore_inbox(item.update_id, item.claim_token)
        return

    text = message.get("text")
    is_voice = isinstance(message.get("voice"), dict)
    if not (isinstance(text, str) and text.strip()) and not is_voice:
        store.ignore_inbox(item.update_id, item.claim_token)
        return

    try:
        if is_voice and not (isinstance(text, str) and text.strip()):
            text = _voice_transcription(store, item, message, chat_id)
            if text is None:
                return
        text = text.strip()
        route = route_message(text)
        history = store.history(chat_id, item.update_id, MEMORY_LIMIT)

        def requeue_handler(kind, item_id):
            changed = store.requeue_dead_letter(
                kind, item_id, chat_id,
                command_update_id=item.update_id, claim_token=item.claim_token,
            )
            return (
                "Запись возвращена в очередь."
                if changed
                else "Dead-letter запись с таким ID в этом чате не найдена."
            )

        answer = process_message(
            text,
            history,
            codex_handler=ask_codex,
            status_handler=lambda: _status_text(store, mode),
            requeue_handler=requeue_handler,
        )
        store.complete_inbox(item.update_id, chat_id, route, answer, item.claim_token, item.generation)
    except STTUnsupported:
        store.permanent_inbox_error(
            item.update_id, chat_id,
            "Не удалось распознать голосовое сообщение. Попробуй отправить более короткую запись в формате голосового сообщения Telegram.",
            "voice_unsupported", item.claim_token, item.generation,
        )
    except STTConfigurationError:
        store.permanent_inbox_error(
            item.update_id, chat_id,
            "Локальное распознавание голоса сейчас не настроено. Сообщение сохранено.",
            "stt_not_configured", item.claim_token, item.generation,
        )
    except STTUnavailable as exc:
        store.retry_inbox(
            item.update_id, item.attempts, type(exc).__name__, item.claim_token,
            "Не удалось распознать голос после повторных попыток. Сообщение сохранено для разбора.",
            chat_id, item.generation,
        )
        LOG.warning("Local transcription temporarily unavailable for update %s", item.update_id)
    except NoProviderConfigured:
        store.permanent_inbox_error(
            item.update_id, chat_id,
            "AI-провайдер не настроен. Сообщение сохранено; сообщите администратору.",
            "no_provider_configured", item.claim_token, item.generation,
        )
    except ProviderAuthenticationError:
        store.permanent_inbox_error(
            item.update_id, chat_id,
            "Не удалось авторизоваться у AI-провайдера. Сообщение сохранено; проверьте ключ в настройках сервиса.",
            "provider_authentication", item.claim_token, item.generation,
        )
    except ProviderUnavailable as exc:
        terminal = "Провайдеры не восстановились после повторных попыток. Сообщение сохранено для разбора."
        store.retry_inbox(
            item.update_id, item.attempts, type(exc).__name__, item.claim_token,
            terminal, chat_id, item.generation,
        )
        LOG.warning("Provider temporarily unavailable for update %s", item.update_id)
    except ProviderError as exc:
        store.permanent_inbox_error(
            item.update_id, chat_id,
            "AI-провайдер не смог обработать сообщение. Сообщение сохранено для разбора.",
            type(exc).__name__, item.claim_token, item.generation,
        )
    except subprocess.TimeoutExpired:
        store.complete_inbox(
            item.update_id, chat_id, route,
            "Codex не завершил задание за 5 минут. Попробуй более короткий запрос.",
            item.claim_token, item.generation,
        )
    except Exception as exc:
        terminal = "Не удалось обработать сообщение после повторных попыток. Оно сохранено для разбора."
        store.retry_inbox(
            item.update_id, item.attempts, type(exc).__name__, item.claim_token,
            terminal, chat_id, item.generation,
        )
        LOG.error("Update %s processing failed (%s)", item.update_id, type(exc).__name__)


if __name__ == "__main__":
    try:
        main()
    except (StorageConfigurationError, StorageUnavailable) as exc:
        LOG.critical("Startup/storage failure (%s)", str(exc))
        raise SystemExit(1) from None

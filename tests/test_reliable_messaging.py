import copy
import http.client
import io
import json
import socket
import threading
import time
import urllib.error
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import assistant.providers as providers
import assistant.core as core
import assistant.storage as storage
import bridge
from assistant.storage import (
    MAX_ATTEMPTS,
    MAX_RETRY_SECONDS,
    PostgresStore,
    StorageUnavailable,
    retry_delay,
)


def test_retry_delay_is_exponential_and_capped():
    assert retry_delay(1, jitter=False) == 2
    assert retry_delay(2, jitter=False) == 4
    assert retry_delay(50, jitter=False) == MAX_RETRY_SECONDS


def test_provider_diagnostics_only_return_names(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  mock-credential  ")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = providers.configured_provider_names()
    assert result == ("Groq",)
    assert "mock-credential" not in repr(result)


def test_provider_whitespace_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(providers.NoProviderConfigured):
        providers.generate_answer("hello")


def test_core_preserves_codex_and_regular_ai_routes(monkeypatch):
    codex_calls = []
    ai_calls = []
    monkeypatch.setattr(core, "generate_answer", lambda text, history: ai_calls.append(text) or "AI")
    codex_answer = core.process_message(
        "/codex исправь код",
        codex_handler=lambda prompt, history: codex_calls.append(prompt) or "Codex",
    )
    ai_answer = core.process_message("привет", history=[])
    assert codex_answer == "Codex"
    assert codex_calls == ["исправь код"]
    assert ai_answer == "AI"
    assert ai_calls == ["привет"]


def test_identity_and_model_questions_use_primary_ai(monkeypatch):
    calls = []
    codex_calls = []
    monkeypatch.setattr(core, "generate_answer", lambda text, history: calls.append(text) or "AI")
    prompt = "Привет, Клодд! Скажи, что ты работаешь через AI и назови свою модель"
    assert core.process_message(prompt, codex_handler=lambda *args: codex_calls.append(args)) == "AI"
    assert calls == [prompt]
    assert codex_calls == []


def test_model_context_uses_configured_provider_and_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("GROQ_MODEL", "vendor/model-live")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    assert providers.generate_answer("Как называется твоя модель?") == "ok"
    assert captured["model"] == "vendor/model-live"
    assert "Groq" in captured["messages"][0]["content"]
    assert "vendor/model-live" in captured["messages"][0]["content"]


def test_exact_identity_prompt_is_not_routed_to_codex():
    assert bridge.route_message(
        "Привет, Клодд! Скажи, что ты работаешь через AI и назови свою модель"
    ) == "chat"


def test_technical_prompt_is_routed_to_codex():
    assert bridge.route_message("исправь ошибку в файле bridge.py") == "codex"


def test_codex_failure_does_not_expose_stdout_or_stderr(monkeypatch, caplog):
    private_output = "PRIVATE PATH /srv/private/config and SECRET_VALUE"
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=private_output, stderr=private_output),
    )

    answer = bridge.ask_codex("исправь ошибку")

    assert answer == "Не удалось выполнить техническую задачу. Попробуй ещё раз позже."
    assert private_output not in answer
    assert private_output not in caplog.text
    assert "returncode=1" in caplog.text


def test_temporary_provider_error_is_retried_without_marking_complete(monkeypatch):
    monkeypatch.setattr(bridge, "process_message", lambda *args, **kwargs: (_ for _ in ()).throw(providers.ProviderUnavailable("safe")))
    store = FakeStore()
    bridge.process_inbox_item(store, InboxItemForTest({
        "update_id": 11,
        "message": {"from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 22}, "text": "hello"},
    }))
    assert store.retries[0][0] == 11
    assert not store.completed
    assert not store.permanent


def test_missing_provider_is_terminal_and_queued_for_delivery(monkeypatch):
    monkeypatch.setattr(bridge, "process_message", lambda *args, **kwargs: (_ for _ in ()).throw(providers.NoProviderConfigured("safe")))
    store = FakeStore()
    bridge.process_inbox_item(store, InboxItemForTest({
        "update_id": 12,
        "message": {"from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 22}, "text": "hello"},
    }))
    assert store.permanent[0][0] == 12
    assert "secret" not in repr(store.permanent)


def test_telegram_error_does_not_echo_token_from_request_url(monkeypatch):
    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fail)
    monkeypatch.setattr(bridge, "API", "https://api.telegram.org/botTOKEN-MUST-NOT-LEAK")
    with pytest.raises(bridge.TelegramTransientError) as error:
        bridge.telegram("getUpdates")
    assert "TOKEN-MUST-NOT-LEAK" not in str(error.value)


def test_telegram_429_uses_retry_after_from_api_body(monkeypatch):
    monkeypatch.setattr(bridge, "API", "https://api.telegram.org/botFAKE")
    monkeypatch.setattr(
        bridge.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            b'{"ok":false,"error_code":429,"parameters":{"retry_after":23}}'
        ),
    )
    with pytest.raises(bridge.TelegramTransientError) as error:
        bridge.telegram("getUpdates")
    assert error.value.code == 429
    assert error.value.retry_after == 23


def test_telegram_http_429_uses_retry_after_from_error_body(monkeypatch):
    monkeypatch.setattr(bridge, "API", "https://api.telegram.org/botFAKE")

    def fail(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 429, "rate limited", {},
            io.BytesIO(b'{"parameters":{"retry_after":17}}'),
        )

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fail)
    with pytest.raises(bridge.TelegramTransientError) as error:
        bridge.telegram("getUpdates")
    assert error.value.retry_after == 17


def test_telegram_409_is_permanent_and_diagnostic(monkeypatch):
    monkeypatch.setattr(bridge, "API", "https://api.telegram.org/botFAKE")
    monkeypatch.setattr(
        bridge.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b'{"ok":false,"error_code":409}'),
    )
    with pytest.raises(bridge.TelegramPermanentError) as error:
        bridge.telegram("getUpdates")
    assert error.value.code == 409


def test_database_errors_are_split_by_sqlstate_without_details():
    class DatabaseException(Exception):
        def __init__(self, sqlstate, message):
            super().__init__(message)
            self.sqlstate = sqlstate

    with pytest.raises(StorageUnavailable) as transient:
        PostgresStore._raise_db_error(DatabaseException("08006", "secret connection string"))
    with pytest.raises(storage.StorageConfigurationError) as permanent:
        PostgresStore._raise_db_error(DatabaseException("42501", "secret connection string"))
    assert "secret connection string" not in str(transient.value)
    assert "secret connection string" not in str(permanent.value)


def test_lock_watchdog_stops_polling_when_session_dies(monkeypatch):
    class LockCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params=None):
            if "pg_try_advisory_lock" in query:
                self.result = (True,)
            elif "SELECT pg_backend_pid(), backend_start" in query:
                self.result = (123, "session-start")
            elif "FROM pg_locks" in query:
                raise OSError("simulated lost lock connection")

        def fetchone(self):
            return self.result

    class LockConnection:
        def cursor(self):
            return LockCursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(storage, "LOCK_CHECK_SECONDS", 0.01)
    store = PostgresStore(connect=LockConnection)
    stop = threading.Event()
    with store.polling_lock(stop) as lost:
        assert stop.wait(0.5)
        assert lost.is_set()


def test_second_polling_instance_is_rejected():
    class BusyCursor:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, query, params=None): self.result = (False,)
        def fetchone(self): return self.result

    class BusyConnection:
        def cursor(self): return BusyCursor()
        def commit(self): pass
        def close(self): pass

    store = PostgresStore(connect=BusyConnection)
    with pytest.raises(storage.StorageConfigurationError, match="Другой polling"):
        with store.polling_lock(threading.Event()):
            pytest.fail("second polling instance must not enter the lock")


def test_polling_fails_closed_before_get_updates_if_lock_is_unconfirmed(monkeypatch):
    store = FakeStore()
    store.verify_polling_lock = lambda: False
    stop = threading.Event()
    fatal = threading.Event()
    telegram_calls = []
    monkeypatch.setattr(bridge, "telegram", lambda *args, **kwargs: telegram_calls.append(args))
    bridge.poll_loop(store, stop, fatal)
    assert telegram_calls == []
    assert stop.is_set()
    assert fatal.is_set()


def test_polling_discards_get_updates_response_when_lock_is_lost(monkeypatch):
    class PollStore:
        def __init__(self):
            self.checks = 0
            self.persisted = []

        def verify_polling_lock(self):
            self.checks += 1
            return self.checks < 3

        def offset(self):
            return None

        def persist_updates(self, updates):
            self.persisted.append(updates)

    store = PollStore()
    stop = threading.Event()
    fatal = threading.Event()
    monkeypatch.setattr(bridge, "telegram", lambda *args, **kwargs: [{"update_id": 1}])

    bridge.poll_loop(store, stop, fatal)

    assert store.checks == 3
    assert store.persisted == []
    assert stop.is_set() and fatal.is_set()


def test_polling_rechecks_lock_after_loading_offset(monkeypatch):
    class PollStore:
        def __init__(self):
            self.checks = 0

        def verify_polling_lock(self):
            self.checks += 1
            return self.checks == 1

        def offset(self):
            return None

    store = PollStore()
    stop = threading.Event()
    fatal = threading.Event()
    telegram_calls = []
    monkeypatch.setattr(bridge, "telegram", lambda *args, **kwargs: telegram_calls.append(args))

    bridge.poll_loop(store, stop, fatal)

    assert store.checks == 2
    assert telegram_calls == []
    assert stop.is_set() and fatal.is_set()


def test_shutdown_signals_request_stop(monkeypatch):
    handlers = {}
    monkeypatch.setattr(bridge.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))
    stop = threading.Event()

    bridge._install_shutdown_handlers(stop)
    handlers[bridge.signal.SIGTERM](bridge.signal.SIGTERM, None)
    assert stop.is_set()
    stop.clear()
    handlers[bridge.signal.SIGINT](bridge.signal.SIGINT, None)
    assert stop.is_set()


def test_shutdown_waits_for_worker_before_returning():
    stop = threading.Event()
    worker_started = threading.Event()
    release_worker = threading.Event()

    def active_worker():
        worker_started.set()
        release_worker.wait()

    poller = threading.Thread(target=lambda: None)
    worker = threading.Thread(target=active_worker)
    poller.start()
    worker.start()
    assert worker_started.wait(1)

    joined = threading.Event()
    shutdown = threading.Thread(
        target=lambda: (bridge._stop_and_join_runtime(stop, poller, worker), joined.set())
    )
    shutdown.start()
    assert stop.wait(1)
    assert not joined.wait(0.05)
    release_worker.set()
    assert joined.wait(1)
    shutdown.join()
    assert not poller.is_alive() and not worker.is_alive()


class FakeCursor:
    def __init__(self, database):
        self.database = database
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        self.database.statements.append(statement)
        if self.database.fail_on_insert and statement.startswith("INSERT INTO inbox"):
            raise OSError("simulated database outage")
        if ("RETURNING id" in statement or "RETURNING update_id" in statement) and not statement.startswith("INSERT INTO inbox"):
            self.result = (params[0],)
        if statement.startswith("INSERT INTO inbox"):
            update_id, payload = params
            existed = update_id in self.database.inbox
            self.database.inbox.setdefault(update_id, json.loads(payload))
            if "RETURNING update_id" in statement and not existed:
                self.result = (update_id,)
        elif statement.startswith("UPDATE bot_state SET next_offset"):
            self.database.offset = max(self.database.offset or 0, params[0])

    def fetchone(self):
        return self.result


class FakeConnection:
    def __init__(self, database):
        self.database = database

    def __enter__(self):
        self.snapshot = copy.deepcopy((self.database.inbox, self.database.offset))
        return self

    def __exit__(self, exc_type, *_):
        if exc_type:
            self.database.inbox, self.database.offset = self.snapshot
        return False

    def cursor(self):
        return FakeCursor(self.database)


class FakeDatabase:
    def __init__(self):
        self.inbox = {}
        self.offset = None
        self.fail_on_insert = False
        self.statements = []

    def connect(self):
        return FakeConnection(self)


def test_offset_advances_only_with_committed_inbox():
    db = FakeDatabase()
    store = PostgresStore(connect=db.connect, allowed_user_id=bridge.ALLOWED_USER_ID)
    update = {"update_id": 31, "message": {"from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 22}, "text": "persist me"}}
    assert store.persist_updates([update]) == 32
    assert db.offset == 32
    assert db.inbox[31] == update

    db.fail_on_insert = True
    with pytest.raises(StorageUnavailable):
        store.persist_updates([{"update_id": 32, "message": {"text": "must retry"}}])
    assert db.offset == 32
    assert 32 not in db.inbox


def test_duplicate_update_is_idempotently_kept_once():
    db = FakeDatabase()
    store = PostgresStore(connect=db.connect, allowed_user_id=bridge.ALLOWED_USER_ID)
    update = {"update_id": 34, "message": {"from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 22}, "text": "once"}}
    store.persist_updates([update])
    store.persist_updates([update])
    assert list(db.inbox) == [34]


def test_webhook_persistence_is_idempotent_and_does_not_change_polling_offset():
    db = FakeDatabase()
    db.offset = 120
    store = PostgresStore(connect=db.connect, allowed_user_id=bridge.ALLOWED_USER_ID)
    update = {"update_id": 35, "message": {"from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 22}, "text": "once"}}

    assert store.persist_webhook_update(update) is True
    assert store.persist_webhook_update(update) is False

    assert list(db.inbox) == [35]
    assert db.offset == 120
    assert not any("next_offset" in statement for statement in db.statements)


def _webhook_request(monkeypatch, body, secret="correct-secret", provided="correct-secret"):
    class WebhookStore:
        def __init__(self): self.updates = []
        def persist_webhook_update(self, update): self.updates.append(update)
        def health(self, mode="polling"): return {"ok": True, "mode": mode}

    store = WebhookStore()
    status = bridge.persist_webhook_request(store, provided, secret, len(body), body)
    return status, store


def test_webhook_accepts_valid_update_after_persistence(monkeypatch):
    body = json.dumps({"update_id": 1, "message": {"text": "hello"}}).encode()
    status, store = _webhook_request(monkeypatch, body)
    assert status == 200
    assert store.updates == [{"update_id": 1, "message": {"text": "hello"}}]


def test_webhook_rejects_wrong_secret(monkeypatch):
    status, store = _webhook_request(monkeypatch, b'{"update_id":1}', provided="wrong")
    assert status == 403
    assert store.updates == []


def test_webhook_rejects_oversized_body(monkeypatch):
    monkeypatch.setattr(bridge, "WEBHOOK_MAX_BODY_BYTES", 4)
    status, store = _webhook_request(monkeypatch, b"12345")
    assert status == 413
    assert store.updates == []


@pytest.mark.parametrize("body", [b"{bad", b"[]", b'{"message":{}}', b'{"update_id":true}', b'{"update_id":-1}', b'{"update_id":"4"}', b'{"update_id":9223372036854775808}'])
def test_webhook_rejects_invalid_json_or_update_id(monkeypatch, body):
    status, store = _webhook_request(monkeypatch, body)
    assert status == 400
    assert store.updates == []


def test_webhook_returns_503_when_database_is_unavailable(monkeypatch):
    class UnavailableStore:
        def persist_webhook_update(self, update): raise StorageUnavailable("down")

    store = UnavailableStore()
    body = b'{"update_id":9}'
    assert bridge.persist_webhook_request(store, "correct-secret", "correct-secret", len(body), body) == 503


def test_late_update_is_retained_without_claiming_strict_order():
    db = FakeDatabase()
    store = PostgresStore(connect=db.connect)
    assert store.persist_webhook_update({"update_id": 90})
    assert store.persist_webhook_update({"update_id": 89})
    assert list(db.inbox) == [90, 89]


def test_webhook_duplicate_delivery_is_acknowledged_without_duplicate_inbox_rows():
    db = FakeDatabase()
    store = PostgresStore(connect=db.connect)
    body = b'{"update_id":95}'
    persist = lambda: bridge.persist_webhook_request(store, "local", "local", len(body), body)

    assert persist() == 200
    assert persist() == 200
    assert list(db.inbox) == [95]


def test_webhook_worker_runs_without_polling_lock_and_shutdown_closes_server(monkeypatch, tmp_path):
    token_file = tmp_path / "bot_token"
    token_file.write_text("local-test-token", encoding="utf-8")
    monkeypatch.setattr(bridge, "TOKEN_FILE", token_file)
    monkeypatch.setenv("TELEGRAM_MODE", "webhook")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "local-secret")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setattr(bridge, "_install_shutdown_handlers", lambda stop: None)

    class Store:
        lock_calls = 0
        def __init__(self, **kwargs): pass
        def polling_lock(self, stop):
            self.lock_calls += 1
            raise AssertionError("webhook mode must not acquire polling lock")
        def initialize(self, path): pass
        def recover_stale(self): pass
        def begin_runtime(self): pass
        def import_legacy_history(self, path): return "absent"

    instances = []
    def make_store(**kwargs):
        result = Store()
        instances.append(result)
        return result

    class Server:
        shutdown_called = False
        waited = False
        closed = False
        def shutdown(self): self.shutdown_called = True
        def wait_for_requests(self): self.waited = True
        def server_close(self): self.closed = True

    server = Server()
    monkeypatch.setattr(bridge, "PostgresStore", make_store)
    monkeypatch.setattr(bridge, "start_http_server", lambda *args: server)
    monkeypatch.setattr(bridge, "worker_loop", lambda store, stop, fatal, mode: stop.set())

    bridge.main()

    assert instances[0].lock_calls == 0
    assert server.shutdown_called and server.waited and server.closed


class HealthCursor:
    def __init__(self, row): self.row = row
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, statement, params=None): self.statement = statement
    def fetchone(self): return self.row


class HealthConnection:
    def __init__(self, row): self.row = row
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def cursor(self): return HealthCursor(self.row)


def test_storage_health_uses_mode_specific_polling_requirement():
    now = datetime.now(timezone.utc)
    row = (now, False, None, now, None, None, 0, 0, 0, 0, False, False, False, False)
    store = PostgresStore(connect=lambda: HealthConnection(row))

    polling = store.health(mode="polling")
    webhook = store.health(mode="webhook")

    assert polling["ok"] is False
    assert polling["polling"] is False
    assert webhook["ok"] is True
    assert webhook["polling"] is False
    assert webhook["mode"] == "webhook"


def test_status_text_reports_webhook_mode_without_claiming_polling():
    class StatusStore:
        def health(self, mode):
            return {
                "polling": False, "worker": True, "inbox_stuck": False, "outbox_stuck": False,
                "inbox_pending": 0, "outbox_pending": 0, "inbox_dead_letter": 0, "outbox_dead_letter": 0,
            }

    status = bridge._status_text(StatusStore(), mode="webhook")
    assert "Режим: webhook." in status
    assert "Polling:" not in status
    assert "Worker: работает." in status


def _start_listener_or_skip(monkeypatch, store, secret="listener-secret"):
    monkeypatch.setenv("PORT", "0")
    try:
        return bridge.start_http_server(store, secret)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets запрещены средой: {exc}")


def test_real_http_listener_accepts_valid_webhook(monkeypatch):
    class ListenerStore:
        def __init__(self): self.updates = []
        def persist_webhook_update(self, update): self.updates.append(update)
        def health(self, mode="polling"): return {"ok": True, "mode": mode}

    store = ListenerStore()
    server = _start_listener_or_skip(monkeypatch, store)
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
    body = b'{"update_id":501}'
    try:
        connection.request("POST", "/telegram/webhook", body=body, headers={
            "X-Telegram-Bot-Api-Secret-Token": "listener-secret",
            "Content-Type": "application/json",
        })
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        assert store.updates == [{"update_id": 501}]
    finally:
        connection.close()
        bridge.shutdown_http_server(server)


def test_real_http_listener_checks_secret_before_reading_body(monkeypatch):
    class ListenerStore:
        def persist_webhook_update(self, update): raise AssertionError("invalid secret must not reach storage")
        def health(self, mode="polling"): return {"ok": True}

    server = _start_listener_or_skip(monkeypatch, ListenerStore())
    try:
        client = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=2)
    except PermissionError as exc:
        bridge.shutdown_http_server(server)
        pytest.skip(f"loopback sockets запрещены средой: {exc}")
    client.settimeout(2)
    try:
        client.sendall(
            b"POST /telegram/webhook HTTP/1.0\r\n"
            b"Content-Length: 100\r\n"
            b"X-Telegram-Bot-Api-Secret-Token: wrong\r\n\r\n"
        )
        response = client.recv(1024)
        assert b"403" in response
    finally:
        client.close()
        bridge.shutdown_http_server(server)


def test_real_http_listener_bounds_concurrency_and_shutdown(monkeypatch):
    monkeypatch.setattr(bridge, "HTTP_MAX_CONCURRENT_REQUESTS", 1)
    monkeypatch.setattr(bridge, "WEBHOOK_READ_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(bridge, "HTTP_SHUTDOWN_WAIT_SECONDS", 0.1)

    class ListenerStore:
        def persist_webhook_update(self, update): pass
        def health(self, mode="polling"): return {"ok": True}

    server = _start_listener_or_skip(monkeypatch, ListenerStore())
    try:
        first = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=2)
    except PermissionError as exc:
        bridge.shutdown_http_server(server)
        pytest.skip(f"loopback sockets запрещены средой: {exc}")
    first.settimeout(2)
    try:
        first.sendall(
            b"POST /telegram/webhook HTTP/1.0\r\n"
            b"Content-Length: 100\r\n"
            b"X-Telegram-Bot-Api-Secret-Token: listener-secret\r\n\r\n"
            b"partial"
        )
        second = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
        second.request("GET", "/")
        response = second.getresponse()
        assert response.status == 503
        response.read()
        second.close()

        started = time.monotonic()
        bridge.shutdown_http_server(server)
        assert time.monotonic() - started < 2
    finally:
        first.close()


def test_real_http_listener_times_out_an_incomplete_body(monkeypatch):
    monkeypatch.setattr(bridge, "WEBHOOK_READ_TIMEOUT_SECONDS", 0.1)

    class ListenerStore:
        def persist_webhook_update(self, update): raise AssertionError("incomplete body must not reach storage")
        def health(self, mode="polling"): return {"ok": True}

    server = _start_listener_or_skip(monkeypatch, ListenerStore())
    try:
        client = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=2)
    except PermissionError as exc:
        bridge.shutdown_http_server(server)
        pytest.skip(f"loopback sockets запрещены средой: {exc}")
    client.settimeout(2)
    try:
        client.sendall(
            b"POST /telegram/webhook HTTP/1.0\r\n"
            b"Content-Length: 100\r\n"
            b"X-Telegram-Bot-Api-Secret-Token: listener-secret\r\n\r\n"
            b"partial"
        )
        response = client.recv(1024)
        assert b"400" in response
    finally:
        client.close()
        bridge.shutdown_http_server(server)


def test_recovery_only_reclaims_stale_processing_and_sending_rows():
    db = FakeDatabase()
    store = PostgresStore(connect=db.connect)
    store.recover_stale()
    statements = "\n".join(db.statements)
    assert "processing_at < now()" in statements
    assert "processing_at IS NULL OR" in statements
    assert "sending_at < now()" in statements
    assert "sending_at IS NULL OR" in statements


def test_exhausted_retries_are_retained_in_dead_letter():
    db = FakeDatabase()
    store = PostgresStore(connect=db.connect)
    store.retry_inbox(
        44, MAX_ATTEMPTS, "ProviderUnavailable", uuid.uuid4(), "still saved", 22,
    )
    statements = "\n".join(db.statements).upper()
    assert "UPDATE INBOX SET STATUS = 'DEAD_LETTER'" in statements
    assert "INSERT INTO OUTBOX" in statements
    assert "DELETE" not in statements


class FencedInboxCursor:
    def __init__(self, database):
        self.database = database
        self.result = None

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def execute(self, statement, params=None):
        statement = " ".join(statement.split())
        db = self.database
        db.statements.append(statement)
        if statement.startswith("SELECT update_id, payload, attempts, generation FROM inbox"):
            if db.status in ("received", "retry"):
                self.result = (db.update_id, db.payload, db.attempts, db.generation)
        elif statement.startswith("UPDATE inbox SET status = 'processing'"):
            db.attempts, db.claim_token, update_id = params
            assert update_id == db.update_id
            db.status = "processing"
            db.processing_at = "fresh"
        elif statement.startswith("UPDATE inbox SET status = 'retry', processing_at = NULL") and "processing_at IS NULL OR" in statement:
            if db.status == "processing" and (db.processing_at is None or db.processing_at == "stale"):
                db.status = "retry"
                db.claim_token = None
                db.processing_at = None
        elif statement.startswith("UPDATE inbox SET status = 'completed'"):
            update_id, token = params
            if db.status == "processing" and db.claim_token == token and update_id == db.update_id:
                db.status = "completed"
                db.claim_token = None
                db.processing_at = None
                self.result = (update_id,)
        elif statement.startswith("UPDATE inbox SET status = 'dead_letter'"):
            _, update_id, token = params
            if db.status == "processing" and db.claim_token == token and update_id == db.update_id:
                db.status = "dead_letter"
                db.claim_token = None
                db.processing_at = None
                self.result = (update_id,)
        elif statement.startswith("UPDATE inbox SET status = 'retry'"):
            _, _, update_id, token = params
            if db.status == "processing" and db.claim_token == token and update_id == db.update_id:
                db.status = "retry"
                db.claim_token = None
                db.processing_at = None
                self.result = (update_id,)
        elif statement.startswith("UPDATE inbox SET status = 'handled_error'"):
            _, update_id, token = params
            if db.status == "processing" and db.claim_token == token and update_id == db.update_id:
                db.status = "handled_error"
                db.claim_token = None
                db.processing_at = None
                self.result = (update_id,)
        elif statement.startswith("INSERT INTO messages (chat_id, update_id, role, content, route)"):
            db.messages.append(params)
        elif statement.startswith("INSERT INTO outbox"):
            if db.fail_on_outbox_insert:
                raise OSError("simulated outbox insert failure")
            db.outbox.append(params)

    def fetchone(self): return self.result


class FencedInboxConnection:
    def __init__(self, database): self.database = database
    def __enter__(self):
        db = self.database
        self.snapshot = (db.status, db.attempts, db.claim_token, db.processing_at, list(db.messages), list(db.outbox))
        return self
    def __exit__(self, exc_type, *_):
        if exc_type:
            db = self.database
            db.status, db.attempts, db.claim_token, db.processing_at, db.messages, db.outbox = self.snapshot
        return False
    def cursor(self): return FencedInboxCursor(self.database)


class FencedInboxDatabase:
    update_id = 91
    payload = {"update_id": 91, "message": {"text": "hello"}}
    status = "received"
    attempts = 0
    generation = 3
    claim_token = None
    processing_at = None
    fail_on_outbox_insert = False
    messages = []
    outbox = []
    statements = []

    def connect(self): return FencedInboxConnection(self)


def test_inbox_claim_token_normal_completion_commits_answer_and_outbox():
    db = FencedInboxDatabase()
    db.status, db.attempts, db.generation = "received", 0, 3
    db.claim_token, db.processing_at, db.messages, db.outbox = None, None, [], []
    db.statements = []
    store = PostgresStore(connect=db.connect)

    item = store.claim_inbox()
    assert item is not None and item.claim_token == db.claim_token
    claim_sql = next(sql for sql in db.statements if sql.startswith("SELECT update_id, payload, attempts, generation FROM inbox"))
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    assert store.complete_inbox(item.update_id, 22, "chat", "answer", item.claim_token, item.generation)
    assert db.status == "completed"
    assert db.claim_token is None
    assert db.messages
    assert len(db.outbox) == 1


def test_inbox_completion_sql_error_rolls_back_result_and_outbox():
    db = FencedInboxDatabase()
    db.status, db.attempts, db.generation = "received", 0, 3
    db.claim_token, db.processing_at, db.messages, db.outbox = None, None, [], []
    store = PostgresStore(connect=db.connect)
    item = store.claim_inbox()
    db.fail_on_outbox_insert = True

    with pytest.raises(StorageUnavailable):
        store.complete_inbox(item.update_id, 22, "chat", "answer", item.claim_token, item.generation)

    assert db.status == "processing"
    assert db.claim_token == item.claim_token
    assert db.messages == [] and db.outbox == []


def test_stale_recovery_invalidates_token_before_next_claim():
    db = FencedInboxDatabase()
    db.status, db.attempts, db.generation = "received", 0, 3
    db.claim_token, db.processing_at, db.messages, db.outbox = None, None, [], []
    store = PostgresStore(connect=db.connect)
    stale_item = store.claim_inbox()

    db.processing_at = "stale"
    store.recover_stale()
    assert db.status == "retry"
    assert db.claim_token is None
    current_item = store.claim_inbox()
    assert current_item is not None
    assert current_item.claim_token != stale_item.claim_token
    assert current_item.generation == stale_item.generation


def test_null_processing_at_is_recovered_and_old_worker_is_fenced():
    db = FencedInboxDatabase()
    db.status, db.attempts, db.generation = "received", 0, 3
    db.claim_token, db.processing_at, db.messages, db.outbox = None, None, [], []
    store = PostgresStore(connect=db.connect)
    stale_item = store.claim_inbox()
    db.processing_at = None

    store.recover_stale()
    current_item = store.claim_inbox()

    assert current_item is not None and current_item.claim_token != stale_item.claim_token
    assert not store.complete_inbox(
        stale_item.update_id, 22, "chat", "obsolete", stale_item.claim_token, stale_item.generation,
    )
    assert db.status == "processing"
    assert db.claim_token == current_item.claim_token
    assert db.messages == [] and db.outbox == []


def test_recovery_does_not_reclaim_fresh_inbox_processing():
    db = FencedInboxDatabase()
    db.status, db.attempts, db.generation = "received", 0, 3
    db.claim_token, db.processing_at, db.messages, db.outbox = None, None, [], []
    store = PostgresStore(connect=db.connect)
    item = store.claim_inbox()

    store.recover_stale()

    assert db.status == "processing"
    assert db.claim_token == item.claim_token
    assert db.processing_at == "fresh"


def test_old_worker_cannot_complete_or_create_outbox_after_reclaim():
    db = FencedInboxDatabase()
    db.status, db.attempts, db.generation = "received", 0, 3
    db.claim_token, db.processing_at, db.messages, db.outbox = None, None, [], []
    store = PostgresStore(connect=db.connect)
    stale_item = store.claim_inbox()
    db.processing_at = "stale"
    store.recover_stale()
    current_item = store.claim_inbox()

    assert not store.complete_inbox(
        stale_item.update_id, 22, "chat", "obsolete", stale_item.claim_token, stale_item.generation,
    )
    assert db.status == "processing"
    assert db.claim_token == current_item.claim_token
    assert db.messages == []
    assert db.outbox == []


def test_old_worker_retry_and_permanent_error_are_rejected_after_reclaim():
    db = FencedInboxDatabase()
    db.status, db.attempts, db.generation = "received", 0, 3
    db.claim_token, db.processing_at, db.messages, db.outbox = None, None, [], []
    store = PostgresStore(connect=db.connect)
    stale_item = store.claim_inbox()
    db.processing_at = "stale"
    store.recover_stale()
    current_item = store.claim_inbox()

    assert not store.retry_inbox(
        stale_item.update_id, stale_item.attempts, "old_failure", stale_item.claim_token,
        "obsolete retry", 22, stale_item.generation,
    )
    assert not store.permanent_inbox_error(
        stale_item.update_id, 22, "obsolete error", "old_error",
        stale_item.claim_token, stale_item.generation,
    )
    assert db.status == "processing"
    assert db.claim_token == current_item.claim_token
    assert db.messages == []
    assert db.outbox == []


def test_manual_outbox_requeue_is_scoped_to_originating_chat():
    db = RetryFenceDatabase()
    db.manual_outbox_id = 55
    db.manual_outbox_status = "dead_letter"
    db.manual_outbox_chat = 22
    db.manual_outbox_error = "telegram_error"
    store = PostgresStore(connect=db.connect)
    assert store.requeue_dead_letter(
        "outbox", 55, 22,
        command_update_id=db.command_id, claim_token=db.command_token,
    )
    statement = next(s for s, _ in db.statements if "RETURNING id" in s and "chat_id = %s" in s)
    assert "chat_id = %s" in statement
    assert "status = 'dead_letter'" in statement


class RetryFenceCursor:
    def __init__(self, database):
        self.database = database
        self.result = None

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def execute(self, query, params=None):
        query = " ".join(query.split())
        db = self.database
        db.statements.append((query, params))
        if query.startswith("SELECT update_id FROM inbox"):
            update_id, token = params
            if db.command_id == update_id and db.command_status == "processing" and db.command_token == token:
                self.result = (update_id,)
        elif query.startswith("SELECT generation FROM inbox"):
            update_id, chat_id = params
            if db.target_id == update_id and db.target_chat == int(chat_id) and db.target_status == "dead_letter":
                self.result = (db.target_generation,)
        elif query.startswith("INSERT INTO message_revisions"):
            db.revisions.append(params)
        elif query.startswith("UPDATE outbox SET status = 'dead_letter'"):
            update_id, generation = params
            if update_id == db.target_id and generation == db.target_generation:
                for row in db.outbox:
                    if row["generation"] == generation and row["status"] in ("pending", "sending"):
                        row["status"] = "dead_letter"
                        row["claim_token"] = None
        elif query.startswith("UPDATE outbox SET status = 'pending', attempts = 0"):
            item_id, chat_id = params
            if (item_id == db.manual_outbox_id and chat_id == db.manual_outbox_chat
                    and db.manual_outbox_status == "dead_letter"
                    and db.manual_outbox_error != "superseded_by_inbox_retry"):
                db.manual_outbox_status = "pending"
                self.result = (item_id,)
        elif query.startswith("UPDATE inbox SET status = 'retry'"):
            (update_id,) = params
            if update_id == db.target_id and db.target_status == "dead_letter":
                if db.fail_on_target_update:
                    raise OSError("simulated transaction failure")
                db.target_status = "retry"
                db.target_generation += 1
                self.result = (update_id,)

    def fetchone(self): return self.result


class RetryFenceConnection:
    def __init__(self, database): self.database = database
    def __enter__(self):
        db = self.database
        self.snapshot = (
            db.command_status, db.command_token, db.target_status, db.target_generation,
            copy.deepcopy(db.outbox), list(db.revisions), db.manual_outbox_status,
        )
        return self
    def __exit__(self, exc_type, *_):
        if exc_type:
            db = self.database
            (db.command_status, db.command_token, db.target_status, db.target_generation,
             db.outbox, db.revisions, db.manual_outbox_status) = self.snapshot
        return False
    def cursor(self): return RetryFenceCursor(self.database)


class RetryFenceDatabase:
    def __init__(self):
        self.command_id = 101
        self.command_status = "processing"
        self.command_token = uuid.UUID("b17324b4-4739-4307-a0ea-5607dc8d08b8")
        self.target_id = 55
        self.target_chat = 22
        self.target_status = "dead_letter"
        self.target_generation = 2
        self.outbox = [
            {"generation": 2, "status": "pending"},
            {"generation": 2, "status": "sending"},
            {"generation": 2, "status": "sent"},
        ]
        self.revisions = []
        self.statements = []
        self.manual_outbox_id = None
        self.manual_outbox_status = None
        self.manual_outbox_chat = None
        self.manual_outbox_error = None
        self.fail_on_target_update = False

    def connect(self): return RetryFenceConnection(self)


def _retry_fence_snapshot(db):
    return (
        db.target_status, db.target_generation, copy.deepcopy(db.outbox),
        list(db.revisions),
    )


def test_inbox_retry_with_current_command_claim_starts_new_generation():
    db = RetryFenceDatabase()
    store = PostgresStore(connect=db.connect)

    assert store.requeue_dead_letter(
        "inbox", db.target_id, db.target_chat,
        command_update_id=db.command_id, claim_token=db.command_token,
    )
    assert db.target_status == "retry"
    assert db.target_generation == 3
    assert len(db.revisions) == 1
    assert [row["status"] for row in db.outbox] == ["dead_letter", "dead_letter", "sent"]
    assert db.statements[0][0].startswith("SELECT update_id FROM inbox")
    assert "status = 'processing' AND claim_token = %s" in db.statements[0][0]


def test_inbox_retry_rejects_stale_command_claim_without_side_effects():
    db = RetryFenceDatabase()
    store = PostgresStore(connect=db.connect)
    before = _retry_fence_snapshot(db)

    assert not store.requeue_dead_letter(
        "inbox", db.target_id, db.target_chat,
        command_update_id=db.command_id, claim_token=uuid.uuid4(),
    )
    assert _retry_fence_snapshot(db) == before
    assert len(db.statements) == 1


def test_inbox_retry_old_worker_is_blocked_after_command_reclaim():
    db = RetryFenceDatabase()
    old_token = db.command_token
    db.command_token = uuid.uuid4()
    store = PostgresStore(connect=db.connect)
    before = _retry_fence_snapshot(db)

    assert not store.requeue_dead_letter(
        "inbox", db.target_id, db.target_chat,
        command_update_id=db.command_id, claim_token=old_token,
    )
    assert _retry_fence_snapshot(db) == before


def test_inbox_retry_token_rejection_changes_neither_target_nor_outbox():
    db = RetryFenceDatabase()
    db.command_status = "retry"
    store = PostgresStore(connect=db.connect)
    before = _retry_fence_snapshot(db)

    assert not store.requeue_dead_letter(
        "inbox", db.target_id, db.target_chat,
        command_update_id=db.command_id, claim_token=db.command_token,
    )
    assert _retry_fence_snapshot(db) == before
    assert len(db.statements) == 1


def test_inbox_retry_generation_changes_only_for_valid_claim():
    db = RetryFenceDatabase()
    store = PostgresStore(connect=db.connect)
    assert not store.requeue_dead_letter(
        "inbox", db.target_id, db.target_chat,
        command_update_id=db.command_id, claim_token=uuid.uuid4(),
    )
    assert db.target_generation == 2
    assert store.requeue_dead_letter(
        "inbox", db.target_id, db.target_chat,
        command_update_id=db.command_id, claim_token=db.command_token,
    )
    assert db.target_generation == 3


def test_inbox_retry_preserves_supersede_scope():
    db = RetryFenceDatabase()
    store = PostgresStore(connect=db.connect)
    assert store.requeue_dead_letter(
        "inbox", db.target_id, db.target_chat,
        command_update_id=db.command_id, claim_token=db.command_token,
    )
    supersede_sql = next(query for query, _ in db.statements if query.startswith("UPDATE outbox SET status = 'dead_letter'"))
    assert "sent" not in supersede_sql
    assert "status IN ('pending', 'sending')" in supersede_sql
    assert "claim_token = NULL" in supersede_sql


def test_inbox_retry_sql_failure_rolls_back_history_and_supersede_changes():
    db = RetryFenceDatabase()
    db.fail_on_target_update = True
    store = PostgresStore(connect=db.connect)
    before = _retry_fence_snapshot(db)

    with pytest.raises(StorageUnavailable):
        store.requeue_dead_letter(
            "inbox", db.target_id, db.target_chat,
            command_update_id=db.command_id, claim_token=db.command_token,
        )

    assert _retry_fence_snapshot(db) == before


def test_superseded_outbox_item_is_not_sent(monkeypatch):
    class Store:
        def __init__(self):
            self.marked_sent = []

        def outbox_is_current(self, item_id, claim_token):
            return False

        def deliver_outbox(self, outgoing, send_callback):
            return False

        def mark_outbox_sent(self, item_id, claim_token):
            self.marked_sent.append(item_id)

    store = Store()
    sends = []
    monkeypatch.setattr(bridge, "send", lambda *args: sends.append(args))
    outgoing = storage.OutboxItem(
        id=8, chat_id=22, text="stale part", attempts=1, claim_token=uuid.uuid4(),
    )

    assert bridge.deliver_outbox_item(store, outgoing) is False
    assert sends == []
    assert store.marked_sent == []


def test_outbox_send_guard_requires_current_generation():
    class GuardCursor:
        def __init__(self, db): self.db = db
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, statement, params=None): self.db.statement, self.db.params = statement, params
        def fetchone(self): return (self.db.current,)

    class GuardConnection:
        def __init__(self, db): self.db = db
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def cursor(self): return GuardCursor(self.db)

    class GuardDatabase:
        current = False
        statement = ""
        params = None
        def connect(self): return GuardConnection(self)

    db = GuardDatabase()
    store = PostgresStore(connect=db.connect)

    claim_token = uuid.uuid4()
    assert store.outbox_is_current(8, claim_token) is False
    assert "o.generation = i.generation" in db.statement
    assert "o.status = 'sending'" in db.statement
    assert "o.claim_token = %s" in db.statement
    assert db.params == (8, claim_token)
    db.current = True
    assert store.outbox_is_current(8, claim_token) is True


def test_mark_outbox_sent_requires_a_current_claim_even_after_supersede():
    class MarkCursor:
        def __init__(self, db): self.db = db
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, statement, params=None): self.db.statements.append((statement, params))
        def fetchone(self): return None

    class MarkConnection:
        def __init__(self, db): self.db = db
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def cursor(self): return MarkCursor(self.db)

    class MarkDatabase:
        statements = []
        def connect(self): return MarkConnection(self)

    db = MarkDatabase()
    assert not PostgresStore(connect=db.connect).mark_outbox_sent(8, uuid.uuid4())
    mark_sql, params = next((statement, params) for statement, params in db.statements if statement.startswith("UPDATE outbox SET status = 'sent'"))
    assert "status = 'sending'" in mark_sql
    assert "claim_token = %s" in mark_sql
    assert len(params) == 2


class OutboxFenceCursor:
    def __init__(self, database):
        self.database = database
        self.result = None

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def execute(self, query, params=None):
        query = " ".join(query.split())
        db = self.database
        db.statements.append((query, params))
        if query.startswith("SELECT update_id FROM inbox"):
            command_id, token = params
            if db.command_status == "processing" and db.command_id == command_id and db.command_token == token:
                self.result = (command_id,)
        elif query.startswith("SELECT i.generation FROM inbox i"):
            db.delivery_lock.acquire()
            db.delivery_locked = True
            self.result = (db.inbox_generation,)
        elif query.startswith("SELECT status, claim_token, generation FROM outbox"):
            if db.status == "sending":
                self.result = (db.status, db.claim_token, db.generation)
        elif query.startswith("UPDATE outbox SET status = 'sent', sending_at = NULL, sent_at = now()") and "AND generation = %s RETURNING id" in query:
            item_id, token, generation = params
            if item_id == db.item_id and db.status == "sending" and db.claim_token == token and db.generation == generation:
                db.status, db.claim_token, db.sending_at = "sent", None, None
                self.result = (item_id,)
        elif query.startswith("SELECT id, chat_id, text, attempts FROM outbox"):
            if db.status == "pending":
                self.result = (db.item_id, db.chat_id, db.text, db.attempts)
        elif query.startswith("UPDATE outbox SET status = 'sending'"):
            attempts, token, item_id = params
            if item_id == db.item_id and db.status == "pending":
                db.status, db.attempts, db.claim_token, db.sending_at = "sending", attempts, token, "fresh"
                self.result = (item_id,)
        elif query.startswith("SELECT EXISTS"):
            item_id, token = params
            self.result = (item_id == db.item_id and db.status == "sending"
                           and db.claim_token == token and db.generation == db.inbox_generation,)
        elif query.startswith("UPDATE outbox SET status = 'sent'"):
            item_id, token = params
            if item_id == db.item_id and db.status == "sending" and db.claim_token == token:
                db.status, db.claim_token, db.sending_at = "sent", None, None
                self.result = (item_id,)
        elif query.startswith("UPDATE outbox SET status = 'dead_letter'"):
            error_code, item_id, token = params
            if item_id == db.item_id and db.status == "sending" and db.claim_token == token:
                db.status, db.claim_token, db.sending_at, db.last_error_code = "dead_letter", None, None, error_code
                self.result = (item_id,)
        elif query.startswith("UPDATE outbox SET status = 'pending', sending_at = NULL, last_error_code"):
            error_code, _delay, item_id, token = params
            if item_id == db.item_id and db.status == "sending" and db.claim_token == token:
                db.status, db.claim_token, db.sending_at, db.last_error_code = "pending", None, None, error_code
                self.result = (item_id,)
        elif query.startswith("UPDATE outbox SET status = 'pending', sending_at = NULL, claim_token = NULL"):
            db.delivery_lock.acquire()
            db.delivery_locked = True
            if (db.status == "sending" and "sending_at IS NULL OR" in query
                    and (db.sending_at is None or db.sending_at == "stale")):
                db.status, db.claim_token, db.sending_at = "pending", None, None
        elif query.startswith("UPDATE outbox SET status = 'pending', attempts = 0"):
            item_id, chat_id = params
            if item_id == db.item_id and chat_id == db.chat_id and db.status == "dead_letter":
                db.status, db.attempts, db.claim_token, db.sending_at, db.last_error_code = "pending", 0, None, None, None
                self.result = (item_id,)
        elif query.startswith("UPDATE bot_state SET last_outbox_progress_at"):
            if db.fail_on_progress:
                raise OSError("simulated progress update failure")
            db.progress_updates += 1

    def fetchone(self): return self.result


class OutboxFenceConnection:
    def __init__(self, database): self.database = database
    def __enter__(self):
        db = self.database
        self.snapshot = (
            db.status, db.attempts, db.claim_token, db.sending_at, db.last_error_code,
            db.progress_updates, db.command_status, db.command_token,
        )
        return self
    def __exit__(self, exc_type, *_):
        if exc_type:
            (self.database.status, self.database.attempts, self.database.claim_token,
             self.database.sending_at, self.database.last_error_code, self.database.progress_updates,
             self.database.command_status, self.database.command_token) = self.snapshot
        if getattr(self.database, "delivery_locked", False):
            self.database.delivery_locked = False
            self.database.delivery_lock.release()
        return False
    def cursor(self): return OutboxFenceCursor(self.database)


class OutboxFenceDatabase:
    def __init__(self):
        self.item_id = 8
        self.chat_id = 22
        self.text = "reply"
        self.status = "pending"
        self.attempts = 0
        self.generation = 4
        self.inbox_generation = 4
        self.claim_token = None
        self.sending_at = None
        self.last_error_code = None
        self.progress_updates = 0
        self.fail_on_progress = False
        self.delivery_lock = threading.Lock()
        self.delivery_locked = False
        self.statements = []
        self.command_id = 101
        self.command_status = "processing"
        self.command_token = uuid.UUID("2e2b442a-312e-4831-a014-a71a102a7640")

    def connect(self): return OutboxFenceConnection(self)


def _outbox_snapshot(db):
    return db.status, db.attempts, db.claim_token, db.sending_at, db.last_error_code, db.progress_updates


def test_outbox_claim_and_successful_send_release_token():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)

    item = store.claim_outbox()
    assert item is not None
    assert item.claim_token == db.claim_token
    claim_sql = next(query for query, _ in db.statements if query.startswith("SELECT id, chat_id, text, attempts FROM outbox"))
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    assert db.status == "sending"
    assert store.outbox_is_current(item.id, item.claim_token)
    assert store.mark_outbox_sent(item.id, item.claim_token)
    assert db.status == "sent"
    assert db.claim_token is None


def test_outbox_stale_recovery_clears_token_and_next_claim_is_new():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    old_item = store.claim_outbox()

    db.sending_at = "stale"
    store.recover_stale()
    assert db.status == "pending"
    assert db.claim_token is None
    new_item = store.claim_outbox()
    assert new_item is not None
    assert new_item.claim_token != old_item.claim_token


def test_null_sending_at_is_recovered_and_old_worker_is_fenced():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    old_item = store.claim_outbox()
    db.sending_at = None

    store.recover_stale()
    current_item = store.claim_outbox()

    assert current_item is not None and current_item.claim_token != old_item.claim_token
    assert not store.mark_outbox_sent(old_item.id, old_item.claim_token)
    assert not store.retry_outbox(old_item.id, old_item.attempts, "late_error", old_item.claim_token)
    assert db.status == "sending"
    assert db.claim_token == current_item.claim_token


def test_recovery_does_not_reclaim_fresh_outbox_sending():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    item = store.claim_outbox()

    store.recover_stale()

    assert db.status == "sending"
    assert db.claim_token == item.claim_token
    assert db.sending_at == "fresh"


def test_old_outbox_worker_cannot_mark_new_claim_sent():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    old_item = store.claim_outbox()
    db.sending_at = "stale"
    store.recover_stale()
    current_item = store.claim_outbox()
    before = _outbox_snapshot(db)

    assert not store.mark_outbox_sent(old_item.id, old_item.claim_token)
    assert _outbox_snapshot(db) == before
    assert db.claim_token == current_item.claim_token


def test_old_outbox_worker_does_not_send_after_reclaim(monkeypatch):
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    old_item = store.claim_outbox()
    db.sending_at = "stale"
    store.recover_stale()
    store.claim_outbox()
    sends = []
    monkeypatch.setattr(bridge, "send", lambda *args: sends.append(args))

    assert bridge.deliver_outbox_item(store, old_item) is False
    assert sends == []


@pytest.mark.parametrize("permanent", [False, True])
def test_old_outbox_worker_cannot_retry_or_dead_letter_new_claim(permanent):
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    old_item = store.claim_outbox()
    db.sending_at = "stale"
    store.recover_stale()
    current_item = store.claim_outbox()
    before = _outbox_snapshot(db)

    assert not store.retry_outbox(
        old_item.id, old_item.attempts, "old_error", old_item.claim_token,
        permanent=permanent,
    )
    assert _outbox_snapshot(db) == before
    assert db.claim_token == current_item.claim_token


def test_manual_outbox_retry_does_not_let_old_worker_change_new_attempt():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    old_item = store.claim_outbox()
    assert store.retry_outbox(
        old_item.id, old_item.attempts, "permanent", old_item.claim_token,
        permanent=True,
    )
    assert store.requeue_dead_letter(
        "outbox", old_item.id, old_item.chat_id,
        command_update_id=db.command_id, claim_token=db.command_token,
    )
    current_item = store.claim_outbox()
    before = _outbox_snapshot(db)

    assert not store.retry_outbox(
        old_item.id, old_item.attempts, "old_worker", old_item.claim_token,
    )
    assert _outbox_snapshot(db) == before
    assert db.claim_token == current_item.claim_token


def test_superseded_outbox_generation_invalidates_claim_and_blocks_send(monkeypatch):
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    item = store.claim_outbox()
    db.inbox_generation += 1
    db.status = "dead_letter"
    db.claim_token = None
    sends = []
    monkeypatch.setattr(bridge, "send", lambda *args: sends.append(args))

    assert bridge.deliver_outbox_item(store, item) is False
    assert sends == []
    assert not store.mark_outbox_sent(item.id, item.claim_token)
    assert db.status == "dead_letter"
    assert db.claim_token is None


def test_outbox_claim_mismatch_does_not_change_progress_or_record_state():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    item = store.claim_outbox()
    db.claim_token = uuid.uuid4()
    before = _outbox_snapshot(db)

    assert not store.mark_outbox_sent(item.id, item.claim_token)
    assert not store.retry_outbox(item.id, item.attempts, "old_error", item.claim_token)
    assert _outbox_snapshot(db) == before


def test_outbox_transition_sql_error_rolls_back_state_change():
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    item = store.claim_outbox()
    db.fail_on_progress = True
    before = _outbox_snapshot(db)

    with pytest.raises(StorageUnavailable):
        store.retry_outbox(item.id, item.attempts, "temporary", item.claim_token)

    assert _outbox_snapshot(db) == before


def test_history_order_is_deterministic_for_same_update_id():
    class HistoryCursor:
        def __init__(self, db):
            self.db = db

        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, statement, params=None): self.db.statement = statement
        def fetchall(self): return self.db.rows

    class HistoryConnection:
        def __init__(self, db): self.db = db
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def cursor(self): return HistoryCursor(self.db)

    class HistoryDatabase:
        # The CASE expression yields assistant before user in the descending
        # query; history() reverses this to conversation order.
        rows = [
            ("assistant", "new answer", 12),
            ("user", "new question", 12),
            ("assistant", "old answer", 11),
            ("user", "old question", 11),
        ]
        statement = ""
        def connect(self): return HistoryConnection(self)

    db = HistoryDatabase()
    history = PostgresStore(connect=db.connect).history(chat_id=22, before_update_id=13, limit=10)

    assert "ORDER BY update_id DESC" in db.statement
    assert "CASE role WHEN 'assistant' THEN 0 WHEN 'user' THEN 1 ELSE 2 END ASC" in db.statement
    assert history == [
        {"role": "user", "text": "old question"},
        {"role": "assistant", "text": "old answer"},
        {"role": "user", "text": "new question"},
        {"role": "assistant", "text": "new answer"},
    ]


def test_legacy_history_import_is_idempotent_and_keeps_source(tmp_path):
    source = tmp_path / "chat_memory.json"
    original = json.dumps({"22": [
        {"role": "user", "text": "first"},
        {"role": "assistant", "text": "second"},
    ]}).encode()
    source.write_bytes(original)
    database = ImportDatabase()
    store = PostgresStore(connect=database.connect)

    assert store.import_legacy_history(source) == "imported"
    assert store.import_legacy_history(source) == "already_imported"
    assert source.read_bytes() == original
    assert len(database.messages) == 2
    assert [row[2] for row in database.messages] == ["user", "assistant"]
    assert database.messages[0][1] < database.messages[1][1] < 0


class ImportCursor:
    def __init__(self, database):
        self.database = database
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        if statement.startswith("SELECT source_hash"):
            self.result = (self.database.marker,) if self.database.marker else None
        elif statement.startswith("INSERT INTO messages"):
            self.database.messages.append(params)
        elif statement.startswith("INSERT INTO data_imports"):
            self.database.marker = True

    def fetchone(self):
        return self.result


class ImportConnection:
    def __init__(self, database):
        self.database = database

    def __enter__(self):
        self.snapshot = (self.database.marker, list(self.database.messages))
        return self

    def __exit__(self, exc_type, *_):
        if exc_type:
            self.database.marker, self.database.messages = self.snapshot
        return False

    def cursor(self):
        return ImportCursor(self.database)


class ImportDatabase:
    def __init__(self):
        self.marker = False
        self.messages = []

    def connect(self):
        return ImportConnection(self)


class FakeStore:
    def __init__(self):
        self.completed = []
        self.retries = []
        self.permanent = []
        self.requeued = []

    def history(self, *args):
        return []

    def complete_inbox(self, *args):
        self.completed.append(args)

    def retry_inbox(self, *args):
        self.retries.append(args)

    def permanent_inbox_error(self, *args):
        self.permanent.append(args)

    def ignore_inbox(self, *args):
        pass

    def requeue_dead_letter(self, *args, **kwargs):
        self.requeued.append((args, kwargs))
        return True


def test_retry_inbox_bridge_passes_source_claim_to_storage(monkeypatch):
    store = FakeStore()
    item = InboxItemForTest({
        "update_id": 72,
        "message": {
            "from": {"id": bridge.ALLOWED_USER_ID},
            "chat": {"id": 22},
            "text": "/retry inbox 55",
        },
    })
    monkeypatch.setattr(core, "generate_answer", lambda *args: pytest.fail("local retry must not call AI"))

    bridge.process_inbox_item(store, item)

    assert store.requeued == [(
        ("inbox", 55, 22),
        {"command_update_id": item.update_id, "claim_token": item.claim_token},
    )]


def test_one_update_completes_once_without_intermediate_send(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "generate_answer", lambda text, history: calls.append(text) or "Ответ")
    store = FakeStore()
    update = {"update_id": 70, "message": {"from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 22}, "text": "привет"}}
    bridge.process_inbox_item(store, InboxItemForTest(update))
    assert calls == ["привет"]
    assert len(store.completed) == 1
    assert not hasattr(store, "sent")


def test_repeat_processing_does_not_call_codex_or_send_status(monkeypatch):
    codex_calls = []
    monkeypatch.setattr(core, "generate_answer", lambda text, history: "Ответ")
    monkeypatch.setattr(bridge, "ask_codex", lambda *args: codex_calls.append(args))
    store = FakeStore()
    update = {"update_id": 71, "message": {"from": {"id": bridge.ALLOWED_USER_ID}, "chat": {"id": 22}, "text": "привет"}}
    bridge.process_inbox_item(store, InboxItemForTest(update))
    assert codex_calls == []
    assert len(store.completed) == 1


def test_configured_model_pairs_are_truthful(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("GROQ_MODEL", "groq/custom")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert providers.configured_provider_models() == (("Groq", "groq/custom"),)


class InboxItemForTest:
    attempts = 1
    generation = 0
    claim_token = uuid.UUID("c4f683aa-8c32-438f-bdc8-8a8d7331e0dc")

    def __init__(self, payload):
        self.update_id = payload["update_id"]
        self.payload = payload


def test_outbox_stale_recovery_race_does_not_duplicate_send(monkeypatch):
    db = OutboxFenceDatabase()
    store = PostgresStore(connect=db.connect)
    old_item = store.claim_outbox()

    a_at_send = threading.Event()
    release_a = threading.Event()
    sends = []
    old_results = []
    worker_errors = []

    def fake_send(chat_id, text):
        if threading.current_thread().name == "outbox-worker-a":
            a_at_send.set()
            if not release_a.wait(5):
                raise AssertionError("worker A was not released")
        sends.append((chat_id, text))

    def deliver_old_item():
        try:
            old_results.append(bridge.deliver_outbox_item(store, old_item))
        except BaseException as exc:
            worker_errors.append(exc)

    monkeypatch.setattr(bridge, "send", fake_send)

    worker_a = threading.Thread(
        target=deliver_old_item,
        name="outbox-worker-a",
    )
    worker_a.start()

    try:
        assert a_at_send.wait(5), "worker A did not reach Telegram send"
        db.sending_at = "stale"
        recovery_done = threading.Event()

        def recover_then_claim():
            store.recover_stale()
            recovery_done.set()

        recovery = threading.Thread(target=recover_then_claim)
        recovery.start()
        assert not recovery_done.wait(0.05), "stale recovery bypassed the active delivery lock"
    finally:
        release_a.set()
        worker_a.join(5)
    recovery.join(5)

    assert not worker_a.is_alive()
    assert not recovery.is_alive()
    assert worker_errors == []
    assert old_results == [True]
    assert sends == [(22, "reply")]
    assert db.status == "sent"
    assert db.claim_token is None



def test_main_waits_for_previous_polling_instance_during_render_handover(monkeypatch, tmp_path):
    token_file = tmp_path / "bot_token"
    token_file.write_text("local-test-token", encoding="utf-8")
    monkeypatch.setattr(bridge, "TOKEN_FILE", token_file)
    monkeypatch.delenv("TELEGRAM_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setattr(bridge, "POLLING_LOCK_RETRY_SECONDS", 0.001)
    monkeypatch.setattr(bridge, "_install_shutdown_handlers", lambda stop: None)

    class LockContext:
        def __enter__(self): return threading.Event()
        def __exit__(self, *args): return False

    class Store:
        lock_calls = 0
        def __init__(self, **kwargs): pass
        def polling_lock(self, stop):
            self.lock_calls += 1
            if self.lock_calls == 1:
                raise storage.StorageConfigurationError("Другой polling-процесс уже запущен")
            return LockContext()
        def initialize(self, path): pass
        def recover_stale(self): pass
        def begin_runtime(self): pass
        def import_legacy_history(self, path): return "absent"

    class Server:
        shutdown_called = False
        waited = False
        closed = False
        def shutdown(self): self.shutdown_called = True
        def wait_for_requests(self): self.waited = True
        def server_close(self): self.closed = True

    stores = []
    def make_store(**kwargs):
        result = Store()
        stores.append(result)
        return result

    server = Server()
    monkeypatch.setattr(bridge, "PostgresStore", make_store)
    monkeypatch.setattr(bridge, "start_http_server", lambda *args: server)
    monkeypatch.setattr(bridge, "shutdown_http_server", lambda value: (
        value.shutdown(), value.wait_for_requests(), value.server_close()
    ))
    monkeypatch.setattr(bridge, "poll_loop", lambda store, stop, fatal: stop.set())
    monkeypatch.setattr(bridge, "worker_loop", lambda store, stop, fatal, mode: stop.set())

    bridge.main()

    assert stores[0].lock_calls == 2
    assert server.shutdown_called and server.waited and server.closed

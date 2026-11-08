"""PostgreSQL-backed durable inbox, conversation history, and outbox."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread


MAX_ATTEMPTS = 8
BASE_RETRY_SECONDS = 2
MAX_RETRY_SECONDS = 15 * 60
POLL_LOCK_ID = 731_904_251
LOCK_CHECK_SECONDS = 3
INBOX_STALE_SECONDS = 360
OUTBOX_STALE_SECONDS = 60
WORKER_STALE_SECONDS = 360


class StorageConfigurationError(Exception):
    pass


class StorageTransientError(Exception):
    pass


class StorageUnavailable(StorageTransientError):
    pass


class HistoryImportError(Exception):
    pass


@dataclass(frozen=True)
class InboxItem:
    update_id: int
    payload: dict
    attempts: int
    claim_token: uuid.UUID
    generation: int = 0


@dataclass(frozen=True)
class OutboxItem:
    id: int
    chat_id: int
    text: str
    attempts: int
    claim_token: uuid.UUID


def retry_delay(attempts: int, *, jitter: bool = True) -> int:
    """Return capped exponential backoff; attempts is one-based."""
    exponent = max(0, min(attempts - 1, 20))
    delay = min(MAX_RETRY_SECONDS, BASE_RETRY_SECONDS * (2**exponent))
    return random.randint(max(1, delay // 2), delay) if jitter else delay


class PostgresStore:
    def __init__(self, database_url: str | None = None, connect=None, allowed_user_id: int = 1193680242):
        self.database_url = database_url or os.environ.get("DATABASE_URL", "")
        self._connect = connect
        self.allowed_user_id = allowed_user_id
        if not self.database_url and connect is None:
            raise StorageConfigurationError("DATABASE_URL не задана")

    def _connection(self):
        if self._connect:
            return self._connect()
        try:
            import psycopg
        except ImportError as exc:
            raise StorageConfigurationError("Не установлен PostgreSQL-драйвер") from exc
        return psycopg.connect(
            self.database_url,
            connect_timeout=8,
            sslmode="require",
            application_name="klodd-telegram",
            keepalives=1,
            keepalives_idle=5,
            keepalives_interval=2,
            keepalives_count=3,
            tcp_user_timeout=10000,
            options="-c statement_timeout=4000 -c lock_timeout=3000",
        )

    @staticmethod
    def _raise_db_error(exc):
        sqlstate = getattr(exc, "sqlstate", None)
        retryable_states = (
            "08", "53", "55P03", "57014", "57P01", "57P02", "57P03",
            "40001", "40P01",
        )
        transient = bool(sqlstate and any(sqlstate.startswith(state) for state in retryable_states))
        if sqlstate is None and type(exc).__name__ in (
            "OperationalError", "InterfaceError", "ConnectionError",
            "ConnectionTimeout", "TimeoutError", "OSError",
        ):
            transient = True
        safe_code = sqlstate or type(exc).__name__
        if transient:
            raise StorageUnavailable(safe_code) from None
        raise StorageConfigurationError("Database operation failed (" + safe_code + ")") from None

    @contextmanager
    def _db(self):
        try:
            with self._connection() as connection:
                yield connection
        except StorageConfigurationError:
            raise
        except Exception as exc:
            self._raise_db_error(exc)

    def initialize(self, migrations_path: str | Path) -> None:
        migration_dir = Path(migrations_path)
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                     migration_name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
            )
        for migration in sorted(migration_dir.glob("*.sql")):
            sql = migration.read_text(encoding="utf-8")
            with self._db() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM schema_migrations WHERE migration_name = %s",
                    (migration.name,),
                )
                if cursor.fetchone():
                    continue
                cursor.execute(sql)
                cursor.execute(
                    "INSERT INTO schema_migrations (migration_name) VALUES (%s)",
                    (migration.name,),
                )

    @contextmanager
    def polling_lock(self, stop_event: Event):
        """Hold and actively monitor one session advisory-lock connection."""
        try:
            connection = self._connection()
        except Exception as exc:
            self._raise_db_error(exc)

        monitor_stop = Event()
        lock_lost = Event()
        acquired = False
        lock_backend_pid = None
        lock_backend_start = None

        def monitor():
            while not monitor_stop.wait(LOCK_CHECK_SECONDS):
                try:
                    if not self._polling_lock_owned(lock_backend_pid, lock_backend_start):
                        raise StorageUnavailable("polling_lock_not_owned")
                except Exception:
                    lock_lost.set()
                    stop_event.set()
                    return

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_try_advisory_lock(%s)", (POLL_LOCK_ID,))
                acquired = bool(cursor.fetchone()[0])
                if acquired:
                    cursor.execute(
                        "SELECT pg_backend_pid(), backend_start FROM pg_stat_activity WHERE pid = pg_backend_pid()"
                    )
                    lock_backend_pid, lock_backend_start = cursor.fetchone()
            connection.commit()
            if not acquired:
                raise StorageConfigurationError("Другой polling-процесс уже запущен")
            self._poll_lock_backend_pid = lock_backend_pid
            self._poll_lock_backend_start = lock_backend_start
            self._poll_lock_lost = lock_lost
            watcher = Thread(target=monitor, name="postgres-lock-watchdog")
            watcher.start()
            try:
                yield lock_lost
            finally:
                monitor_stop.set()
                # Do not close/release the lock connection while its watchdog
                # may still be checking ownership on a separate connection.
                watcher.join()
        except StorageConfigurationError:
            raise
        except Exception as exc:
            self._raise_db_error(exc)
        finally:
            if acquired and not lock_lost.is_set():
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT pg_advisory_unlock(%s)", (POLL_LOCK_ID,))
                    connection.commit()
                except Exception:
                    stop_event.set()
            try:
                connection.close()
            except Exception:
                pass
            self._poll_lock_backend_pid = None
            self._poll_lock_backend_start = None
            self._poll_lock_lost = None

    def _polling_lock_owned(self, backend_pid: int | None, backend_start=None) -> bool:
        """Check the lock belongs to the original PostgreSQL backend session."""
        if backend_pid is None or backend_start is None:
            return False
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT EXISTS (SELECT 1 FROM pg_locks l
                   JOIN pg_stat_activity a ON a.pid = l.pid
                   WHERE l.locktype = 'advisory' AND l.granted AND l.pid = %s
                     AND a.backend_start = %s
                     AND l.classid = 0 AND l.objid = %s AND l.objsubid = 1)""",
                (backend_pid, backend_start, POLL_LOCK_ID),
            )
            row = cursor.fetchone()
            return bool(row and row[0])

    def verify_polling_lock(self) -> bool:
        """Fail closed if the watchdog has already detected a lost session."""
        lost = getattr(self, "_poll_lock_lost", None)
        if lost is not None and lost.is_set():
            return False
        try:
            return self._polling_lock_owned(
                getattr(self, "_poll_lock_backend_pid", None),
                getattr(self, "_poll_lock_backend_start", None),
            )
        except (StorageUnavailable, StorageConfigurationError):
            return False

    def offset(self) -> int | None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT next_offset FROM bot_state WHERE singleton = TRUE")
            row = cursor.fetchone()
            return row[0] if row else None

    def record_poll(self, succeeded: bool, error_code: str | None = None) -> None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE bot_state SET last_poll_at = now(),
                   last_poll_ok = %s, last_error_code = %s WHERE singleton = TRUE""",
                (succeeded, error_code),
            )

    def record_worker_heartbeat(self) -> None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE bot_state SET last_worker_at = now() WHERE singleton = TRUE")

    def begin_runtime(self) -> None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE bot_state SET last_poll_at = now(), last_poll_ok = FALSE,
                   last_error_code = 'starting', last_worker_at = NULL WHERE singleton = TRUE"""
            )

    def persist_updates(self, updates: list[dict]) -> int | None:
        if not updates:
            return self.offset()
        next_offset = max(int(update["update_id"]) for update in updates) + 1
        with self._db() as connection, connection.cursor() as cursor:
            for update in updates:
                update_id = int(update["update_id"])
                payload = json.dumps(update, ensure_ascii=False)
                cursor.execute(
                    """INSERT INTO inbox (update_id, payload)
                       VALUES (%s, %s::jsonb) ON CONFLICT (update_id) DO NOTHING""",
                    (update_id, payload),
                )
                message = update.get("message") or {}
                sender = message.get("from") or {}
                chat_id = (message.get("chat") or {}).get("id")
                text = message.get("text")
                if sender.get("id") == self.allowed_user_id and chat_id and isinstance(text, str):
                    cursor.execute(
                        """INSERT INTO messages
                           (chat_id, update_id, role, content, route)
                           VALUES (%s, %s, 'user', %s, NULL)
                           ON CONFLICT (update_id, role) DO NOTHING""",
                        (chat_id, update_id, text),
                    )
            cursor.execute(
                """UPDATE bot_state SET next_offset = GREATEST(
                     COALESCE(next_offset, 0), %s), last_update_at = now()
                   WHERE singleton = TRUE""",
                (next_offset,),
            )
        return next_offset

    def persist_webhook_update(self, update: dict) -> bool:
        """Persist one webhook delivery without touching the polling offset.

        The inbox primary key makes Telegram retries idempotent. A late update
        is retained even if newer IDs have already been processed; ordering is
        therefore only guaranteed among updates concurrently present in inbox.
        """
        update_id = update["update_id"]
        payload = json.dumps(update, ensure_ascii=False)
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO inbox (update_id, payload) VALUES (%s, %s::jsonb)
                   ON CONFLICT (update_id) DO NOTHING RETURNING update_id""",
                (update_id, payload),
            )
            inserted = cursor.fetchone() is not None
            message = update.get("message")
            message = message if isinstance(message, dict) else {}
            sender = message.get("from")
            sender = sender if isinstance(sender, dict) else {}
            chat = message.get("chat")
            chat = chat if isinstance(chat, dict) else {}
            chat_id = chat.get("id")
            text = message.get("text")
            if inserted and sender.get("id") == self.allowed_user_id and chat_id and isinstance(text, str):
                cursor.execute(
                    """INSERT INTO messages (chat_id, update_id, role, content, route)
                       VALUES (%s, %s, 'user', %s, NULL)
                       ON CONFLICT (update_id, role) DO NOTHING""",
                    (chat_id, update_id, text),
                )
            if inserted:
                cursor.execute("UPDATE bot_state SET last_update_at = now() WHERE singleton = TRUE")
        return inserted

    def save_transcription(
        self, update_id: int, chat_id: int, transcript: str,
        claim_token: uuid.UUID, generation: int = 0,
    ) -> bool:
        """Cache a voice transcript and add it to history under the active claim."""
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE inbox
                   SET payload = jsonb_set(
                       payload, '{_klodd}',
                       COALESCE(payload -> '_klodd', '{}'::jsonb)
                           || jsonb_build_object('transcription', %s::text),
                       true)
                   WHERE update_id = %s AND status = 'processing'
                     AND claim_token = %s AND generation = %s
                   RETURNING update_id""",
                (transcript, update_id, claim_token, generation),
            )
            if cursor.fetchone() is None:
                return False
            cursor.execute(
                """INSERT INTO messages (chat_id, update_id, role, content, route)
                   VALUES (%s, %s, 'user', %s, NULL)
                   ON CONFLICT (update_id, role) DO UPDATE SET content = EXCLUDED.content""",
                (chat_id, update_id, transcript),
            )
            return True

    def recover_processing(self) -> None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE inbox SET status = 'retry', processing_at = NULL, claim_token = NULL, next_attempt_at = now() WHERE status = 'processing'"
            )
            cursor.execute(
                "UPDATE outbox SET status = 'pending', sending_at = NULL, claim_token = NULL, next_attempt_at = now() WHERE status = 'sending'"
            )

    def recover_stale(self) -> None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE inbox SET status = 'retry', processing_at = NULL,
                   claim_token = NULL, next_attempt_at = now() WHERE status = 'processing'
                   AND (processing_at IS NULL OR
                        processing_at < now() - (%s * interval '1 second'))""",
                (INBOX_STALE_SECONDS,),
            )
            cursor.execute(
                """UPDATE outbox SET status = 'pending', sending_at = NULL,
                   claim_token = NULL, next_attempt_at = now() WHERE status = 'sending'
                   AND (sending_at IS NULL OR
                        sending_at < now() - (%s * interval '1 second'))""",
                (OUTBOX_STALE_SECONDS,),
            )

    def claim_inbox(self) -> InboxItem | None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT update_id, payload, attempts, generation FROM inbox
                   WHERE status IN ('received', 'retry') AND next_attempt_at <= now()
                     AND NOT EXISTS (
                       SELECT 1 FROM inbox earlier
                       WHERE earlier.update_id < inbox.update_id
                         AND earlier.status IN ('received', 'retry', 'processing')
                         AND earlier.payload #>> '{message,chat,id}' =
                             inbox.payload #>> '{message,chat,id}'
                     )
                   ORDER BY update_id LIMIT 1 FOR UPDATE SKIP LOCKED"""
            )
            row = cursor.fetchone()
            if not row:
                return None
            update_id, payload, attempts, generation = row
            attempts += 1
            claim_token = uuid.uuid4()
            cursor.execute(
                "UPDATE inbox SET status = 'processing', processing_at = now(), attempts = %s, claim_token = %s WHERE update_id = %s",
                (attempts, claim_token, update_id),
            )
            return InboxItem(update_id, payload, attempts, claim_token, generation)

    def history(self, chat_id: int, before_update_id: int, limit: int = 10) -> list[dict]:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT role, content, update_id FROM messages
                   WHERE chat_id = %s AND update_id < %s
                   ORDER BY update_id DESC,
                     CASE role WHEN 'assistant' THEN 0 WHEN 'user' THEN 1 ELSE 2 END ASC
                   LIMIT %s""",
                (chat_id, before_update_id, limit),
            )
            return [
                {"role": role, "text": content}
                for role, content, _ in reversed(cursor.fetchall())
            ]

    def complete_inbox(
        self, update_id: int, chat_id: int, route: str, answer: str,
        claim_token: uuid.UUID, generation: int = 0,
    ) -> bool:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE inbox SET status = 'completed', processing_at = NULL,
                   claim_token = NULL, processed_at = now()
                   WHERE update_id = %s AND status = 'processing' AND claim_token = %s
                   RETURNING update_id""",
                (update_id, claim_token),
            )
            if cursor.fetchone() is None:
                return False
            cursor.execute(
                "UPDATE messages SET route = %s WHERE update_id = %s AND role = 'user'",
                (route, update_id),
            )
            cursor.execute(
                """INSERT INTO messages (chat_id, update_id, role, content, route)
                   VALUES (%s, %s, 'assistant', %s, %s)
                   ON CONFLICT (update_id, role) DO UPDATE SET content = EXCLUDED.content, route = EXCLUDED.route""",
                (chat_id, update_id, answer, route),
            )
            for part_no, start in enumerate(range(0, len(answer), 4000)):
                cursor.execute(
                    """INSERT INTO outbox (update_id, generation, part_no, chat_id, text)
                       VALUES (%s, %s, %s, %s, %s) ON CONFLICT (update_id, generation, part_no) DO NOTHING""",
                    (update_id, generation, part_no, chat_id, answer[start:start + 4000]),
                )
            cursor.execute("UPDATE bot_state SET last_inbox_progress_at = now() WHERE singleton = TRUE")
            return True

    def ignore_inbox(self, update_id: int, claim_token: uuid.UUID) -> bool:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE inbox SET status = 'ignored', processing_at = NULL,
                   claim_token = NULL, processed_at = now()
                   WHERE update_id = %s AND status = 'processing' AND claim_token = %s
                   RETURNING update_id""",
                (update_id, claim_token),
            )
            if cursor.fetchone() is None:
                return False
            cursor.execute("UPDATE bot_state SET last_inbox_progress_at = now() WHERE singleton = TRUE")
            return True

    def retry_inbox(
        self, update_id: int, attempts: int, error_code: str, claim_token: uuid.UUID,
        terminal_text: str | None = None, chat_id: int | None = None, generation: int = 0,
    ) -> bool:
        terminal = attempts >= MAX_ATTEMPTS
        with self._db() as connection, connection.cursor() as cursor:
            if terminal:
                cursor.execute(
                    """UPDATE inbox SET status = 'dead_letter', processing_at = NULL, last_error_code = %s,
                       claim_token = NULL, processed_at = now()
                       WHERE update_id = %s AND status = 'processing' AND claim_token = %s
                       RETURNING update_id""",
                    (error_code, update_id, claim_token),
                )
                if cursor.fetchone() is None:
                    return False
                if terminal_text and chat_id:
                    self._insert_terminal_response(cursor, update_id, chat_id, terminal_text, generation)
            else:
                delay = retry_delay(attempts)
                cursor.execute(
                    """UPDATE inbox SET status = 'retry', processing_at = NULL, last_error_code = %s,
                       claim_token = NULL, next_attempt_at = now() + (%s * interval '1 second')
                       WHERE update_id = %s AND status = 'processing' AND claim_token = %s
                       RETURNING update_id""",
                    (error_code, delay, update_id, claim_token),
                )
                if cursor.fetchone() is None:
                    return False
            cursor.execute("UPDATE bot_state SET last_inbox_progress_at = now() WHERE singleton = TRUE")
            return True

    def permanent_inbox_error(
        self, update_id: int, chat_id: int, text: str, error_code: str,
        claim_token: uuid.UUID, generation: int = 0,
    ) -> bool:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE inbox SET status = 'handled_error', processing_at = NULL, last_error_code = %s,
                   claim_token = NULL, processed_at = now()
                   WHERE update_id = %s AND status = 'processing' AND claim_token = %s
                   RETURNING update_id""",
                (error_code, update_id, claim_token),
            )
            if cursor.fetchone() is None:
                return False
            self._insert_terminal_response(cursor, update_id, chat_id, text, generation)
            cursor.execute("UPDATE bot_state SET last_inbox_progress_at = now() WHERE singleton = TRUE")
            return True

    @staticmethod
    def _insert_terminal_response(cursor, update_id: int, chat_id: int, text: str, generation: int = 0) -> None:
        cursor.execute(
            """INSERT INTO messages (chat_id, update_id, role, content, route)
               VALUES (%s, %s, 'assistant', %s, 'error')
               ON CONFLICT (update_id, role) DO UPDATE SET content = EXCLUDED.content, route = EXCLUDED.route""",
            (chat_id, update_id, text),
        )
        for part_no, start in enumerate(range(0, len(text), 4000)):
            cursor.execute(
                """INSERT INTO outbox (update_id, generation, part_no, chat_id, text)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT (update_id, generation, part_no) DO NOTHING""",
                (update_id, generation, part_no, chat_id, text[start:start + 4000]),
            )

    def claim_outbox(self) -> OutboxItem | None:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT id, chat_id, text, attempts FROM outbox o
                   WHERE status = 'pending' AND next_attempt_at <= now()
                     AND NOT EXISTS (
                       SELECT 1 FROM outbox earlier
                       WHERE earlier.chat_id = o.chat_id AND earlier.id < o.id
                         AND earlier.status IN ('pending', 'sending')
                     )
                   ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"""
            )
            row = cursor.fetchone()
            if not row:
                return None
            item_id, chat_id, text, attempts = row
            attempts += 1
            claim_token = uuid.uuid4()
            cursor.execute(
                """UPDATE outbox SET status = 'sending', sending_at = now(), attempts = %s,
                   claim_token = %s WHERE id = %s AND status = 'pending' RETURNING id""",
                (attempts, claim_token, item_id),
            )
            if cursor.fetchone() is None:
                return None
            return OutboxItem(item_id, chat_id, text, attempts, claim_token)

    def mark_outbox_sent(self, item_id: int, claim_token: uuid.UUID) -> bool:
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE outbox SET status = 'sent', sending_at = NULL, sent_at = now(),
                   claim_token = NULL
                   WHERE id = %s AND status = 'sending' AND claim_token = %s
                   RETURNING id""",
                (item_id, claim_token),
            )
            if cursor.fetchone() is None:
                return False
            cursor.execute("UPDATE bot_state SET last_outbox_progress_at = now() WHERE singleton = TRUE")
            return True

    def outbox_is_current(self, item_id: int, claim_token: uuid.UUID) -> bool:
        """Only send a claim that is still owned and in the current generation."""
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT EXISTS (
                     SELECT 1 FROM outbox o JOIN inbox i USING (update_id)
                     WHERE o.id = %s AND o.status = 'sending'
                       AND o.claim_token = %s
                       AND o.generation = i.generation
                   )""",
                (item_id, claim_token),
            )
            row = cursor.fetchone()
            return bool(row and row[0])

    def deliver_outbox(self, item: OutboxItem, send_callback) -> bool:
        """Fence external delivery from stale recovery until its result commits."""
        send_error = None
        delivered = False
        with self._db() as connection, connection.cursor() as cursor:
            # Keep lock order consistent with inbox retry, which locks inbox
            # before superseding its outbox rows.
            cursor.execute(
                """SELECT i.generation FROM inbox i
                   JOIN outbox o USING (update_id) WHERE o.id = %s FOR UPDATE OF i""",
                (item.id,),
            )
            inbox_row = cursor.fetchone()
            if inbox_row is None:
                return False
            generation = inbox_row[0]
            cursor.execute(
                """SELECT status, claim_token, generation FROM outbox
                   WHERE id = %s FOR UPDATE""",
                (item.id,),
            )
            outbox_row = cursor.fetchone()
            if (outbox_row is None or outbox_row[0] != "sending"
                    or outbox_row[1] != item.claim_token
                    or outbox_row[2] != generation):
                return False

            try:
                send_callback(item.chat_id, item.text)
            except BaseException as exc:
                # Preserve Telegram's exception type for worker retry policy;
                # _db intentionally translates exceptions raised in its body.
                send_error = exc
                connection.rollback()
            else:
                cursor.execute(
                    """UPDATE outbox SET status = 'sent', sending_at = NULL, sent_at = now(),
                       claim_token = NULL WHERE id = %s AND status = 'sending'
                       AND claim_token = %s AND generation = %s RETURNING id""",
                    (item.id, item.claim_token, generation),
                )
                if cursor.fetchone() is not None:
                    cursor.execute("UPDATE bot_state SET last_outbox_progress_at = now() WHERE singleton = TRUE")
                    delivered = True
        if send_error is not None:
            raise send_error
        return delivered

    def retry_outbox(
        self,
        item_id: int,
        attempts: int,
        error_code: str,
        claim_token: uuid.UUID,
        retry_after: int | None = None,
        permanent: bool = False,
    ) -> bool:
        terminal = permanent or attempts >= MAX_ATTEMPTS
        with self._db() as connection, connection.cursor() as cursor:
            if terminal:
                cursor.execute(
                    """UPDATE outbox SET status = 'dead_letter', sending_at = NULL,
                       claim_token = NULL, last_error_code = %s
                       WHERE id = %s AND status = 'sending' AND claim_token = %s
                       RETURNING id""",
                    (error_code, item_id, claim_token),
                )
            else:
                delay = retry_after if retry_after is not None else retry_delay(attempts)
                cursor.execute(
                    """UPDATE outbox SET status = 'pending', sending_at = NULL, last_error_code = %s,
                       claim_token = NULL, next_attempt_at = now() + (%s * interval '1 second')
                       WHERE id = %s AND status = 'sending' AND claim_token = %s
                       RETURNING id""",
                    (error_code, max(1, delay), item_id, claim_token),
                )
            if cursor.fetchone() is None:
                return False
            cursor.execute("UPDATE bot_state SET last_outbox_progress_at = now() WHERE singleton = TRUE")
            return True

    def requeue_dead_letter(
        self, kind: str, item_id: int, chat_id: int, *,
        command_update_id: int | None = None, claim_token: uuid.UUID | None = None,
    ) -> bool:
        if kind not in ("inbox", "outbox"):
            return False
        with self._db() as connection, connection.cursor() as cursor:
            # Retry is a side effect of the currently claimed command update.
            # Lock it so recovery/reclaim cannot invalidate ownership mid-operation.
            if command_update_id is None or claim_token is None:
                return False
            cursor.execute(
                """SELECT update_id FROM inbox
                   WHERE update_id = %s AND status = 'processing' AND claim_token = %s
                   FOR UPDATE""",
                (command_update_id, claim_token),
            )
            if cursor.fetchone() is None:
                return False
            if kind == "inbox":
                # /retry inbox means regenerate a new answer. Keep the old
                # answer in an audit table and retain sent delivery records.
                # Mark pending/claimed parts obsolete; an HTTP request already
                # in flight cannot be recalled, so the worker also revalidates
                # generation immediately before starting sendMessage.
                cursor.execute(
                    """SELECT generation FROM inbox WHERE update_id = %s AND status = 'dead_letter'
                       AND payload #>> '{message,chat,id}' = %s FOR UPDATE""",
                    (item_id, str(chat_id)),
                )
                row = cursor.fetchone()
                if row is None:
                    return False
                generation = row[0]
                cursor.execute(
                    """INSERT INTO message_revisions (update_id, chat_id, content, generation)
                       SELECT update_id, chat_id, content, %s FROM messages
                       WHERE update_id = %s AND role = 'assistant'""",
                    (generation, item_id),
                )
                cursor.execute(
                    """UPDATE outbox SET status = 'dead_letter', sending_at = NULL,
                       claim_token = NULL, last_error_code = 'superseded_by_inbox_retry'
                       WHERE update_id = %s AND generation = %s AND status IN ('pending', 'sending')""",
                    (item_id, generation),
                )
                cursor.execute(
                    """UPDATE inbox SET status = 'retry', attempts = 0, generation = generation + 1,
                       processing_at = NULL, last_error_code = NULL, next_attempt_at = now(), processed_at = NULL
                       WHERE update_id = %s AND status = 'dead_letter'
                       RETURNING update_id""",
                    (item_id,),
                )
                changed = cursor.fetchone() is not None
            else:
                cursor.execute(
                    """UPDATE outbox SET status = 'pending', attempts = 0, sending_at = NULL,
                       claim_token = NULL, last_error_code = NULL, next_attempt_at = now()
                       WHERE id = %s AND chat_id = %s AND status = 'dead_letter'
                         AND last_error_code IS DISTINCT FROM 'superseded_by_inbox_retry'
                       RETURNING id""",
                    (item_id, chat_id),
                )
                changed = cursor.fetchone() is not None
            if changed:
                cursor.execute(
                    "UPDATE bot_state SET last_inbox_progress_at = now(), last_outbox_progress_at = now() WHERE singleton = TRUE"
                )
            return changed

    def import_legacy_history(self, path: str | Path) -> str:
        source = Path(path)
        if not source.is_file():
            return "absent"
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            history_by_chat = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HistoryImportError("Legacy history file is invalid JSON") from None
        if not isinstance(history_by_chat, dict):
            raise HistoryImportError("Legacy history has an invalid format")

        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT source_hash FROM data_imports WHERE import_key = 'chat_memory_v1'")
            if cursor.fetchone():
                return "already_imported"

            valid_histories = []
            total = 0
            for chat_key, items in history_by_chat.items():
                try:
                    chat_id = int(chat_key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(items, list):
                    continue
                valid = [
                    (item["role"], item["text"])
                    for item in items
                    if isinstance(item, dict)
                    and item.get("role") in ("user", "assistant")
                    and isinstance(item.get("text"), str)
                ]
                valid_histories.append((chat_id, valid))
                total += len(valid)

            # Imported IDs are negative, ordered, and cannot collide with Telegram IDs.
            next_id = -1
            for chat_id, items in sorted(valid_histories):
                first_id = next_id - len(items) + 1
                for index, (role, content) in enumerate(items):
                    cursor.execute(
                        """INSERT INTO messages (chat_id, update_id, role, content, route)
                           VALUES (%s, %s, %s, %s, 'legacy')
                           ON CONFLICT (update_id, role) DO NOTHING""",
                        (chat_id, first_id + index, role, content),
                    )
                next_id = first_id - 1
            cursor.execute(
                """INSERT INTO data_imports (import_key, source_hash, row_count)
                   VALUES ('chat_memory_v1', %s, %s)""",
                (digest, total),
            )
        return "imported"

    def health(self, mode: str = "polling") -> dict:
        if mode not in ("polling", "webhook"):
            raise ValueError("mode must be polling or webhook")
        with self._db() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT last_poll_at, last_poll_ok, last_update_at,
                          last_worker_at, last_inbox_progress_at, last_outbox_progress_at,
                          (SELECT count(*) FROM inbox WHERE status IN ('received','retry','processing')),
                          (SELECT count(*) FROM outbox WHERE status IN ('pending','sending')),
                          (SELECT count(*) FROM inbox WHERE status = 'dead_letter'),
                          (SELECT count(*) FROM outbox WHERE status = 'dead_letter'),
                          EXISTS (SELECT 1 FROM inbox WHERE status = 'processing'
                                  AND processing_at < now() - (%s * interval '1 second')),
                          EXISTS (SELECT 1 FROM outbox WHERE status = 'sending'
                                  AND sending_at < now() - (%s * interval '1 second')),
                          EXISTS (SELECT 1 FROM inbox WHERE status IN ('received','retry')
                                  AND next_attempt_at <= now()
                                  AND received_at < now() - (%s * interval '1 second')),
                          EXISTS (SELECT 1 FROM outbox WHERE status = 'pending'
                                  AND next_attempt_at <= now()
                                  AND created_at < now() - (%s * interval '1 second'))
                   FROM bot_state WHERE singleton = TRUE""",
                (INBOX_STALE_SECONDS, OUTBOX_STALE_SECONDS,
                 INBOX_STALE_SECONDS, OUTBOX_STALE_SECONDS),
            )
            row = cursor.fetchone()
            if not row:
                return {"ok": False, "database": True, "mode": mode, "polling": False}
            (
                last_poll_at, poll_ok, last_update, worker_at, inbox_progress,
                outbox_progress, inbox_count, outbox_count, inbox_dead, outbox_dead,
                inbox_processing_stuck, outbox_sending_stuck, inbox_queue_stuck,
                outbox_queue_stuck,
            ) = row
            poll_fresh = last_poll_at is not None and (time.time() - last_poll_at.timestamp()) < 90
            worker_fresh = worker_at is not None and (time.time() - worker_at.timestamp()) < WORKER_STALE_SECONDS
            inbox_stuck = bool(inbox_processing_stuck or inbox_queue_stuck)
            outbox_stuck = bool(outbox_sending_stuck or outbox_queue_stuck)
            runtime_ok = worker_fresh and not inbox_stuck and not outbox_stuck
            if mode == "polling":
                runtime_ok = runtime_ok and poll_ok and poll_fresh
            return {
                "ok": bool(runtime_ok),
                "database": True,
                "mode": mode,
                "polling": bool(poll_ok and poll_fresh),
                "worker": worker_fresh,
                "inbox_stuck": inbox_stuck,
                "outbox_stuck": outbox_stuck,
                "last_update_at": last_update.isoformat() if last_update else None,
                "last_inbox_progress_at": inbox_progress.isoformat() if inbox_progress else None,
                "last_outbox_progress_at": outbox_progress.isoformat() if outbox_progress else None,
                "inbox_pending": inbox_count,
                "outbox_pending": outbox_count,
                "inbox_dead_letter": inbox_dead,
                "outbox_dead_letter": outbox_dead,
            }

from assistant.storage import PostgresStore


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []
        self.rowcount = -1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.connection.queries.append((query, params))
        if query.lstrip().startswith("SELECT i.update_id"):
            self.rows = self.connection.eligible_ids
        elif "DELETE FROM outbox" in query:
            self.rowcount = self.connection.deleted_outbox
        elif "DELETE FROM inbox" in query:
            self.rows = [(item_id,) for item_id in self.connection.eligible_ids]

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, eligible_ids, deleted_outbox=0):
        self.eligible_ids = [(item_id,) for item_id in eligible_ids]
        self.deleted_outbox = deleted_outbox
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return FakeCursor(self)


def test_cleanup_deletes_sent_outbox_before_terminal_inbox():
    connection = FakeConnection([101, 102], deleted_outbox=3)
    store = PostgresStore(connect=lambda: connection)

    result = store.cleanup_retention()

    assert result == {"inbox": 2, "outbox": 3}
    assert len(connection.queries) == 3
    assert "FOR UPDATE OF i SKIP LOCKED" in connection.queries[0][0]
    assert "status IN ('completed', 'ignored')" in connection.queries[0][0]
    assert "interval '30 days'" in connection.queries[0][0]
    assert "DELETE FROM outbox" in connection.queries[1][0]
    assert "DELETE FROM inbox" in connection.queries[2][0]


def test_cleanup_does_nothing_when_no_rows_are_eligible():
    connection = FakeConnection([])
    store = PostgresStore(connect=lambda: connection)

    result = store.cleanup_retention()

    assert result == {"inbox": 0, "outbox": 0}
    assert len(connection.queries) == 1
    assert connection.queries[0][0].lstrip().startswith("SELECT i.update_id")

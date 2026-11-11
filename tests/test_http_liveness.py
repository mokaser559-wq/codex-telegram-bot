import urllib.error
import urllib.request

import bridge


class UnreadyStore:
    def health(self, mode="polling"):
        return {"ok": False, "database": False, "polling": False}


def test_root_liveness_is_available_while_database_is_unready(monkeypatch):
    monkeypatch.setenv("PORT", "0")
    server = bridge.start_http_server(UnreadyStore())
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
            assert response.status == 200
            assert response.read() == b"ok"

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 503
        else:
            raise AssertionError("/health should report that the database is not ready")
    finally:
        bridge.shutdown_http_server(server)

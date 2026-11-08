import hmac
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from assistant.stt import STTError, transcribe_audio

TOKEN = os.environ.get("WHISPER_TEST_TOKEN", "")
PORT = int(os.environ.get("PORT", "10000"))
MAX_SIZE = 20 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in ("/", "/health"):
            self.respond(404, {"error": "not_found"})
            return
        self.respond(200, {"status": "ok", "service": "whisper-test"})

    def do_POST(self):
        if self.path != "/transcribe":
            self.respond(404, {"error": "not_found"})
            return

        supplied = self.headers.get("Authorization", "")
        expected = "Bearer " + TOKEN
        if not TOKEN or not hmac.compare_digest(supplied, expected):
            self.respond(401, {"error": "unauthorized"})
            return

        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.respond(400, {"error": "invalid_content_length"})
            return

        if size <= 0 or size > MAX_SIZE:
            self.respond(413, {"error": "file_size_limit"})
            return

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as audio:
                temporary_path = Path(audio.name)
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        self.respond(400, {"error": "incomplete_upload"})
                        return
                    audio.write(chunk)
                    remaining -= len(chunk)

            transcript = transcribe_audio(temporary_path)
            self.respond(200, {"transcript": transcript})
        except STTError as exc:
            self.respond(422, {"error": str(exc)})
        except Exception:
            self.respond(500, {"error": "internal_error"})
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def log_message(self, format, *args):
        print("HTTP:", args[0] if args else "request")


print("Starting protected Whisper test server on port", PORT, flush=True)
HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

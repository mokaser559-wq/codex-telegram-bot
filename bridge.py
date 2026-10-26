import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE = Path.home() / ".local/share/codex-telegram"
TOKEN_FILE = BASE / "bot_token"
STATE_FILE = BASE / "update_offset"
ALLOWED_USER_ID = 1193680242
_LOCAL_CODEX = "/home/rozzen/.local/bin/codex"
CODEX = _LOCAL_CODEX if os.path.exists(_LOCAL_CODEX) else (shutil.which("codex") or "codex")
SANDBOX = os.environ.get("CODEX_SANDBOX", "workspace-write")
API = "https://api.telegram.org/bot" + TOKEN_FILE.read_text().strip()
TIMEOUT = 300
MEMORY_FILE = BASE / "chat_memory.json"
MEMORY_LIMIT = 10

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


def start_ping_server():
    port = os.environ.get("PORT")
    if not port:
        return
    server = HTTPServer(("0.0.0.0", int(port)), Ping)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Ping-сервер запущен на порту", port, flush=True)


def telegram(method, data=None, timeout=40):
    body = urllib.parse.urlencode(data or {}).encode()
    req = urllib.request.Request(API + "/" + method, data=body)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError("Telegram API returned an error")
    return result["result"]


def send(chat_id, text):
    text = text.strip() or "(Codex не вернул текстовый ответ)"
    for start in range(0, len(text), 4000):
        telegram("sendMessage", {
            "chat_id": chat_id,
            "text": text[start:start + 4000]
        })


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
        if result.returncode != 0 and not answer:
            error = (result.stderr or result.stdout or "Неизвестная ошибка").strip()
            return "Codex завершился с ошибкой:\n" + error[-2500:]
        return answer or "Codex завершился без текстового ответа."


def main():
    if not TOKEN_FILE.is_file():
        raise SystemExit("Не найден файл bot_token")
    os.chmod(TOKEN_FILE, 0o600)
    start_ping_server()

    offset = int(STATE_FILE.read_text().strip()) if STATE_FILE.exists() else None
    print("Telegram-Codex bridge запущен. Разрешённый ID:", ALLOWED_USER_ID, flush=True)

    while True:
        try:
            params = {"timeout": 25}
            if offset is not None:
                params["offset"] = offset

            updates = telegram("getUpdates", params, timeout=35)

            for update in updates:
                offset = update["update_id"] + 1
                STATE_FILE.write_text(str(offset))
                os.chmod(STATE_FILE, 0o600)

                message = update.get("message", {})
                user = message.get("from", {})
                text = message.get("text", "").strip()
                chat_id = message.get("chat", {}).get("id")

                if user.get("id") != ALLOWED_USER_ID or not text or not chat_id:
                    continue

                if text == "/start":
                    send(chat_id, "Привет! 😊 Я на связи. Пиши, что у тебя на уме — разберёмся вместе.")
                    continue

                if text == "/help":
                    send(chat_id, "Пиши как другу 🙂 Могу поболтать, помочь разобраться в вопросе или взяться за задачу на компьютере.")
                    continue

                try:
                    memory = json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else {}
                    history = memory.get(str(chat_id), [])
                    send(chat_id, "Секундочку, сейчас посмотрю 😊")
                    answer = ask_codex(text, history[-MEMORY_LIMIT:])
                    history.extend([
                        {"role": "user", "text": text},
                        {"role": "assistant", "text": answer}
                    ])
                    memory[str(chat_id)] = history[-MEMORY_LIMIT:]
                    MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=False, indent=2))
                    os.chmod(MEMORY_FILE, 0o600)
                    send(chat_id, answer)
                except subprocess.TimeoutExpired:
                    send(chat_id, "Codex не завершил задание за 5 минут. Попробуй более короткий запрос.")
                except Exception as exc:
                    print("Ошибка обработки задания:", repr(exc), flush=True)
                    send(chat_id, "Не удалось выполнить задание. Подробности доступны в локальном терминале.")

        except KeyboardInterrupt:
            print("\nМост остановлен.", flush=True)
            break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            print("Временная ошибка:", repr(exc), flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()

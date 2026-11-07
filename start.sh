set -e
if [ -z "${DATABASE_URL:-}" ]; then
  echo "Required environment variable is missing: DATABASE_URL" >&2
  exit 1
fi

require_secret_file() {
  if [ ! -r "$1" ] || [ ! -s "$1" ]; then
    echo "Required Secret File is missing, empty, or unreadable: ${1##*/}" >&2
    exit 1
  fi
}

require_secret_file /etc/secrets/auth.json
require_secret_file /etc/secrets/credentials.json
require_secret_file /etc/secrets/config.toml
require_secret_file /etc/secrets/bot_token

mkdir -p "$HOME/.codex" "$HOME/.local/share/codex-telegram"
cp /etc/secrets/auth.json "$HOME/.codex/auth.json"
cp /etc/secrets/credentials.json "$HOME/.codex/.credentials.json"
cp /etc/secrets/config.toml "$HOME/.codex/config.toml"
cp /etc/secrets/bot_token "$HOME/.local/share/codex-telegram/bot_token"
export BOT_TOKEN_FILE="$HOME/.local/share/codex-telegram/bot_token"
chmod 600 "$HOME/.local/share/codex-telegram/bot_token"
chmod 600 "$HOME/.codex/auth.json" "$HOME/.codex/.credentials.json" "$HOME/.codex/config.toml"
exec python3 bridge.py

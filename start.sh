set -e
mkdir -p "$HOME/.codex" "$HOME/.local/share/codex-telegram"
cp /etc/secrets/auth.json "$HOME/.codex/auth.json"
cp /etc/secrets/credentials.json "$HOME/.codex/.credentials.json"
cp /etc/secrets/config.toml "$HOME/.codex/config.toml"
cp /etc/secrets/bot_token "$HOME/.local/share/codex-telegram/bot_token"
chmod 600 "$HOME/.codex/auth.json" "$HOME/.codex/.credentials.json" "$HOME/.codex/config.toml"
exec python3 bridge.py

#!/bin/sh
set -eu

whisper_cli="${WHISPER_CLI:-/usr/local/bin/whisper-cli}"
model_path="${WHISPER_MODEL_PATH:-/opt/whisper/models/ggml-tiny-q5_1.bin}"
port="${PORT:-10000}"

printf '%s\n' '=== Whisper diagnostics ==='
if [ -x "$whisper_cli" ]; then
  printf 'whisper-cli: available (%s)\n' "$whisper_cli"
else
  printf 'whisper-cli: missing or not executable (%s)\n' "$whisper_cli"
fi

if [ -f "$model_path" ]; then
  model_size=$(wc -c < "$model_path" | tr -d ' ')
  printf 'Model: present (%s), %s bytes\n' "$model_path" "$model_size"
else
  printf 'Model: missing (%s)\n' "$model_path"
fi

printf '%s\n' 'Available memory:'
if [ -r /proc/meminfo ]; then
  awk '/^(MemTotal|MemAvailable):/ { print }' /proc/meminfo
else
  printf '%s\n' '/proc/meminfo is unavailable'
fi

printf '%s\n' 'whisper-cli --help output:'
if [ -x "$whisper_cli" ]; then
  if "$whisper_cli" --help; then
    printf '%s\n' 'whisper-cli --help exited successfully'
  else
    help_status=$?
    printf 'whisper-cli --help exited with status %s\n' "$help_status"
  fi
else
  printf '%s\n' 'Skipped because whisper-cli is unavailable'
fi

printf 'Starting HTTP server on 0.0.0.0:%s\n' "$port"
exec python3 -m http.server "$port" --bind 0.0.0.0

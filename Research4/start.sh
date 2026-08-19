#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p output/launcher

pid_file="output/launcher/large.pid"
log_file="output/launcher/large.log"

if [[ -f "$pid_file" ]]; then
  old_pid="$(tr -d '[:space:]' < "$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "large suite is already running | PID $old_pid | log $log_file"
    exit 0
  fi
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
nohup python -u main.py run --size large >> "$log_file" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$pid_file"

echo "large suite started | PID $launcher_pid"
echo "log    | $log_file"
echo "status | bash status.sh"

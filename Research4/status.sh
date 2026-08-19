#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
pid_file="output/launcher/large.pid"
state_file="output/suites/large/state.json"
log_file="output/launcher/large.log"

if [[ -f "$pid_file" ]]; then
  launcher_pid="$(tr -d '[:space:]' < "$pid_file")"
  if [[ "$launcher_pid" =~ ^[0-9]+$ ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    echo "process | running | PID $launcher_pid"
  else
    echo "process | not running | last PID $launcher_pid"
  fi
else
  echo "process | not started"
fi

if [[ -f "$state_file" ]]; then
  python -c "import json; p=json.load(open('$state_file', encoding='utf-8')); print('suite   |', p.get('status')); [print(f\"phase   | {k:<24} | {v.get('status')}\") for k,v in p.get('phases',{}).items()]"
else
  echo "suite   | no state yet"
fi

echo "log     | $log_file"
if [[ -f "$log_file" ]]; then
  tail -n 12 "$log_file"
fi

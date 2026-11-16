#!/bin/sh
set -e
cd -- "$(dirname -- "$0")"
if [ -x "../../../.venv/bin/python" ]; then
    exec "../../../.venv/bin/python" CapPeeler.py "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 CapPeeler.py "$@"
else
    exec python CapPeeler.py "$@"
fi

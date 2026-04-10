#!/usr/bin/env bash
set -e

MODEL_PATH=""
MESSAGE=""

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
	echo "Usage: bash ./inference.sh <message> <model_path>"
	exit 0
fi

if [ -n "$1" ]; then
	MESSAGE="$1"
else
	echo "Error: missing required message argument."
	echo "Usage: bash ./inference.sh <message> <model_path>"
	exit 1
fi

if [ -n "$2" ]; then
	MODEL_PATH="$2"
else
	echo "Error: missing required model_path argument."
	echo "Usage: bash ./inference.sh <message> <model_path>"
	exit 1
fi

if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/Scripts/python.exe" ]; then
	PYTHON_BIN="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
	PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "./venv/Scripts/python.exe" ]; then
	PYTHON_BIN="./venv/Scripts/python.exe"
elif [ -x "./venv/bin/python" ]; then
	PYTHON_BIN="./venv/bin/python"
elif command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
else
	echo "Error: Python not found. Activate venv or install Python." >&2
	exit 1
fi

"$PYTHON_BIN" scripts/inference.py --model_path "$MODEL_PATH" --message "$MESSAGE"
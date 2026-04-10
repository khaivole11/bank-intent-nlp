#!/usr/bin/env bash
set -e

CONFIG_PATH="configs/train.yaml"
MODEL_NAME="${1:-}"
MAX_LENGTH="${2:-}"

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
	echo "Usage: bash ./train.sh [model_name] [max_length]"
	echo "Run with config defaults: bash ./train.sh"
	exit 0
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

CMD=("$PYTHON_BIN" scripts/train.py --config "$CONFIG_PATH")

if [ -n "$MODEL_NAME" ]; then
	CMD+=(--model_name "$MODEL_NAME")
fi

if [ -n "$MAX_LENGTH" ]; then
	CMD+=(--max_length "$MAX_LENGTH")
fi

echo "=== Training Runner ==="
echo "Config file : $CONFIG_PATH"
if [ -n "$MODEL_NAME" ]; then
	echo "Model      : $MODEL_NAME (override)"
else
	echo "Model      : from config"
fi
if [ -n "$MAX_LENGTH" ]; then
	echo "Max length : $MAX_LENGTH (override)"
else
	echo "Max length : from config"
fi
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "=== Training Finished ==="
echo "Checkpoint : outputs/checkpoints/final"
echo "Metrics    : outputs/checkpoints/metrics.json"

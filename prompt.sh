#!/usr/bin/env bash
#
# prompt.sh – Entry-point for the harness.
#
# Runs harness/harness.py, which connects to a remote Ollama server
# over SSH, deploys tools.py to a VM's root dir, and opens an
# interactive chat session.
#
# Usage:
#   ./prompt.sh              # use settings from local.env
#   ./prompt.sh --model qwen2.5
#   ./prompt.sh --system "You are a helpful assistant."
#   ./prompt.sh myproject        # start the tools in <VM_ROOT_DIR>/myproject
#
# All arguments are passed through to harness/harness.py.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/harness/harness.py" "$@"

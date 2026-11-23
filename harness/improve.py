#!/usr/bin/env python3
"""
improve.py – Analyse a chat-transcript log for tool-use pain points.

Reads a JSON log produced by the harness (``_dump_transcript`` writes a
timestamped ``chat-*.json`` into ``LOG_DIR``: a list of the messages the
model saw and produced, in ``/api/chat`` shape -- ``system`` / ``user`` /
``assistant`` / ``tool``, where an assistant message may carry
``tool_calls`` and a ``tool`` message carries that call's result).  The
log is handed to the Ollama model, which is asked to find the pain points
where the *agent* struggled with tool use, and to report them.

Connection and model settings come from the same ``env/local.env`` the
harness reads (loaded at module level by :mod:`config`); the model is
reached over the same SSH tunnel the harness uses (``OllamaTunnel`` in
:mod:`remote`).  No VM tools server is started -- analysis only needs the
model, not the remote tools.

Unlike :mod:`harness`, which drives ``/api/chat``, this script talks to
Ollama's ``/api/generate`` endpoint: the transcript is serialized into a
single text *prompt* (with a separate *system* prompt) rather than sent as
a message list.

Usage
-----
    python improve.py chat-20260101-123456Z.json

The only argument is the path to a ``.json`` transcript log.  No override
flags: everything comes from ``local.env``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

# The env-driven configuration dataclasses (incl. the local.env loader) live in
# config.py; the SSH tunnel lives in remote.py -- the same modules harness.py
# uses, so improve.py reads the very same local.env values.
from config import (
    OllamaConfig,
    SSHConfig,
)
from remote import (
    OllamaTunnel,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The analyst persona and the catalogue of tool-use problems to look for.  Sent
# as the /api/generate "system" field so the model stays in "analyst" mode
# rather than treating the transcript as a task to continue.
_ANALYZER_SYSTEM_PROMPT = """\
Log is a message history of an AI agent. Find the pain points it had.
"""

# The transcript itself, framed as the thing to analyse.  The *log_text* slot
# is filled in per run.
_ANALYZER_PROMPT = """\
Transcript:
{log_text}
"""


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def _load_log(path: str) -> Any:
    """
    Read and parse the transcript *path*.

    Returns the parsed JSON (normally a list of message dicts).  Raises
    :class:`FileNotFoundError` if the path is missing and
    :class:`ValueError` if it is not valid JSON, so the caller can report a
    clear message instead of a traceback.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Ollama /api/generate
# ---------------------------------------------------------------------------

def _collect_response(body: str) -> str:
    """
    Return the full reply text from an ``/api/generate`` body.

    Works for both response shapes: a single JSON object (``stream: false``)
    and a newline-delimited stream of partial objects (``stream: true``), where
    each line carries a ``"response"`` fragment to concatenate.  This is what
    makes the call robust to Ollama defaulting to a streamed response.
    """
    body = body.strip()
    if not body:
        return ""
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        parts: list[str] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            part = json.loads(line)
            if isinstance(part, dict):
                parts.append(part.get("response", ""))
        return "".join(parts)
    return obj.get("response", "") if isinstance(obj, dict) else ""


def _generate(
    base_url: str,
    model: str,
    *,
    system: str,
    prompt: str,
    timeout: float,
) -> str:
    """
    Make one ``/api/generate`` call and return the reply.

    Unlike the harness's ``/api/chat``, the transcript is sent as a single text
    *prompt* (plus a *system* prompt) rather than a message list.  Ollama
    returns a single JSON object whose ``"response"`` key holds the full
    reply.
    """
    # Consume the streamed (default) response: Ollama returns a
    # newline-delimited stream of partial JSON objects, each carrying a
    # fragment of ``response``.  ``_collect_response`` joins those fragments
    # back into the full reply.
    payload: dict[str, Any] = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": True,
    }
    with requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
    ) as r:
        r.raise_for_status()
        response = _collect_response(r.text)
        print(response)
    return response


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse a chat-transcript .json log for tool-use pain "
                    "points via the Ollama /api/generate endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # The only argument is the transcript log.  No override flags: the model,
    # host, ports and key all come from local.env via config.py.
    p.add_argument(
        "log",
        help="Path to a chat-transcript .json log to analyse.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Read and render the transcript before opening the tunnel, so a bad path or
    # unparseable JSON fails fast without spinning up an SSH session.
    try:
        log = _load_log(args.log)
    except FileNotFoundError:
        print(f"[!] log not found: {args.log}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"[!] could not read {args.log}: {exc}", file=sys.stderr)
        return 1

    log_text = json.dumps(log, indent=2, ensure_ascii=False)

    # Connection and model settings come from local.env (loaded at module
    # level by config.py) -- the same values the harness uses.
    ssh_cfg = SSHConfig()
    ollama_cfg = OllamaConfig()

    with OllamaTunnel(ssh_cfg, ollama_cfg) as tunnel:
        tunnel.wait_until_ready()

        print(f"[*] Analysing {args.log} with model '{ollama_cfg.model}'.")
        _generate(
            tunnel.base_url(),
            ollama_cfg.model,
            system=_ANALYZER_SYSTEM_PROMPT,
            prompt=_ANALYZER_PROMPT.format(log_text=log_text),
            timeout=300.0,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

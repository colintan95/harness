#!/usr/bin/env python3
"""
harness.py – Connect to a remote Ollama server over SSH and start a model
via the /api/chat endpoint.

Instead of a pure-Python SSH library this script shells out to the system
``ssh`` binary.  For the Ollama host it uses ``-L`` (local port forwarding)
to tunnel a local port to the remote Ollama daemon (typically
``127.0.0.1:11434``) and then talks to Ollama's ``/api/chat`` endpoint
locally.  For the VM it runs ``tools.py`` as a remote RPC server over the
same SSH connection style; VM settings come from ``VM_*`` keys in
``local.env``.

It also deploys ``tools.py`` to the VM, runs it as a background RPC
server for the lifetime of the harness, and makes an initial demo call
(``tools.hello()``) that returns ``"Hello, World!"``.  When the harness
exits (e.g. Ctrl+C) it sends a ``shutdown`` message to ``tools.py`` and
waits for the tool to confirm it is stopping before closing the SSH
session.

Configuration is read from ``env/local.env`` in the project root
(the parent of ``src/``).  Copy ``env/local.env.example`` to
``env/local.env`` and edit the values.  CLI flags
override ``local.env`` values.

Connection and model settings are read from ``local.env``.  The remaining
CLI flags (``--key``, ``--system``) override those settings or carry values
that are not part of ``local.env``.

The program opens an interactive chat: a ``>>>`` prompt reads your input,
forwards it to the model, prints the reply, and repeats until you press
Ctrl+C.

Usage
-----
    python harness.py

This opens an interactive session: type a message at the ``>>>`` prompt, the
reply is printed, and you are prompted again.  Press Ctrl+C (or Ctrl+D) to
quit.

Examples
--------
    # key auth via a specific key file
    python harness.py --key ~/.ssh/id_rsa

    # with a system prompt
    python harness.py --system "You are a helpful assistant."

    # override the model (local.env OLLAMA_MODEL)
    python harness.py --model qwen2.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Optional, TypeAlias, Union

import requests


# SSH tunneling, the remote-tools lifecycle, and their configuration live in
# tunnel.py; this module is the application layer (chat REPL + CLI).
from tunnel import (
    OllamaConfig,
    OllamaTunnel,
    RemoteTools,
    SSHConfig,
    ToolsConfig,
    VMSSHConfig,
)


# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------

_GREY = "\033[90m"   # bright black == grey
_RESET = "\033[0m"


def _grey(text: str) -> str:
    """Wrap *text* in ANSI grey, but only when stdout is a real terminal."""
    if not sys.stdout.isatty():
        return text
    return f"{_GREY}{text}{_RESET}"


# requests types the ``json=`` argument as this recursive "JSON value" shape;
# mirror it so the chat payload type-checks without a ``cast``.
JsonType: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    Sequence["JsonType"],
    Mapping[str, "JsonType"],
]


# ---------------------------------------------------------------------------
# Ollama /api/show – capability probe
# ---------------------------------------------------------------------------
def model_supports_think(
    base_url: str,
    model: str,
    *,
    timeout: float = 30.0,
) -> bool:
    """
    Query ``/api/show`` for *model* and return *True* when the model's
    capabilities include ``"thinking"``.  This is a cheap metadata lookup —
    it does not load the model into RAM.
    """
    try:
        resp = requests.post(
            f"{base_url}/api/show",
            json={"model": model, "verbose": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        info = resp.json()
    except (requests.RequestException, ValueError):
        return False
    return "thinking" in info.get("capabilities", [])


# ---------------------------------------------------------------------------
# Ollama /api/chat
# ---------------------------------------------------------------------------
def chat(
    base_url: str,
    model: str,
    *,
    timeout: float = 300.0,
    system: Optional[str] = None,
) -> None:
    """
    Interactive REPL over /api/chat.

    Prints a ``>>>`` prompt, reads a user turn from stdin, POSTs the growing
    conversation to the remote Ollama server, streams the model's reply
    token-by-token, and repeats.  Runs until the user interrupts with Ctrl+C
    (or Ctrl+D / EOF).

    Ollama loads the model into memory on the first request if it is not
    already in RAM, so the first turn also doubles as a "start model" call.
    Conversation context is kept between turns.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    print(f"[*] Chat with model '{model}'. Press Ctrl+C to quit.")

    # Probe the model to see whether it actually produces thinking / reasoning content.
    supports_think = model_supports_think(base_url, model)
    if supports_think:
        print(_grey(f"[*] Model '{model}' supports think (reasoning enabled)."))
    else:
        print(_grey(f"[*] Model '{model}' does not support think."))

    print()  # separator before the first prompt
    while True:
        # Read a user turn.  input() raises KeyboardInterrupt on Ctrl+C and
        # EOFError on Ctrl+D / end-of-input; either ends the session.
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()                 # move to a fresh line after the interrupt
            print()
            print("[*] Bye.")
            return

        if not prompt:
            continue               # empty line: just prompt again

        messages.append({"role": "user", "content": prompt})
        payload: dict[str, JsonType] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        # Only ask the model to reason if it supports it.
        if supports_think:
            payload["think"] = True

        reply: list[str] = []
        reply_think: list[str] = []
        total_tokens: int = 0
        try:
            # stream=True + iter_lines(): Ollama emits NDJSON, one partial
            # message object per line, finished by a {"done": true} line.
            print()  # blank line before the response
            with requests.post(
                f"{base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = obj.get("message", {})
                    # Reasoning models stream their "thoughts" into
                    # message.thinking; print them in grey so they don't get
                    # confused with the actual answer.
                    thought = msg.get("thinking")
                    if thought:
                        reply_think.append(thought)
                        print(_grey(thought), end="", flush=True)
                    chunk = msg.get("content", "")
                    if chunk:
                        # Move to a new line when transitioning from thinking
                        # to the actual response so they don't share a line.
                        if reply_think and not reply:
                            print()
                            print()  # extra blank line after thinking section
                        reply.append(chunk)
                        print(chunk, end="", flush=True)  # print token-by-token
                    if obj.get("done"):
                        total_tokens = obj.get("prompt_eval_count", 0) + obj.get("eval_count", 0)
                        break
            print()  # newline so the next prompt starts on a fresh line
            print()
            print(f"[*] Tokens: {total_tokens}")
            print()  # separator after tokens, before next prompt
        except KeyboardInterrupt:
            # Interrupted mid-stream: drop to a fresh line, print a goodbye, and
            # return so the with-blocks in main() close the tunnels.
            print()
            print()
            print("[*] Bye.")
            return
        except requests.RequestException as exc:
            print(f"[!] request failed: {exc}")
            messages.pop()         # drop the user turn so it isn't replayed
            continue

        assistant_msg: dict[str, str] = {
            "role": "assistant",
            "content": "".join(reply),
        }
        # Feed the model's reasoning back into the conversation so it can
        # build on its own chain-of-thought on the next turn.  Ollama emits
        # reasoning in the "thinking" field, so send it back under the same key.
        if reply_think:
            assistant_msg["thinking"] = "".join(reply_think)
        messages.append(assistant_msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Connect to a remote Ollama server over SSH and start a model via /api/chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Connection/model settings come from local.env; only a few CLI overrides
    # remain (the ones that are not part of local.env).
    p.add_argument("--key", default=None,
                   help="SSH key file (local.env SSH_KEY, optional)")
    p.add_argument("--model", default=None,
                   help="Model name to chat with (local.env OLLAMA_MODEL, default llama3)")
    p.add_argument("--system", default=None,
                   help="Optional system prompt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Connection and model settings come from local.env (loaded at module
    # level); --key and --model override those when supplied.
    ssh_cfg = SSHConfig(
        key_filename=os.path.expanduser(args.key) if args.key else SSHConfig().key_filename,
    )
    ollama_cfg = OllamaConfig()
    if args.model:
        ollama_cfg.model = args.model

    vm_ssh_cfg = VMSSHConfig(
        key_filename=os.path.expanduser(args.key) if args.key else VMSSHConfig().key_filename,
    )
    tools_cfg = ToolsConfig()

    with OllamaTunnel(ssh_cfg, ollama_cfg) as tunnel:
        tunnel.wait_until_ready()

        # Deploy tools.py to the VM and keep it alive for the same
        # lifetime as the harness.  The context-manager __exit__ asks the
        # remote process to shut down gracefully (and waits for its
        # confirmation) when the with-block ends (normal exit, exception,
        # or KeyboardInterrupt from Ctrl+C in the chat).
        with RemoteTools(vm_ssh_cfg, tools_cfg) as tools:
            # --- demo call: show that the RPC round-trip works ---
            result = tools.call("hello")
            print(f"[*] tools.hello() -> {result!r}")

            # Run the interactive chat session while tools is alive.
            chat(tunnel.base_url(), ollama_cfg.model, system=args.system)
        # ^ when this with-block exits (chat returns on Ctrl+C / EOF),
        #   RemoteTools.__exit__ -> stop() asks tools.py to shut down
        #   gracefully and waits for its confirmation.

    return 0


if __name__ == "__main__":
    sys.exit(main())

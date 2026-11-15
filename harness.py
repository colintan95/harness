#!/usr/bin/env python3
"""
harness.py – Connect to a remote Ollama server over SSH and start a model
via the /api/chat endpoint.

Instead of a pure-Python SSH library this script shells out to the system
``ssh`` binary with ``-L`` (local port forwarding).  That gives us a tunnel
from a local port to the remote Ollama daemon (typically
``127.0.0.1:11434``) and then we just talk to Ollama's ``/api/chat`` endpoint
locally.

Configuration is read from ``local.env`` in the same directory.  Copy
``local.env.example`` to ``local.env`` and edit the values.  CLI flags
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
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias, Union

import requests


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
# local.env loader
# ---------------------------------------------------------------------------

def _load_local_env(path: str) -> dict[str, str]:
    """
    Read a simple ``KEY=VALUE`` file (one per line, ``#`` comments,
    optional surrounding whitespace or quotes on the value).  Missing file
    returns an empty dict.
    """
    env: dict[str, str] = {}
    if not os.path.isfile(path):
        return env
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"[!] local.env line {lineno}: skipping '{line}' (no '=')",
                      file=sys.stderr)
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # strip matching surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            env[key] = value
    return env


_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_ENV = _load_local_env(os.path.join(_LOCAL_DIR, "local.env"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SSHConfig:
    host: str = _LOCAL_ENV.get("SSH_HOST", "127.0.0.1")
    port: int = 22  # always use the standard SSH port
    username: str = _LOCAL_ENV.get("SSH_USER", "deploy")
    # SSH_KEY is optional: when unset, leave the key path empty so we omit
    # the -i flag and let ssh fall back to its own default key selection.
    key_filename: str = os.path.expanduser(
        _LOCAL_ENV.get("SSH_KEY", "")
    )


@dataclass
class OllamaConfig:
    """The remote Ollama endpoint and the local tunnel port."""
    remote_host: str = "127.0.0.1"   # Ollama listens on loopback *on the remote host*
    remote_port: int = int(_LOCAL_ENV.get("OLLAMA_API_PORT", "11434"))
    local_port: int = int(_LOCAL_ENV.get("OLLAMA_LOCAL_PORT", "18434"))
    model: str = _LOCAL_ENV.get("OLLAMA_MODEL", "llama3")


# ---------------------------------------------------------------------------
# SSH tunnel – shells out to the system ``ssh`` binary
# ---------------------------------------------------------------------------

class SSHTunnel:
    """
    Starts ``ssh -L local_port:remote_host:remote_port user@host -N`` as a
    background subprocess.  While the tunnel is alive every TCP connection to
    ``127.0.0.1:local_port`` on the *local* machine is forwarded through SSH
    to ``remote_host:remote_port`` on the *remote* machine.

    Usage::

        with SSHTunnel(ssh_cfg, ollama_cfg) as tunnel:
            tunnel.wait_until_ready()
            chat(tunnel.base_url(), ollama_cfg.model)
    """

    def __init__(self, ssh_cfg: SSHConfig, ollama_cfg: OllamaConfig):
        self.ssh_cfg = ssh_cfg
        self.ollama_cfg = ollama_cfg
        self.local_port = ollama_cfg.local_port
        self._proc: Optional[subprocess.Popen[Any]] = None

    # -- context-manager ---------------------------------------------------

    def __enter__(self) -> "SSHTunnel":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Spawn ``ssh -L ...`` as a background process."""
        cmd = [
            "ssh",
            "-L", f"{self.local_port}:{self.ollama_cfg.remote_host}:{self.ollama_cfg.remote_port}",
        ]

        # Host / port / user.
        cmd.extend(["-p", str(self.ssh_cfg.port)])
        if self.ssh_cfg.key_filename:
            cmd.extend(["-i", self.ssh_cfg.key_filename])
        cmd.append(f"{self.ssh_cfg.username}@{self.ssh_cfg.host}")

        # -N : do not execute a remote command, just forward ports
        # -o StrictHostKeyChecking=accept-new so first connect doesn't hang
        # -o ExitOnForwardFailure=yes so ssh exits if the port is in use
        cmd.extend([
            "-N",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes",
        ])

        target = f"{self.ollama_cfg.remote_host}:{self.ollama_cfg.remote_port} (on remote)"
        print(f"[*] SSH  -> {self.ssh_cfg.username}@{self.ssh_cfg.host}:{self.ssh_cfg.port}")
        print(f"[*] Tunnel 127.0.0.1:{self.local_port} -> {target}")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Give ssh a brief moment to fail-fast (bad host, auth error, port in use)
        time.sleep(0.5)
        if self._proc.poll() is not None:
            err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"ssh failed to start (exit {self._proc.returncode}):\n{err}")

        print(f"[*] ssh PID {self._proc.pid}")

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            print(f"[*] ssh (PID {self._proc.pid}) terminated. Tunnel closed.")
            self._proc = None

    # -- helpers -----------------------------------------------------------

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def wait_until_ready(self, *, timeout: float = 15.0, interval: float = 0.5) -> None:
        """
        Poll the local tunnel port with a raw TCP connect until the port is
        accepting connections or *timeout* elapses.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
                raise RuntimeError(f"ssh exited early (code {self._proc.returncode}):\n{err}")
            try:
                with socket.create_connection(("127.0.0.1", self.local_port), timeout=2):
                    print(f"[OK] Tunnel port {self.local_port} is accepting connections.")
                    return
            except OSError:
                pass
            time.sleep(interval)
        raise TimeoutError(
            f"Tunnel port {self.local_port} not reachable within {timeout}s"
        )


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

    while True:
        # Read a user turn.  input() raises KeyboardInterrupt on Ctrl+C and
        # EOFError on Ctrl+D / end-of-input; either ends the session.
        try:
            print()
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()                 # move to a fresh line after the interrupt
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

        print(f"[*] POST /api/chat  model='{model}'  prompt='{prompt}'")
        reply: list[str] = []
        reply_think: list[str] = []
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
                    # message.think; print them in grey so they don't get
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
                        break
            print()  # newline so the next prompt starts on a fresh line
            if reply_think:
                joined = "".join(reply_think)
                print()
                print(_grey(f"[*] thoughts: {joined}"))
        except KeyboardInterrupt:
            print()                # interrupted mid-stream; stop and let the
            print("[*] Bye.")      # tunnel close in the context manager
            return
        except requests.RequestException as exc:
            print(f"[!] request failed: {exc}")
            messages.pop()         # drop the user turn so it isn't replayed
            continue

        messages.append({"role": "assistant", "content": "".join(reply)})


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

    with SSHTunnel(ssh_cfg, ollama_cfg) as tunnel:
        tunnel.wait_until_ready()
        chat(tunnel.base_url(), ollama_cfg.model, system=args.system)
    return 0


if __name__ == "__main__":
    sys.exit(main())

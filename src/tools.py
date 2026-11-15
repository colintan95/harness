#!/usr/bin/env python3
"""
tools.py – Remote RPC server managed by the harness.

Copied to the VM's home directory (``~/tools.py``) and run as a direct
child of the SSH session by :mod:`harness`  (not ``nohup``'d).  It
listens on a TCP port and accepts newline-delimited JSON requests of
the form::

    {"method": "<name>", "args": [...], "kwargs": {}}

and replies with a single JSON object::

    {"result": <value>}                  # on success
    {"result": null, "error": "<msg>"}   # on failure

The harness opens an SSH port-forward so the caller reaches this server
on the local machine.  Because the process is a child of the SSH
session, closing that connection terminates ``tools.py`` automatically —
but the *normal* way to stop it is the graceful handshake below.

Graceful shutdown
-----------------
Instead of killing the process, the harness stops the tool *by message*:
it sends a ``{"method": "shutdown"}`` control request.  This server runs
its cleanup hook, replies with a confirmation that it is about to stop,
and then exits.  The harness waits for that confirmation so it can be
sure the tool really is shutting down (rather than the SSH session
silently killing it).

The harness makes one call at a time, so the server handles connections
sequentially — no threading needed.

Usage (on the VM, normally started by the harness)::

    python3 tools.py --port 18435
"""

from __future__ import annotations

import argparse
import json
import socket
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Control message name the harness uses to ask this server to stop.
SHUTDOWN_METHOD = "shutdown"


def hello() -> str:
    """Return a simple greeting.  This is the first demo call."""
    return "Hello, World!"


def shutdown() -> dict[str, Any]:
    """
    Control message: ask the server to stop *after* replying.

    Runs the cleanup hook, then returns a confirmation payload the
    harness uses to verify the tool received the request and is shutting
    itself down (as opposed to the SSH session killing it with SIGTERM).
    """
    _cleanup()
    return {
        "status": "shutting_down",
        "message": "tools.py is shutting down",
    }


# Map of method-name -> callable.  Add new tools here.  ``shutdown`` is a
# control message (see :data:`SHUTDOWN_METHOD`) rather than a "tool", but
# it lives in the registry so it is dispatched through the same path.
TOOLS: dict[str, Callable[..., Any]] = {
    "hello": hello,
    SHUTDOWN_METHOD: shutdown,
}


# ---------------------------------------------------------------------------
# Cleanup hook
# ---------------------------------------------------------------------------

def _cleanup() -> None:
    """
    Best-effort cleanup run once, just before the server stops.  Add
    resource releases here (close files, flush logs, signal sibling
    processes, ...).  Invoked by the ``shutdown`` control handler.
    """
    # Nothing to release yet; add cleanup here as tools grow.
    return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def serve(host: str, port: int) -> None:
    """
    Bind a TCP socket and handle connections sequentially.

    Each connection carries one newline-delimited JSON request; the
    response is a single JSON object.  The loop runs until a
    ``shutdown`` control message is received (a graceful stop) or the
    process is terminated.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    print(f"[tools] listening on {host}:{port}", flush=True)

    while True:
        conn, _ = srv.accept()
        stop = _handle_one(conn)
        if stop:
            print("[tools] shutdown confirmed; closing server", flush=True)
            break
    srv.close()
    print("[tools] server stopped", flush=True)


# ---------------------------------------------------------------------------
# Per-connection handling
# ---------------------------------------------------------------------------

def _handle_one(conn: socket.socket) -> bool:
    """
    Read one JSON request, dispatch, send the JSON response, close.

    Returns ``True`` when the request was the ``shutdown`` control
    message, so the serve loop knows to stop after this connection.
    """
    stop = False
    try:
        # ---- read request (newline-delimited JSON) ----
        buf = bytearray()
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
        raw = buf.decode("utf-8", errors="replace").strip()
        if not raw:
            return False
        request = json.loads(raw)
        method = request.get("method", "")

        # ---- dispatch ----
        result_obj = _dispatch(request)

        # ---- write response ----
        payload = json.dumps(result_obj).encode("utf-8") + b"\n"
        conn.sendall(payload)

        # A shutdown control message asks the server to stop next.
        stop = method == SHUTDOWN_METHOD
    except Exception as exc:
        # Always try to send an error response so the caller doesn't hang.
        err = json.dumps({"result": None, "error": str(exc)}).encode() + b"\n"
        try:
            conn.sendall(err)
        except OSError:
            pass
    finally:
        conn.close()
    return stop


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """
    Look up *method* in :data:`TOOLS`, call it with *args*/*kwargs*,
    and return a ``{"result": ..., "error": ...}`` dict.
    """
    method: str = request.get("method", "")
    args: list[Any] = request.get("args") or []
    kwargs: dict[str, Any] = request.get("kwargs") or {}

    fn = TOOLS.get(method)
    if fn is None:
        return {"result": None, "error": f"unknown method: {method!r}"}
    try:
        result = fn(*args, **kwargs)
        return {"result": result, "error": None}
    except Exception as exc:
        return {"result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Remote tools RPC server (runs on the VM, managed by the harness).",
    )
    p.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default 127.0.0.1)",
    )
    p.add_argument(
        "--port", type=int, default=18435,
        help="Listen port (default 18435)",
    )
    args = p.parse_args()

    serve(args.host, args.port)


if __name__ == "__main__":
    main()

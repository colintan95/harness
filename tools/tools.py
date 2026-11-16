#!/usr/bin/env python3
"""
tools.py – Remote RPC server managed by the harness.

Copied to the VM as ``~/tools/tools.py`` and run as a direct
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

Control commands vs. model-facing tool commands
-----------------------------------------------
The server keeps two kinds of request deliberately separate:

* **Control commands** — ``manifest`` and ``shutdown`` — are harness
  lifecycle messages.  The harness fetches the tool manifest at start-up
  and asks the server to stop at exit.  They live in :data:`CONTROL` and
  are *never* part of the manifest, so the model never sees or calls them.
* **Tool commands** (e.g. ``web_search``) are the model-facing
  entries.  Each registered tool in :data:`TOOLS` gets its own entry in
  the manifest with its own name, description, and parameter schema, so
  the model calls each tool by name directly.

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

    python3 tools/tools.py --port 18435
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import signal
import socket
import subprocess
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Third-party runtime dependency (provisioned on the VM by the harness)
# ---------------------------------------------------------------------------

# tools.py depends on ``duckduckgo-search`` (import name: ``ddgs``) for its
# web-search tool.  This is a *runtime* dependency of the RPC server, which
# runs on the VM -- not of the harness, which runs locally.  The harness
# ensures it is installed on the VM before starting this server
# (see RemoteTools in harness/tunnel.py), installing it only when missing
# so that repeated runs reuse the existing installation.  A standalone run
# must install it first:  pip install duckduckgo-search
try:
    from ddgs import DDGS
except ModuleNotFoundError as exc:  # pragma: no cover - only on a bare install
    raise SystemExit(
        "tools.py needs the 'duckduckgo-search' package (imported as 'ddgs'); "
        "install it with 'pip install duckduckgo-search'. "
        f"({exc})"
    ) from exc

# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

# Control command: ask the server to stop *after* replying (graceful
# shutdown).  A harness lifecycle message, never advertised to the model.
SHUTDOWN_METHOD = "shutdown"

# Control command: fetch the tool manifest to advertise to the model.  Also a
# harness lifecycle message, never advertised to the model.
MANIFEST_METHOD = "manifest"


# ---------------------------------------------------------------------------
# Tools – the model-facing callables, each advertised in the manifest by name
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the web via DuckDuckGo and return up to *max_results* results.

    Thin wrapper over ``duckduckgo-search`` (imported as ``ddgs``), the
    runtime dependency this server pulls in on the VM.  Returns the raw
    result objects the package produces -- the full dict for each hit
    (``title``, ``href``/``url``, ``body``, ``source`` and any other
    fields the current package version emits), with no field filtering.
    """
    # Coerce to int: LLMs sometimes pass numeric args as strings.
    max_results = int(max_results)
    # Return the whole result body for each hit (all fields the package
    # emits) rather than a curated subset.
    return list(DDGS().text(query, max_results=max_results))


def run_command(command: str, timeout: float = 30.0) -> dict[str, Any]:
    """
    Execute a *command* string and capture its output.

    Runs the command directly via ``subprocess`` on the VM (using the
    system shell so the string is parsed like a normal command line).
    Returns a dict with the process ``exit_code``, captured ``stdout`` and
    ``stderr`` text, and an ``error`` field that is populated (non-empty)
    when the command fails to start or times out instead of running to
    completion.
    """
    # Coerce to float: LLMs sometimes pass numeric args as strings.
    timeout = float(timeout)

    # Two independent hardenings, both kept so the command is correct no
    # matter how the server itself was launched:
    #
    #   * stdin=subprocess.DEVNULL detaches the child's stdin so it can't
    #     block waiting on a terminal/pipe.  The server's own stdin is
    #     already /dev/null (the harness starts it that way via
    #     _spawn_ssh), so this is defense-in-depth: it also covers running
    #     tools.py by some other path (e.g. locally for testing).
    #
    #   * start_new_session=True makes the child its own process-group
    #     leader so that, on timeout, we can kill the whole tree a shell
    #     may have spawned (os.killpg below) instead of leaking orphan
    #     processes.
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # The command outlived its budget.  Kill the whole process group so
        # any children the shell spawned are reaped too, then drain the
        # pipes to capture whatever it had written before dying.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass  # the process (and its group) already exited
        try:
            out, err = proc.communicate()
        except Exception:
            out, err = "", ""
        return {
            "exit_code": None,
            "stdout": out or "",
            "stderr": err or "",
            "error": f"command timed out after {timeout}s",
        }

    return {
        "exit_code": proc.returncode,
        "stdout": out or "",
        "stderr": err or "",
        "error": None,
    }


# The model-facing tools.  Each entry gets its own manifest definition so
# the model can call it directly by name.  Control commands
# (``manifest``, ``shutdown``) do not live here — see :data:`CONTROL`.
TOOLS: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "run_command": run_command,
}


# ---------------------------------------------------------------------------
# Control commands – harness lifecycle messages (not model-facing tools)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool manifest – one entry per tool, each with its own parameters
# ---------------------------------------------------------------------------

# Mapping from Python type annotations to JSON Schema type strings.
_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    """Best-effort mapping of a Python type annotation to a JSON Schema type."""
    # Direct match.
    if annotation in _PY_TO_JSON:
        return _PY_TO_JSON[annotation]
    # typing constructs like list[str], dict[str, Any], Optional[X], etc.
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        if origin in _PY_TO_JSON:
            return _PY_TO_JSON[origin]
        # Optional[X] / Union[X, None] – unwrap to the non-None member.
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
    return "string"


def _parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """
    Build a JSON Schema ``parameters`` object for *fn* from its signature.

    Each positional or keyword parameter becomes a property; parameters
    without a default are listed in ``required``.  The ``self``/``cls``
    parameters and ``*args``/``**kwargs`` are skipped.
    """
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        # Skip bound-method receiver and variadic catch-alls.
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        prop: dict[str, Any] = {"type": _json_type(param.annotation)}

        # Add a default value when present so the model knows it is optional.
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _short_description(fn: Callable[..., Any]) -> str:
    """
    Use the first paragraph of a tool's docstring as its manifest
    description (a one-line summary for the model); fall back to the
    bare function name when there is no docstring.
    """
    doc = inspect.cleandoc(fn.__doc__ or "").strip()
    if not doc:
        return fn.__name__
    return doc.split("\n\n", 1)[0].strip()


def _tool_definition(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
    """
    Build a single tool definition for *name* (the model-facing entry in
    the manifest).

    The model calls this tool directly by its *name* (e.g. ``web_search``)
    and passes arguments matching the generated parameter schema.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _short_description(fn),
            "parameters": _parameters_schema(fn),
        },
    }


def manifest() -> list[dict[str, Any]]:
    """
    Build the tool manifest the harness advertises to the model.

    Returns one entry per tool in :data:`TOOLS` (e.g. ``web_search``).  Each
    entry carries the tool's name, description,
    and parameter schema so the model can call it directly.  The control
    commands (``manifest``, ``shutdown``) are deliberately absent — they
    are harness lifecycle messages, not model-facing tools, so they never
    reach the model.
    """
    return [_tool_definition(name, fn) for name, fn in TOOLS.items()]


# Control commands: harness lifecycle messages, dispatched through the same
# wire path as the model-facing tool commands but kept in their own
# registry (and never part of the manifest) so the model never sees or calls
# them.
CONTROL: dict[str, Callable[..., Any]] = {
    MANIFEST_METHOD: manifest,
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

def _invoke(
    fn: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Call *fn* with *args*/*kwargs*, returning the result/error envelope."""
    try:
        result = fn(*args, **kwargs)
        return {"result": result, "error": None}
    except Exception as exc:
        return {"result": None, "error": str(exc)}


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """
    Route one request to a handler and return a ``{"result", "error"}`` dict.

    Control commands (``manifest``, ``shutdown``) dispatch through
    :data:`CONTROL`; each model-facing tool (e.g. ``web_search``) dispatches
    directly into :data:`TOOLS` by its own name;
    anything else is an unknown method.
    """
    method: str = request.get("method", "")
    args: list[Any] = request.get("args") or []
    kwargs: dict[str, Any] = request.get("kwargs") or {}

    # Control commands: harness lifecycle, separate from tool invocation.
    control = CONTROL.get(method)
    if control is not None:
        return _invoke(control, args, kwargs)

    # Model-facing tools: route directly to the tool by name.
    tool = TOOLS.get(method)
    if tool is not None:
        return _invoke(tool, args, kwargs)

    return {"result": None, "error": f"unknown method: {method!r}"}


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

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
import threading
import time
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

# Control command: ask the server to interrupt the in-flight tool (a user
# Ctrl+C during a tool call).  A harness lifecycle message, never advertised to
# the model.
CANCEL_METHOD = "cancel"

# Poll cadence (seconds) shared by the cancellable tool runner, run_command's
# communicate loop, and the accept loop: how often a blocked call re-checks the
# CANCEL flag.
_POLL = 0.2

# On cancel, how long (seconds) the runner waits for a *cooperative* tool
# (run_command) to finish cleaning up and report a payload such as partial
# output before abandoning it.  A non-cooperative blocking call (web_search)
# simply outlives the grace and is abandoned.
_CANCEL_GRACE = 0.5


class ToolInterrupted(Exception):
    """
    Raised by a cooperative tool (run_command) when a user interrupt cancels
    it, after it has cleaned up its own resources (killing its child process
    group).  Carries whatever partial result it managed to produce.  The
    cancellable runner converts it -- and abandons non-cooperative blocking
    calls -- into a structured ``cancelled`` response envelope.
    """

    def __init__(self, payload: Any) -> None:
        super().__init__("tool call interrupted by the user")
        self.payload = payload


# Set by the ``cancel`` control command; watched by the cancellable runner and
# by cooperative tools (run_command).  Cleared at the start of each tool run so
# a stale cancel can't bleed into the next call.
CANCEL = threading.Event()

# Set by the ``shutdown`` control command; watched by the serve() accept loop so
# a threaded server stops accepting once a graceful stop is requested.
_SHUTDOWN = threading.Event()


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


def _kill_and_drain(proc: subprocess.Popen[str]) -> tuple[str, str]:
    """
    Kill a process's group and drain its output.

    Sends SIGKILL to the whole process group (the child was started in its own
    group via ``start_new_session=True``) so any children the shell spawned are
    reaped too, then drains the pipes to capture whatever was written.  Returns
    ``(stdout, stderr)``.  Used on both the timeout and the interrupt paths.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        pass  # the process (and its group) already exited
    try:
        out, err = proc.communicate()
    except Exception:
        out, err = "", ""
    return out or "", err or ""


def run_command(command: str, timeout: float = 30.0) -> dict[str, Any]:
    """
    Execute a *command* string and capture its output.

    Runs the command directly via ``subprocess`` on the VM (using the
    system shell so the string is parsed like a normal command line).
    Returns a dict with the process ``exit_code``, captured ``stdout`` and
    ``stderr`` text, and an ``error`` field that is populated (non-empty)
    when the command fails to start, times out, or is interrupted by the user
    instead of running to completion.
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
    #     leader so that, on timeout *or* on a user interrupt, we can kill the
    #     whole tree a shell may have spawned (os.killpg in _kill_and_drain)
    #     instead of leaking orphan processes.
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

    # Poll communicate() with a short timeout so a user interrupt (CANCEL) can
    # kill the child *mid-run* instead of waiting for the full budget; the same
    # poll also bounds the original hard timeout.  On cancel we raise
    # ToolInterrupted carrying the partial output, which the cancellable runner
    # turns into a "cancelled" envelope.
    deadline = time.monotonic() + timeout
    out, err = "", ""
    while True:
        if CANCEL.is_set():
            out, err = _kill_and_drain(proc)
            raise ToolInterrupted(
                {
                    "exit_code": None,
                    "stdout": out,
                    "stderr": err,
                    "error": "interrupted by the user",
                }
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # The command outlived its budget: kill the whole process group so
            # any children the shell spawned are reaped too, then drain.
            out, err = _kill_and_drain(proc)
            return {
                "exit_code": None,
                "stdout": out,
                "stderr": err,
                "error": f"command timed out after {timeout}s",
            }
        try:
            out, err = proc.communicate(timeout=min(_POLL, remaining))
            break
        except subprocess.TimeoutExpired:
            continue

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
    # Signal the threaded accept loop to stop so the server exits cleanly.
    _SHUTDOWN.set()
    return {
        "status": "shutting_down",
        "message": "tools.py is shutting down",
    }


def cancel() -> dict[str, Any]:
    """
    Control message: ask the server to interrupt the in-flight tool.

    Sets the module-level CANCEL flag that the cancellable runner watches and
    that cooperative tools (run_command) observe.  The cancelled result is
    returned on the *request* connection where the tool was running; this
    control reply is fire-and-forget (the harness ignores it).
    """
    CANCEL.set()
    return {"status": "cancelled"}


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
    CANCEL_METHOD: cancel,
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
    Bind a TCP socket and handle one connection per thread.

    Each connection carries one newline-delimited JSON request; the
    response is a single JSON object.  Handling each connection in its own
    daemon thread lets a control message such as ``cancel`` be serviced while
    a tool call is running on another connection.  The loop runs until the
    ``_SHUTDOWN`` flag is set (a ``shutdown`` control message) or the process
    is terminated.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    # A short accept timeout so the loop can re-check _SHUTDOWN between
    # connections instead of blocking forever on accept().
    srv.settimeout(_POLL)
    print(f"[tools] listening on {host}:{port}", flush=True)

    while not _SHUTDOWN.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        # Each connection is handled in its own daemon thread so a control
        # message (cancel/shutdown) can be serviced while a tool runs.
        threading.Thread(target=_handle_one, args=(conn,), daemon=True).start()
    srv.close()
    print("[tools] server stopped", flush=True)


# ---------------------------------------------------------------------------
# Per-connection handling
# ---------------------------------------------------------------------------

def _handle_one(conn: socket.socket) -> None:
    """
    Read one JSON request, dispatch, send the JSON response, close.

    Runs in its own thread (serve() spawns one per connection) so a control
    message such as ``cancel`` can be serviced while a tool call is running on
    another connection.  A ``shutdown`` request stops the server via the
    _SHUTDOWN event in serve(), not via any return value here.
    """
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
            return
        request = json.loads(raw)

        # ---- dispatch ----
        result_obj = _dispatch(request)

        # ---- write response ----
        payload = json.dumps(result_obj).encode("utf-8") + b"\n"
        conn.sendall(payload)
    except Exception as exc:
        # Always try to send an error response so the caller doesn't hang.
        err = json.dumps({"result": None, "error": str(exc)}).encode() + b"\n"
        try:
            conn.sendall(err)
        except OSError:
            pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _invoke(
    fn: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Call *fn* with *args*/*kwargs*, returning the result/error envelope.

    Used for control commands (manifest/shutdown/cancel), which run inline and
    are not cancellable.
    """
    try:
        result = fn(*args, **kwargs)
        return {"result": result, "error": None}
    except Exception as exc:
        return {"result": None, "error": str(exc)}


def _run_tool(
    fn: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Invoke a model-facing tool with cancellation.

    The tool runs in a daemon worker thread while this thread polls the
    module-level CANCEL flag.  On cancel we stop waiting and return a
    cancelled envelope; the worker is abandoned (a non-cooperative blocking
    call -- e.g. web_search -- can't be preempted, so it finishes or dies in
    the background).  A cooperative tool that raises ToolInterrupted
    (run_command kills its child group and raises it) contributes its payload
    to the envelope.

    The envelope always carries a ``cancelled`` key (unlike control replies) so
    the client can tell an interrupted call from a normal result or error.
    """
    # A fresh tool run starts with no pending cancel.
    CANCEL.clear()
    box: list[Any] = []

    def worker() -> None:
        try:
            box.append(fn(*args, **kwargs))
        except BaseException as exc:
            # Capture any outcome -- including ToolInterrupted -- for the
            # polling loop below.
            box.append(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        if CANCEL.is_set():
            # Give a cooperative tool a brief grace to clean up its own
            # resources (run_command kills its child group) and report a
            # payload; a non-cooperative blocking call is simply abandoned.
            t.join(timeout=_CANCEL_GRACE)
            if not t.is_alive() and box and isinstance(box[-1], ToolInterrupted):
                return {
                    "result": box[-1].payload,
                    "error": None,
                    "cancelled": True,
                }
            return {"result": None, "error": None, "cancelled": True}
        t.join(timeout=_POLL)
    outcome = box[0]
    if isinstance(outcome, ToolInterrupted):
        return {"result": outcome.payload, "error": None, "cancelled": True}
    if isinstance(outcome, BaseException):
        return {"result": None, "error": str(outcome), "cancelled": False}
    return {"result": outcome, "error": None, "cancelled": False}


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

    # Model-facing tools: route through the cancellable runner so a user
    # interrupt can stop the in-flight call.
    tool = TOOLS.get(method)
    if tool is not None:
        return _run_tool(tool, args, kwargs)

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

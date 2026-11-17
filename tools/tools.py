#!/usr/bin/env python3
"""
tools.py – Remote RPC server managed by the harness.

Copied to the VM's root dir (``<VM_ROOT_DIR>/tools.py``, ``~`` by default)
and run as a direct child of the SSH session by :mod:`harness` (not
``nohup``'d).  It
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

    python3 ~/tools.py --port 18435 --cwd ~/workdir
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import os
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
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
# ToolResult – the uniform result every model-facing tool returns
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    The result a model-facing tool returns -- the two fields the harness stores
    straight into the ``role: "tool"`` message it feeds back to the model:

    * ``content`` -- the text of the result.  Structured results (web_search,
      run_command) JSON-serialize their payload here; an image (view_image)
      puts a short caption here.
    * ``images`` -- a list of raw base64 image strings for Ollama's per-message
      ``images`` field (no ``data:`` prefix; the format /api/chat expects).
      Only view_image populates this, so it defaults to empty and the other
      tools carry content only.

    Because a tool returns exactly the pieces the message needs, the harness
    does not have to infer what kind of result it got -- it just reads these
    two fields.  The result crosses the JSON-RPC wire as a plain dict
    (:func:`_dataclass_default` converts it on the way out, so the harness
    receives it back already as ``{"content": ..., "images": ...}``).
    """

    content: str
    images: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tools – the model-facing callables, each advertised in the manifest by name
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> ToolResult:
    """
    Search the web via DuckDuckGo and return up to *max_results* results.

    Thin wrapper over ``duckduckgo-search`` (imported as ``ddgs``), the
    runtime dependency this server pulls in on the VM.  Returns a content-only
    :class:`ToolResult`: the raw result objects the package produces -- the full
    dict for each hit (``title``, ``href``/``url``, ``body``, ``source`` and
    any other fields the current package version emits, with no field
    filtering) -- JSON-serialized into ``content``.  Web results carry text, so
    ``images`` stays empty.
    """
    # Coerce to int: LLMs sometimes pass numeric args as strings.
    max_results = int(max_results)
    # Content only: the whole result body for each hit (all fields the package
    # emits) JSON-serialized into content; no image.
    results = list(DDGS().text(query, max_results=max_results))
    return ToolResult(content=json.dumps(results))


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


def run_command(command: str, timeout: float = 30.0) -> ToolResult:
    """
    Execute a *command* string and capture its output.

    Runs the command directly via ``subprocess`` on the VM (using the
    system shell so the string is parsed like a normal command line), in the
    working directory the server was started in -- the VM's root dir, or a
    sub-dir of it when launched with ``--cwd`` (the child inherits it).
    Returns a content-only :class:`ToolResult`: the process ``exit_code``,
    captured ``stdout`` and ``stderr`` text, and an ``error`` field that is
    populated (non-empty) when the command fails to start, times out, or is
    interrupted by the user instead of running to completion -- JSON-
    serialized into ``content`` (no image).
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
        return ToolResult(content=json.dumps({
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }))

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
                ToolResult(content=json.dumps({
                    "exit_code": None,
                    "stdout": out,
                    "stderr": err,
                    "error": "interrupted by the user",
                }))
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # The command outlived its budget: kill the whole process group so
            # any children the shell spawned are reaped too, then drain.
            out, err = _kill_and_drain(proc)
            return ToolResult(content=json.dumps({
                "exit_code": None,
                "stdout": out,
                "stderr": err,
                "error": f"command timed out after {timeout}s",
            }))
        try:
            out, err = proc.communicate(timeout=min(_POLL, remaining))
            break
        except subprocess.TimeoutExpired:
            continue

    return ToolResult(content=json.dumps({
        "exit_code": proc.returncode,
        "stdout": out or "",
        "stderr": err or "",
        "error": None,
    }))


# ---------------------------------------------------------------------------
# view_image – read an image off the VM and hand it to the model to see
# ---------------------------------------------------------------------------

# Default cap on the bytes read for one view_image() call.  Ollama's vision
# models turn an image into many prompt tokens and the whole request must fit
# the model's context window, so an unbounded read could produce a payload that
# is both wasteful and likely to overflow the context.  This is a safety rail,
# not a policy: it is a parameter with this default so a caller can raise it
# when it really needs a large image.
_MAX_IMAGE_BYTES = 16 * 1024 * 1024  # 16 MiB


def _image_error(path: str, reason: str, size: int | None = None) -> ToolResult:
    """
    A content-only :class:`ToolResult` reporting a ``view_image`` failure.

    The image can't be produced, so the result carries no ``images``; the error
    (with path and size) is JSON-serialized into ``content`` so the model reads
    the reason and recovers instead of the session crashing.
    """
    return ToolResult(content=json.dumps({"path": path, "size": size, "error": reason}))


def view_image(path: str, max_bytes: int = _MAX_IMAGE_BYTES) -> ToolResult:
    """
    Read an image off the VM, convert it to the base64 Ollama needs, and
    return it so the agent can actually *see* the picture.

    This tool runs on the VM, so *path* is resolved on the VM's filesystem --
    a relative path is taken against the server's working directory (the VM
    root dir, or a sub-dir of it when the server was launched with --cwd); an
    absolute path is used as-is.

    On success it returns a :class:`ToolResult` that carries the image in its
    ``images`` field -- a list of *raw* base64 strings with no ``data:`` URI
    prefix, the format Ollama's per-message ``images`` field (/api/chat)
    expects -- so a vision model renders the picture; ``content`` is just a
    short caption (path, size) so the transcript stays readable instead of a
    multi-kilobyte blob.  On any failure -- missing file, not readable, larger
    than *max_bytes* -- it returns a content-only :class:`ToolResult` whose
    ``content`` is a JSON error (path, size, reason) so the agent reads the
    reason and recovers instead of the session crashing.
    """
    # Coerce: LLMs sometimes pass the size arg as a string.
    max_bytes = int(max_bytes)

    # Resolve on the VM.  A relative path is taken against the server's current
    # working directory (set by --cwd); an absolute path is used unchanged.
    resolved = os.path.abspath(path)

    if not os.path.isfile(resolved):
        return _image_error(resolved, f"no such file or not a regular file: {resolved!r}")

    try:
        size = os.path.getsize(resolved)
    except OSError as exc:
        return _image_error(resolved, f"could not stat {resolved!r}: {exc}")

    if size > max_bytes:
        return _image_error(
            resolved,
            f"image is {size} bytes, exceeding the {max_bytes}-byte limit; "
            "downscale it or pass a larger max_bytes",
            size=size,
        )

    try:
        with open(resolved, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return _image_error(resolved, f"could not read {resolved!r}: {exc}")

    # Ollama wants raw base64 (no "data:..." prefix) and sniffs the format from
    # the bytes itself, so there is nothing more to do but encode them.  The
    # image travels in "images"; "content" is a short caption.
    return ToolResult(
        content=f"[image] {resolved} ({size:,} bytes) attached for the model to view",
        images=[base64.b64encode(data).decode("ascii")],
    )


# The model-facing tools.  Each entry gets its own manifest definition so
# the model can call it directly by name.  Control commands
# (``manifest``, ``shutdown``) do not live here — see :data:`CONTROL`.
TOOLS: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "run_command": run_command,
    "view_image": view_image,
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

# PEP 563 (``from __future__ import annotations``) hands signature annotations to
# ``inspect`` as *strings* ("int", "float", ...), so keep a name->type mirror
# of the same builtins for the string form.  Derived from _PY_TO_JSON so the
# two never drift apart.
_PY_TO_JSON_NAME: dict[str, str] = {
    t.__name__: js for t, js in _PY_TO_JSON.items()
}


def _json_type(annotation: Any) -> str:
    """Best-effort mapping of a Python type annotation to a JSON Schema type."""
    # Direct match.
    if annotation in _PY_TO_JSON:
        return _PY_TO_JSON[annotation]
    # Under ``from __future__ import annotations`` (PEP 563) the signature's
    # annotations arrive as *strings* (e.g. "int", "float") rather than the
    # live types, so a bare numeric/bool param would otherwise fall through to
    # "string".  Resolve the common builtin-name forms; anything more complex
    # (e.g. "list[dict[str, Any]]") is left to the origin/unwrapping below or
    # the "string" fallback.
    if isinstance(annotation, str) and annotation in _PY_TO_JSON_NAME:
        return _PY_TO_JSON_NAME[annotation]
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

def _dataclass_default(obj: Any) -> Any:
    """
    ``json.dumps`` hook so a tool's :class:`ToolResult` crosses the JSON-RPC
    wire as its plain dict form -- the harness receives it back as
    ``{"content": ..., "images": ...}``.  ``default`` is only called for objects
    json can't serialize natively, so plain results (dicts, strings, lists,
    numbers, ``None``) pass through untouched.
    """
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


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
        payload = json.dumps(result_obj, default=_dataclass_default).encode("utf-8") + b"\n"
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
    p.add_argument(
        "--cwd", default=None,
        help="Working directory the server starts in; the model-facing tools "
             "(e.g. run_command) run there.  The harness passes the VM's root "
             "dir, or a sub-dir of it.",
    )
    args = p.parse_args()

    # Start in the requested working dir so the model-facing tools (which run
    # child processes) operate there; they inherit this cwd.  When --cwd is
    # omitted the server keeps the directory it was launched in.
    if args.cwd:
        os.chdir(args.cwd)
        print(f"[tools] working directory: {os.getcwd()}", flush=True)

    serve(args.host, args.port)


if __name__ == "__main__":
    main()

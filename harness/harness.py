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

It also deploys ``tools.py`` to the VM's root dir (``VM_ROOT_DIR`` in
``local.env``, ``~`` by default) and runs it there as a background RPC server
for the lifetime of the harness, starting in that root dir or in a sub-dir of
it when a starting directory is passed on the command line.  When the
harness exits
(e.g. Ctrl+C) it sends a ``shutdown`` message to ``tools.py`` and waits for
the tool to confirm it is stopping before closing the SSH session.

The server's tool manifest -- one entry per tool in Ollama's
``/api/chat`` "tools" shape, each carrying its own name, description,
and parameters -- is
fetched from ``tools.py`` and forwarded to the model so the agent can
request a tool call.  When the model asks for a tool, the harness runs it on
the remote server, feeds the result back as a ``tool`` message, and lets the
model continue until it answers.

Configuration is read from ``env/local.env`` in the project root
(the parent of the ``harness/`` package).  Copy ``env/local.env.example`` to
``env/local.env`` and edit the values.  CLI flags
override ``local.env`` values.

Connection and model settings are read from ``local.env``.  The remaining
CLI flags (``--key``) override those settings or carry values
that are not part of ``local.env``.

The program opens an interactive chat: a ``>>>`` prompt reads your input,
forwards it to the model, prints the reply, and repeats.  At the *prompt*,
the first Ctrl+C clears the input buffer and a second Ctrl+C (on the now
empty buffer) quits the session; Ctrl+C *while the agent is working* (a model
stream or a tool call) cancels the in-flight request and returns to the
prompt (a Ctrl+C while the model is streaming keeps the user's prompt and the
partial response/thought already generated, plus a note that the turn was
interrupted).  Ctrl+D / EOF also quits.

Usage
-----
    python harness.py

This opens an interactive session: type a message at the ``>>>`` prompt, the
reply is printed, and you are prompted again.  Press Ctrl+C twice (or
Ctrl+D) at the prompt to quit -- the first press clears the input; press
Ctrl+C while the model is streaming to keep the partial
response and thought (plus an interruption note) and return to the prompt, or
while a tool call is running to cancel that call and return to the prompt.

Examples
--------
    # key auth via a specific key file
    python harness.py --key ~/.ssh/id_rsa

    # override the model (local.env OLLAMA_MODEL)
    python harness.py --model qwen2.5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import threading
from datetime import datetime, timezone

try:
    import termios
    import tty
    _HAVE_TERMIOS = True
except ImportError:
    _HAVE_TERMIOS = False
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple, TypeAlias, Union

import requests

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings


# The env-driven configuration dataclasses (incl. the local.env loader) live in
# config.py; the SSH tunneling and the remote-tools lifecycle live in remote.py.
# This module is the application layer (chat REPL + CLI).
from config import (
    LogConfig,
    OllamaConfig,
    SSHConfig,
    ToolsConfig,
    VMSSHConfig,
)
from remote import (
    OllamaTunnel,
    RemoteToolInterrupt,
    RemoteTools,
)

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------
#
# Always sent as the first message so the agent starts by exploring the
# project in the tools server's working directory rather than guessing
# or searching in unrelated paths.

_DEFAULT_SYSTEM_PROMPT = """If looking for code and files, start in the current working directory"""

# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------

_GREY = "\033[90m"   # bright black == grey
_WHITE = "\033[97m"  # bright white
_RESET = "\033[0m"


def _grey(text: str) -> str:
    """Wrap *text* in ANSI grey, but only when stdout is a real terminal."""
    if not sys.stdout.isatty():
        return text
    return f"{_GREY}{text}{_RESET}"


def _white(text: str) -> str:
    """Wrap *text* in ANSI white, but only when stdout is a real terminal."""
    if not sys.stdout.isatty():
        return text
    return f"{_WHITE}{text}{_RESET}"


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
# SIGINT scoping
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _signal_scope(
    handler: Callable[[int, Any], None],
) -> Iterator[None]:
    """
    Install *handler* for SIGINT for the duration of a ``with`` block.

    The handler currently installed on entry is saved and restored on exit, so
    the block's signal behaviour does not leak past it.
    """
    prev = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


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
# Tool-calling helpers (used by chat)
# ---------------------------------------------------------------------------

class _Turn(NamedTuple):
    """The assembled pieces of one streamed model response."""

    content: list[str]
    thinking: list[str]
    tool_calls: list[dict[str, Any]]
    total_tokens: int


class _TurnInterrupted(KeyboardInterrupt):
    """
    A model stream interrupted by the user (Ctrl+C mid-generation).

    Carries the partial :class:`_Turn` assembled so far -- whatever content,
    thinking and tool calls had already streamed -- so the caller can commit it
    to the transcript instead of discarding the in-flight turn.
    """

    def __init__(self, turn: "_Turn") -> None:
        super().__init__()
        self.turn = turn


def _stream_turn(
    base_url: str,
    model: str,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[dict[str, Any]] | None,
    supports_think: bool,
    timeout: float,
) -> _Turn:
    """
    Make one streaming ``/api/chat`` call and return its assembled pieces.

    *content* and *thinking* are the streamed text (printed token-by-token as
    it arrives, reasoning in grey); *tool_calls* is the model's requested
    tool calls (empty when it answered instead of requesting one);
    *total_tokens* is the token count from the ``{"done": true}`` line.
    """
    payload: dict[str, JsonType] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    # Advertise the tools so the model can request a call; only when there are
    # any, since some servers reject an empty tools list.
    if tools:
        payload["tools"] = list(tools)
    # Only ask the model to reason if it supports it.
    if supports_think:
        payload["think"] = True

    content: list[str] = []
    thinking: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    total_tokens = 0

    # stream=True + iter_lines(): Ollama emits NDJSON, one partial message
    # object per line, finished by a {"done": true} line.
    print()  # blank line before the response
    try:
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
                # Reasoning models stream their "thoughts" into message.thinking;
                # print them in grey so they aren't confused with the answer.
                thought = msg.get("thinking")
                if thought:
                    thinking.append(thought)
                    print(_grey(thought), end="", flush=True)
                chunk = msg.get("content", "")
                if chunk:
                    # Move to a new line when transitioning from thinking to the
                    # actual response so they don't share a line.
                    if thinking and not content:
                        print()
                        print()  # extra blank line after the thinking section
                    content.append(chunk)
                    print(chunk, end="", flush=True)  # print token-by-token
                # A model that wants to act reports its requests under
                # message.tool_calls.  Ollama puts the complete list in the final
                # chunk, so take the latest rather than accumulating it.
                calls = msg.get("tool_calls")
                if calls:
                    tool_calls = calls
                if obj.get("done"):
                    total_tokens = (
                        obj.get("prompt_eval_count", 0) + obj.get("eval_count", 0)
                    )
                    break
    except KeyboardInterrupt:
        # The user interrupted mid-generation: hand back whatever had already
        # streamed (content, thinking, partial tool calls) as a _TurnInterrupted
        # so the caller can keep it in the transcript instead of discarding the
        # turn.
        raise _TurnInterrupted(
            _Turn(content, thinking, tool_calls, total_tokens)
        ) from None

    print()  # newline so the next prompt starts on a fresh line
    return _Turn(content, thinking, tool_calls, total_tokens)


def _invoke_tool(
    call_tool: Callable[..., Any],
    tool_call: dict[str, Any],
    cancel: threading.Event,
) -> Any:
    """
    Run one model-requested tool call and return its result.

    *call_tool* is the harness's forwarder to the remote tools server; it
    returns the tool's result as a plain dict ``{"content": ..., "images": ...}``
    (the server's ``ToolResult`` dataclass is converted to a dict on the wire),
    which the caller (:func:`_tool_message`) stores straight into the
    ``role: "tool"`` message.  A tool that raises is reported back to the model
    as a content-only result -- a JSON error object in ``content`` -- so it can
    recover instead of the session crashing.

    *cancel* is a threading.Event the SIGINT handler sets while this call is
    running; it is forwarded to *call_tool* so the remote server can interrupt
    the in-flight tool.  A :class:`RemoteToolInterrupt` (the user cancelled the
    call) is *not* swallowed here -- it propagates so the chat loop can record
    an interruption instead of a generic error.
    """
    function = tool_call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        return call_tool(name, arguments, cancel=cancel)
    except RemoteToolInterrupt:
        # The user cancelled this call; let the chat loop record the
        # interruption rather than turning it into a generic tool error.
        raise
    except Exception as exc:
        # Surface the error to the model as a content-only result -- a JSON
        # error object in "content" -- so it can recover; _tool_message stores
        # it into the "tool" message.
        return {"content": json.dumps({"error": str(exc)})}


def _preview(text: str, limit: int = 200) -> str:
    """A single-line preview of a (possibly long) tool result for the console."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + " …"


def _tool_message(result: Any) -> dict[str, Any]:
    """
    Build the ``role: "tool"`` message from a tool's result.

    Every tool returns the pieces the harness stores straight into the message:
    ``content`` (its text) and, for view_image, ``images`` -- a list of raw
    base64 strings for Ollama's per-message ``images`` field, so a vision model
    actually sees the picture.  The harness does not infer what kind of result it
    is; it just takes those two fields.  A result that is not in that shape (a
    tool that raised and was caught in :func:`_invoke_tool`, or some other
    unexpected value) is degraded to a JSON text blob with no images so the
    session never crashes.
    """
    if isinstance(result, dict) and ("content" in result or "images" in result):
        # The normal case: a ToolResult-shaped dict.  Guard each field so a
        # malformed one degrades gracefully rather than crashing the session.
        content = str(result.get("content") or "")
        images = [b for b in (result.get("images") or []) if isinstance(b, str)]
    elif isinstance(result, str):
        content, images = result, []
    else:
        content, images = json.dumps(result), []

    message: dict[str, Any] = {"role": "tool", "content": content}
    # Attach images only when the tool produced some, so non-vision tools get a
    # clean text message and the request stays small.
    if images:
        message["images"] = images
    return message


# ---------------------------------------------------------------------------
# Ollama /api/chat
# ---------------------------------------------------------------------------
def _dump_transcript(log_dir: str, messages: Sequence[dict[str, Any]]) -> None:
    """
    Write the chat *messages* (the full transcript) to a timestamped JSON
    file in *log_dir*.

    Logging is opt-in: when *log_dir* is empty -- no ``LOG_DIR`` in
    ``local.env`` -- this is a no-op.  When it is set the directory is
    created if it does not exist and the message list is dumped
    (``json.dump``) into a file named with a UTC timestamp, e.g.
    ``chat-20260101-123456Z.json`` (the trailing ``Z`` marks UTC).
    """
    if not log_dir:
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(log_dir, f"chat-{ts}Z.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(messages, fh, indent=2)
        fh.write("\n")
    print(f"[*] Transcript written to {path}")


def chat(
    base_url: str,
    model: str,
    *,
    timeout: float = 300.0,
    tools: list[dict[str, Any]] | None = None,
    call_tool: Callable[..., Any] | None = None,
    log_dir: str = "",
) -> None:
    """
    Interactive REPL over /api/chat.

    Prints a ``>>>`` prompt, reads a user turn from stdin, POSTs the growing
    conversation to the remote Ollama server, streams the model's reply
    token-by-token, and repeats.  At the *prompt* the first Ctrl+C clears the
    input buffer; a second Ctrl+C (or Ctrl+D / EOF) ends the session.

    A Ctrl+C *while the agent is working* returns to the prompt instead of
    quitting: during a model stream the in-flight request is aborted but the
    user's prompt and the partial response/thought already generated are kept,
    together with a note that the turn was interrupted by the user; during a
    tool call the in-flight call is cancelled (the server is asked to stop
    it and the tool's result is replaced by an interruption note), and *then*
    the loop stops and returns to the prompt.

    When *tools* (the manifest the remote server advertised) is given, it is
    forwarded to the model under ``tools``; when the model requests a tool
    call, *call_tool* runs it and the result is fed back as a ``tool``
    message so the model can continue.  The model is re-queried in a loop
    until it stops requesting tools (the final answer).

    Every tool returns the two fields the harness stores straight into the
    ``tool`` message: ``content`` (text) and, for ``view_image()``, ``images``
    -- a list of raw base64 strings (no ``data:`` prefix) for Ollama's
    per-message ``images`` field, so a vision model actually *sees* the picture
    rather than reading base64 as text.  The image tool puts a short caption in
    ``content``; the other tools carry content only.

    Ollama loads the model into memory on the first request if it is not
    already in RAM, so the first turn also doubles as a "start model" call.
    Conversation context is kept between turns.
    """
    messages: list[dict[str, Any]] = []
    messages.append({"role": "system", "content": _DEFAULT_SYSTEM_PROMPT})

    print(f"[*] Chat with model '{model}'. Ctrl+C clears the prompt; "
          "a second Ctrl+C quits. Ctrl+C during a response cancels the "
          "request and returns to the prompt.")

    # Probe the model to see whether it actually produces thinking / reasoning content.
    supports_think = model_supports_think(base_url, model)
    if supports_think:
        print(_grey(f"[*] Model '{model}' supports think (reasoning enabled)."))
    else:
        print(_grey(f"[*] Model '{model}' does not support think."))

    print()  # separator before the first prompt

    # prompt_toolkit session for the >>> prompt.
    #
    # multiline=True lets the buffer hold newlines, so a pasted
    # multi-line string (handled via the terminal's bracketed-paste
    # mode) stays as one input instead of being split into several
    # turns.  The Ctrl+J binding below lets the user *type* a
    # newline in the middle of a message without submitting; Enter
    # still submits as before.  InMemoryHistory gives arrow-key
    # recall of previous prompts.
    _pt_bindings = KeyBindings()

    @_pt_bindings.add('c-j')
    def _pt_insert_newline(event: Any) -> None:
        """Ctrl+J inserts a newline in multi-line input."""
        event.current_buffer.insert_text('\n')

    @_pt_bindings.add('enter')
    def _pt_submit(event: Any) -> None:
        """
        Enter submits the prompt when it has content, and is a no-op when
        the buffer is empty.

        prompt_toolkit's default ``enter`` handler is gated behind a
        ``not multiline`` filter, so in multi-line mode it only inserts a
        newline.  This binding overrides that: Enter calls
        ``validate_and_handle()`` to submit the text when the buffer is
        non-empty; when the buffer is empty the call is skipped so the
        user stays on the same ``>>> `` prompt instead of getting a new
        one.  Ctrl+J remains the way to insert a newline mid-message.
        """
        buffer = event.current_buffer
        if buffer.text:
            buffer.validate_and_handle()

    @_pt_bindings.add('c-c')
    @_pt_bindings.add('<sigint>')
    def _pt_ctrl_c(event: Any) -> None:
        """
        Two-stage Ctrl+C at the >>> prompt.

        * First press (buffer has text): clear the input so the user can
          start over without losing the session.
        * Second press (buffer is empty): raise ``KeyboardInterrupt`` via
          ``event.app.exit`` so the surrounding ``prompt()`` call re-raises
          it and the ``except (KeyboardInterrupt, EOFError)`` block below
          ends the session -- preserving the current behaviour.
        """
        buffer = event.current_buffer
        if buffer.text:
            buffer.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    _pt_session = PromptSession(
        history=InMemoryHistory(),
        multiline=True,
        prompt_continuation='... ',
        key_bindings=_pt_bindings,
    )

    # A Ctrl+C during the agent loop should *cancel the current request* and
    # return to the >>> prompt rather than quitting the whole session.  We
    # install a SIGINT handler that behaves differently by phase:
    #
    #   * while the model is streaming (not inside a tool call) it raises
    #     KeyboardInterrupt, which _stream_turn turns into a _TurnInterrupted
    #     carrying the partial response; the loop below catches it to keep the
    #     partial turn and add an interruption note;
    #   * while a remote tool is running it sets the cancel event instead of
    #     raising; the remote read loop watches that event, asks the server to
    #     cancel the tool, and the call is stopped -- an interruption note is
    #     appended in its place before stopping and returning to the prompt.
    #
    # A Ctrl+C *at the >>> prompt* is in neither phase, so the handler raises
    # and input() ends the session as before.
    loop: dict[str, Any] = {"in_tool_call": False, "cancel": threading.Event()}

    def _on_sigint(signum: int, frame: Any) -> None:
        if loop["in_tool_call"]:
            # A user interrupt *during* a tool call: flag cancellation so the
            # remote read loop asks the server to cancel the in-flight tool.
            loop["cancel"].set()
        else:
            raise KeyboardInterrupt

    # Scope the handler to this REPL: _signal_scope installs it for the block
    # body and restores the previous handler on exit (Ctrl+C / EOF at the
    # prompt, a return, or an exception), so the behaviour does not leak
    # past chat().

    def _drain_stdin() -> None:
        """
        Discard any characters the user typed while the agent was streaming.

        prompt_toolkit is not reading input during a model response or tool
        call, so keystrokes sit in the terminal driver's input queue.
        ``tcflush`` flushes that queue (line buffer + OS buffer) in one call;
        without it the next prompt() would pick them up and render a spurious
        newline or partial input at the prompt.
        """
        if not _HAVE_TERMIOS:
            return
        try:
            termios.tcflush(0, termios.TCIFLUSH)
        except termios.error:
            pass

    with _signal_scope(_on_sigint):
        _first = True
        # Track whether the user actually submitted at least one prompt, so a
        # transcript is only written for a session that did something rather
        # than for one that quit before any input.
        _prompted = False
        while True:
            # Ring the bell when returning to the prompt after an agent turn.
            if not _first:
                sys.stdout.write("\a")
                sys.stdout.flush()
            _first = False
            _drain_stdin()
            # Read a user turn.  prompt_toolkit raises KeyboardInterrupt on Ctrl+C
            # and EOFError on Ctrl+D / end-of-input; either ends the session.
            try:
                prompt = _pt_session.prompt(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()                 # move to a fresh line after the interrupt
                # Persist the transcript only when a log directory is configured
                # and the user actually prompted something -- a no-op when
                # LOG_DIR is unset or the session quit before any input.
                if _prompted:
                    _dump_transcript(log_dir, messages)
                print()
                return

            if not prompt:
                continue               # empty line: just prompt again

            messages.append({"role": "user", "content": prompt})
            _prompted = True
            # Clear any stale cancel from a previous turn before a new one.
            loop["cancel"].clear()

            # The model may request a tool call instead of answering.  Loop: call
            # the model; if it asks for a tool, run it (forwarding the call back to
            # the remote tools server via *call_tool*) and feed the result back as a
            # "tool" message; repeat until it produces a final answer.
            committed = False
            while True:
                try:
                    turn = _stream_turn(
                        base_url,
                        model,
                        messages,
                        tools=tools,
                        supports_think=supports_think,
                        timeout=timeout,
                    )
                except _TurnInterrupted as exc:
                    # Ctrl+C while the model was streaming (not in a tool call):
                    # keep the user's prompt and whatever response/thought had
                    # already streamed, and add a note that the turn was
                    # interrupted, then return to the >>> prompt instead of
                    # quitting the session.
                    turn = exc.turn
                    print()
                    print()
                    print(_grey("[*] Interrupted by the user; keeping the partial response and thought."))
                    print()
                    # Commit the partial assistant turn -- its content and
                    # thinking, but not the unreliable in-flight tool calls -- so
                    # the next turn can build on what was already generated.
                    if turn.content or turn.thinking:
                        partial: dict[str, Any] = {
                            "role": "assistant",
                            "content": "".join(turn.content),
                        }
                        if turn.thinking:
                            partial["thinking"] = "".join(turn.thinking)
                        messages.append(partial)
                    # Tell the agent the prompt was interrupted by the user so it
                    # can acknowledge it and await new instructions.
                    messages.append({
                        "role": "user",
                        "content": "[Interrupted by the user]",
                    })
                    break
                except KeyboardInterrupt:
                    # A bare interrupt that bypassed _stream_turn's handler:
                    # cancel this request and return to the >>> prompt instead of
                    # quitting the session, leaving the transcript as it was.
                    print()
                    print()
                    print(_grey("[*] Request cancelled; returning to the prompt."))
                    if not committed:
                        messages.pop()
                    break
                except requests.RequestException as exc:
                    print(f"[!] request failed: {exc}")
                    # Drop the user turn only if nothing for it was committed yet, so
                    # a failure mid tool-loop can't corrupt the transcript.
                    if not committed:
                        messages.pop()
                    break

                committed = True
                print()
                print(f"[*] Tokens: {turn.total_tokens}")
                print()  # separator after tokens, before the next response

                if turn.tool_calls and call_tool is not None:
                    # The model requested a tool: record its request, run each tool,
                    # and feed every result back so the next call can use them.
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": "".join(turn.content),
                        "tool_calls": turn.tool_calls,
                    }
                    if turn.thinking:
                        assistant_msg["thinking"] = "".join(turn.thinking)
                    messages.append(assistant_msg)

                    # Run each requested tool, recording its result as a "tool"
                    # message.  A Ctrl+C cancels the *batch*: if it lands during a
                    # call that call is deferred to completion and its result is
                    # appended first; either way we then stop and return to the
                    # prompt rather than re-querying the model.
                    cancelled = False
                    try:
                        for tool_call in turn.tool_calls:
                            function = tool_call.get("function", {})
                            arguments = function.get("arguments") or {}
                            # The model calls the single `tools` command; the actual
                            # tool it picked is named under arguments["name"], so show
                            # that (falling back to the command name).
                            name = arguments.get("name") or function.get("name", "?")
                            # Show the call and its arguments *before* running the tool,
                            # then print the result it produced -- request, then response.
                            print(_white(f"[*] Tool call: {name} {_preview(json.dumps(arguments))}"))
                            # While this call runs the SIGINT handler sets loop["cancel"]
                            # (a threading.Event) instead of raising; the remote read
                            # loop watches it and asks the server to cancel the tool,
                            # which is then stopped and reported as interrupted.
                            loop["in_tool_call"] = True
                            loop["cancel"].clear()
                            raw_result: Any = None
                            try:
                                raw_result = _invoke_tool(call_tool, tool_call, loop["cancel"])
                            except RemoteToolInterrupt:
                                # The user interrupted this tool: the server cancelled
                                # it and returned a cancelled envelope.  Record the
                                # interruption as the tool's result so the transcript
                                # stays coherent, then stop the batch and return to the
                                # prompt.
                                print()
                                print(_grey("[*] Interrupted by the user; the tool call was cancelled."))
                                messages.append({"role": "tool", "content": "[Interrupted by the user]"})
                                cancelled = True
                                break
                            finally:
                                loop["in_tool_call"] = False
                            # The tool returned the pieces the message needs (content,
                            # and for view_image images); _tool_message stores them
                            # straight in -- no inference about the result's kind.
                            tool_message = _tool_message(raw_result)
                            print()  # blank line between the call and its result
                            # Preview the message text (a caption for an image), not a
                            # multi-kilobyte base64 blob.
                            print(_white(f"[*] Tool result: {name} {_preview(tool_message['content'])}"))
                            messages.append(tool_message)
                            # Backstop: if a cancel was flagged but the server didn't
                            # honour it (e.g. an older server that ignores cancel), the
                            # call ran to completion -- still stop the batch.
                            if loop["cancel"].is_set():
                                print()
                                print(_grey("[*] Interrupted after the current tool call; returning to the prompt."))
                                cancelled = True
                                break
                    except KeyboardInterrupt:
                        # A Ctrl+C that landed *between* tool calls (not inside one)
                        # cancels the batch.  No call is in flight, so nothing is left
                        # running.
                        print()
                        print(_grey("[*] Interrupted; returning to the prompt."))
                        cancelled = True
                    finally:
                        loop["in_tool_call"] = False
                        loop["cancel"].clear()

                    if cancelled:
                        # Cancelled this batch: return to the >>> prompt without
                        # re-querying the model.
                        break
                    # Loop back so the model sees the tool results and can answer.
                    continue

                # No tool requested: this is the final answer for the user turn.
                assistant_msg = {
                    "role": "assistant",
                    "content": "".join(turn.content),
                }
                # Feed the model's reasoning back into the conversation so it can
                # build on its own chain-of-thought on the next turn.  Ollama emits
                # reasoning in the "thinking" field, so send it back under the same key.
                if turn.thinking:
                    assistant_msg["thinking"] = "".join(turn.thinking)
                messages.append(assistant_msg)
                break


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
    p.add_argument("dir", nargs="?", default=None,
                   help="Optional starting directory: a sub-directory within "
                        "VM_ROOT_DIR; when given, the tools start there "
                        "instead of the root dir")
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
    log_cfg = LogConfig()

    with OllamaTunnel(ssh_cfg, ollama_cfg) as tunnel:
        tunnel.wait_until_ready()

        # Deploy tools.py to the VM's root dir and keep it alive for the same
        # lifetime as the harness, starting in that dir (or a sub-dir of it
        # when a starting directory is passed).  The context-manager __exit__
        # asks the remote
        # process to shut down gracefully (and waits for its confirmation) when
        # the with-block ends (normal exit, exception, or KeyboardInterrupt from
        # Ctrl+C in the chat).
        with RemoteTools(vm_ssh_cfg, tools_cfg, work_subdir=args.dir) as tools:
            # --- surface the tool manifest to the model ---
            # Ask the remote server for its manifest: one entry per tool
            # (in Ollama's /api/chat "tools" shape), each with its own
            # name, description, and parameters.  Forward it to chat, which
            # passes it to the model and routes any tool call back here.
            manifest = tools.get_manifest()
            names = ", ".join(
                str(tool.get("function", {}).get("name", "?"))
                for tool in manifest
            )
            print(f"[*] {len(manifest)} tool command(s) advertised to the model: {names}")

            def call_tool(
                name: str,
                arguments: dict[str, Any],
                *,
                cancel: threading.Event | None = None,
            ) -> Any:
                # Forward the model's tool call to the remote server.  `name` is
                # the tool name (e.g. "web_search"); `arguments` carries that
                # tool's parameters, which the server routes directly to the
                # matching function.  `cancel` lets the harness ask the server to
                # interrupt an in-flight call on a user Ctrl+C.
                return tools.call(name, **arguments, cancel=cancel)

            # Run the interactive chat session while tools is alive.
            chat(
                tunnel.base_url(),
                ollama_cfg.model,
                tools=manifest,
                call_tool=call_tool,
                log_dir=log_cfg.dir,
            )
        # ^ when this with-block exits (chat returns on Ctrl+C / EOF),
        #   RemoteTools.__exit__ -> stop() asks the server to shut down
        #   gracefully and waits for its confirmation.

    return 0


if __name__ == "__main__":
    sys.exit(main())

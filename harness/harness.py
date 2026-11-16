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

It also deploys the ``tools/`` folder to the VM and runs it as a background
RPC server for the lifetime of the harness.  When the harness exits
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
CLI flags (``--key``, ``--system``) override those settings or carry values
that are not part of ``local.env``.

The program opens an interactive chat: a ``>>>`` prompt reads your input,
forwards it to the model, prints the reply, and repeats.  Ctrl+C *at the
prompt* quits the session, while Ctrl+C *while the agent is working* (a model
stream or a tool call) cancels the in-flight request and returns to the
prompt (a Ctrl+C while the model is streaming keeps the user's prompt and the
partial response/thought already generated, plus a note that the turn was
interrupted).  Ctrl+D / EOF also quits.

Usage
-----
    python harness.py

This opens an interactive session: type a message at the ``>>>`` prompt, the
reply is printed, and you are prompted again.  Press Ctrl+C (or Ctrl+D) at the
prompt to quit; press Ctrl+C while the model is streaming to keep the partial
response and thought (plus an interruption note) and return to the prompt, or
while a tool call is running to finish that call and return to the prompt.

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
import contextlib
import json
import os
import signal
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple, Optional, TypeAlias, Union

import requests


# SSH tunneling, the remote-tools lifecycle, and their configuration live in
# remote.py; this module is the application layer (chat REPL + CLI).
from remote import (
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
    call_tool: Callable[[str, dict[str, Any]], Any],
    tool_call: dict[str, Any],
) -> str:
    """
    Run one model-requested tool call and return its result as a string.

    *call_tool* is the harness's forwarder to the remote tools server.  Its
    return value (often a JSON-friendly list/dict) is serialized to a string
    for the ``role: "tool"`` message; a tool error is reported back to the
    model as a JSON error object so it can recover instead of the session
    crashing.
    """
    function = tool_call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        result = call_tool(name, arguments)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    if isinstance(result, str):
        return result
    return json.dumps(result)


def _preview(text: str, limit: int = 200) -> str:
    """A single-line preview of a (possibly long) tool result for the console."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + " …"


# ---------------------------------------------------------------------------
# Ollama /api/chat
# ---------------------------------------------------------------------------
def chat(
    base_url: str,
    model: str,
    *,
    timeout: float = 300.0,
    system: Optional[str] = None,
    tools: list[dict[str, Any]] | None = None,
    call_tool: Callable[[str, dict[str, Any]], Any] | None = None,
) -> None:
    """
    Interactive REPL over /api/chat.

    Prints a ``>>>`` prompt, reads a user turn from stdin, POSTs the growing
    conversation to the remote Ollama server, streams the model's reply
    token-by-token, and repeats.  A Ctrl+C (or Ctrl+D / EOF) *at the prompt*
    ends the session.

    A Ctrl+C *while the agent is working* returns to the prompt instead of
    quitting: during a model stream the in-flight request is aborted but the
    user's prompt and the partial response/thought already generated are kept,
    together with a note that the turn was interrupted by the user; during a
    tool call the current call is allowed to finish, its result is appended to
    the conversation, and *then* the loop stops and returns to the prompt.

    When *tools* (the manifest the remote server advertised) is given, it is
    forwarded to the model under ``tools``; when the model requests a tool
    call, *call_tool* runs it and the result is fed back as a ``tool``
    message so the model can continue.  The model is re-queried in a loop
    until it stops requesting tools (the final answer).

    Ollama loads the model into memory on the first request if it is not
    already in RAM, so the first turn also doubles as a "start model" call.
    Conversation context is kept between turns.
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    print(f"[*] Chat with model '{model}'. Ctrl+C at the prompt quits; "
          "Ctrl+C during a response cancels the request and returns to the prompt.")

    # Probe the model to see whether it actually produces thinking / reasoning content.
    supports_think = model_supports_think(base_url, model)
    if supports_think:
        print(_grey(f"[*] Model '{model}' supports think (reasoning enabled)."))
    else:
        print(_grey(f"[*] Model '{model}' does not support think."))

    print()  # separator before the first prompt

    # A Ctrl+C during the agent loop should *cancel the current request* and
    # return to the >>> prompt rather than quitting the whole session.  We
    # install a SIGINT handler that behaves differently by phase:
    #
    #   * while the model is streaming (not inside a tool call) it raises
    #     KeyboardInterrupt, which _stream_turn turns into a _TurnInterrupted
    #     carrying the partial response; the loop below catches it to keep the
    #     partial turn and add an interruption note;
    #   * while a remote tool is running it only flags the cancellation
    #     instead of raising, so the blocking socket read is retried (PEP 475)
    #     and the tool runs to completion.  The loop then appends the tool's
    #     result before stopping and returning to the prompt.
    #
    # A Ctrl+C *at the >>> prompt* is in neither phase, so the handler raises
    # and input() ends the session as before.
    loop = {"in_tool_call": False, "cancel": False}

    def _on_sigint(signum: int, frame: Any) -> None:
        if loop["in_tool_call"]:
            loop["cancel"] = True
        else:
            raise KeyboardInterrupt

    # Scope the handler to this REPL: _signal_scope installs it for the block
    # body and restores the previous handler on exit (Ctrl+C / EOF at the
    # prompt, a return, or an exception), so the behaviour does not leak
    # past chat().
    with _signal_scope(_on_sigint):
        while True:
            # Read a user turn.  input() raises KeyboardInterrupt on Ctrl+C and
            # EOFError on Ctrl+D / end-of-input; either ends the session.
            try:
                prompt = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()                 # move to a fresh line after the interrupt
                print()
                return

            if not prompt:
                continue               # empty line: just prompt again

            messages.append({"role": "user", "content": prompt})
            loop["cancel"] = False

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
                            # Defer a Ctrl+C for the duration of this call: while this
                            # flag is set the SIGINT handler only records the
                            # cancellation (it does not raise), so the blocking read is
                            # retried and the tool finishes; the request is honoured
                            # below, after its result has been recorded.
                            loop["in_tool_call"] = True
                            try:
                                result = _invoke_tool(call_tool, tool_call)
                            finally:
                                loop["in_tool_call"] = False
                            print()  # blank line between the call and its result
                            print(_white(f"[*] Tool result: {name} {_preview(result)}"))
                            messages.append({"role": "tool", "content": result})
                            # A Ctrl+C that landed *during* this tool call: the result
                            # is recorded, but stop the batch and return to the prompt.
                            if loop["cancel"]:
                                print()
                                print(_grey("[*] Interrupted after the current tool call; returning to the prompt."))
                                cancelled = True
                                break
                    except KeyboardInterrupt:
                        # A Ctrl+C that landed *between* tool calls (not inside one)
                        # cancels the batch.  Any in-flight call was already deferred
                        # to completion, so nothing is left running.
                        print()
                        print(_grey("[*] Interrupted; returning to the prompt."))
                        cancelled = True
                    finally:
                        loop["in_tool_call"] = False
                        loop["cancel"] = False

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

        # Deploy the tools/ folder to the VM and keep it alive for the same
        # lifetime as the harness.  The context-manager __exit__ asks the
        # remote process to shut down gracefully (and waits for its
        # confirmation) when the with-block ends (normal exit, exception,
        # or KeyboardInterrupt from Ctrl+C in the chat).
        with RemoteTools(vm_ssh_cfg, tools_cfg) as tools:
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

            def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                # Forward the model's tool call to the remote server.
                # `name` is the tool name (e.g. "web_search"); `arguments`
                # carries that tool's parameters, which the server routes
                # directly to the matching function.
                return tools.call(name, **arguments)

            # Run the interactive chat session while tools is alive.
            chat(
                tunnel.base_url(),
                ollama_cfg.model,
                system=args.system,
                tools=manifest,
                call_tool=call_tool,
            )
        # ^ when this with-block exits (chat returns on Ctrl+C / EOF),
        #   RemoteTools.__exit__ -> stop() asks the server to shut down
        #   gracefully and waits for its confirmation.

    return 0


if __name__ == "__main__":
    sys.exit(main())

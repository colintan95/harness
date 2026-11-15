"""
tunnel.py – SSH port-forwarding and the remote-tools lifecycle.

This module owns everything that reaches a remote host over the system
``ssh`` binary:

* the ``local.env`` loader and the connection/service configuration
  dataclasses (``SSHConfig``, ``VMSSHConfig``, ``OllamaConfig``,
  ``ToolsConfig``) whose defaults it reads;
* the low-level ``ssh`` helpers (``build_tunnel_args``, ``_spawn_ssh``,
  ``_wait_port_ready``);
* :class:`OllamaTunnel`, which forwards a local port to a remote Ollama
  daemon via ``ssh -L ... -N``; and
* :class:`RemoteTools`, which forwards a local port *and* runs
  ``python3 ~/tools.py`` as the remote command over one ``ssh -L``
  session.

``harness.py`` is the application layer (the chat REPL + CLI) and uses
these.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional


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


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_LOCAL_ENV = _load_local_env(os.path.join(_PROJECT_DIR, "env", "local.env"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SSHConnection:
    """
    Connection details shared by every remote we reach over SSH.  Both the
    Ollama host (:class:`SSHConfig`) and the VM (:class:`VMSSHConfig`) build
    on this so the common ``ssh`` / ``scp`` argument construction lives in one
    place.  ``key_filename`` is empty when no key is configured, in which case
    we omit the ``-i`` flag and let ssh fall back to its own default key
    selection.
    """
    host: str
    port: int
    username: str
    key_filename: str

    @property
    def target(self) -> str:
        """The ``user@host`` destination shared by every invocation."""
        return f"{self.username}@{self.host}"

    @property
    def describe(self) -> str:
        """A one-line ``user@host:port`` summary for log lines."""
        return f"{self.username}@{self.host}:{self.port}"

    def base_args(
        self, *, binary: str = "ssh", port_flag: str = "-p"
    ) -> list[str]:
        """
        The connection half of an ``ssh`` / ``scp`` command: the binary, its
        port flag (``-p`` for ssh, ``-P`` for scp), the optional identity
        file, and our standard ``StrictHostKeyChecking`` option.  Callers
        append the destination (:attr:`target`, or ``target:path`` for scp)
        and any command.
        """
        args = [binary, port_flag, str(self.port)]
        if self.key_filename:
            args += ["-i", self.key_filename]
        args += ["-o", "StrictHostKeyChecking=accept-new"]
        return args


@dataclass
class SSHConfig(SSHConnection):
    host: str = _LOCAL_ENV.get("SSH_HOST", "127.0.0.1")
    port: int = 22  # always use the standard SSH port
    username: str = _LOCAL_ENV.get("SSH_USER", "deploy")
    key_filename: str = os.path.expanduser(_LOCAL_ENV.get("SSH_KEY", ""))


@dataclass
class OllamaConfig:
    """The remote Ollama endpoint and the local tunnel port."""
    remote_host: str = "127.0.0.1"   # Ollama listens on loopback *on the remote host*
    remote_port: int = int(_LOCAL_ENV.get("OLLAMA_API_PORT", "11434"))
    local_port: int = int(_LOCAL_ENV.get("OLLAMA_LOCAL_PORT", "18434"))
    model: str = _LOCAL_ENV.get("OLLAMA_MODEL", "llama3")


# ---------------------------------------------------------------------------
# VM configuration – SSH connection to a VM
# ---------------------------------------------------------------------------

@dataclass
class VMSSHConfig(SSHConnection):
    """SSH connection details for the VM (mirrors :class:`SSHConfig`)."""
    host: str = _LOCAL_ENV.get("VM_SSH_HOST", "127.0.0.1")
    port: int = 22
    username: str = _LOCAL_ENV.get("VM_SSH_USER", "deploy")
    key_filename: str = os.path.expanduser(_LOCAL_ENV.get("VM_SSH_KEY", ""))


# ---------------------------------------------------------------------------
# Remote tools configuration
# ---------------------------------------------------------------------------

@dataclass
class ToolsConfig:
    """
    TCP ports for the remote ``tools.py`` RPC server.

    * *remote_port* – the port ``tools.py`` listens on (on the VM).
    * *local_port*  – the local port the harness forwards to via ``ssh -L``.
    """
    remote_port: int = int(_LOCAL_ENV.get("TOOLS_REMOTE_PORT", "18435"))
    local_port: int = int(_LOCAL_ENV.get("TOOLS_LOCAL_PORT", "18436"))


# ---------------------------------------------------------------------------
# SSH tunnel – shells out to the system ``ssh`` binary
# ---------------------------------------------------------------------------

def _spawn_ssh(
    cmd: Sequence[str],
    *,
    settle: float = 0.5,
    fail_msg: str = "ssh exited immediately",
) -> subprocess.Popen[Any]:
    """
    Start *cmd* as a detached ``ssh`` process and fail fast if it dies on
    startup.  ``start_new_session=True`` detaches ssh into its own session /
    process group so a Ctrl+C (SIGINT to the terminal's foreground group)
    does not reach the tunnel; that keeps it alive long enough for a graceful
    shutdown to run.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    # Give ssh a brief moment to fail-fast (bad host, auth error, port in use)
    time.sleep(settle)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"{fail_msg} (code {proc.returncode}):\n{err}")
    return proc


def build_tunnel_args(
    conn: SSHConnection,
    local_port: int,
    remote_host: str,
    remote_port: int,
    *,
    remote_cmd: str | None = None,
) -> list[str]:
    """
    Build the argv for an ``ssh -L`` connection.  When *remote_cmd* is given
    it runs on the remote host (the forwarded port and the command share one
    SSH session); otherwise ``-N`` is passed so ssh only forwards the port.
    """
    args = conn.base_args()
    args += ["-L", f"{local_port}:{remote_host}:{remote_port}"]
    args += ["-o", "ExitOnForwardFailure=yes", conn.target]
    if remote_cmd:
        args.append(remote_cmd)
    else:
        args.append("-N")
    return args


def _wait_port_ready(
    local_port: int,
    proc: Optional[subprocess.Popen[Any]],
    *,
    timeout: float = 15.0,
    interval: float = 0.5,
    announce: bool = True,
    prefix: str = "",
) -> None:
    """
    Poll *local_port* with a raw TCP connect until it accepts connections or
    *timeout* elapses, raising if the backing *proc* dies meanwhile.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc and proc.poll() is not None:
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(f"{prefix}tunnel process died:\n{err}")
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=2):
                if announce:
                    print(f"[OK] {prefix}port {local_port} is accepting connections.")
                return
        except OSError:
            pass
        time.sleep(interval)
    raise TimeoutError(f"{prefix}port {local_port} not reachable within {timeout}s")


class OllamaTunnel:
    """
    Starts ``ssh -L local_port:remote_host:remote_port user@host -N`` as a
    background subprocess.  While the tunnel is alive every TCP connection to
    ``127.0.0.1:local_port`` on the *local* machine is forwarded through SSH
    to ``remote_host:remote_port`` on the *remote* machine.

    Usage::

        with OllamaTunnel(ssh_cfg, ollama_cfg) as tunnel:
            tunnel.wait_until_ready()
            chat(tunnel.base_url(), ollama_cfg.model)
    """

    def __init__(self, ssh_cfg: SSHConfig, ollama_cfg: OllamaConfig):
        self.ssh_cfg = ssh_cfg
        self.ollama_cfg = ollama_cfg
        self.local_port = ollama_cfg.local_port
        self._proc: Optional[subprocess.Popen[Any]] = None

    # -- context-manager ---------------------------------------------------

    def __enter__(self) -> "OllamaTunnel":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Spawn ``ssh -L ...`` as a background process."""
        cmd = build_tunnel_args(
            self.ssh_cfg,
            self.local_port,
            self.ollama_cfg.remote_host,
            self.ollama_cfg.remote_port,
        )

        target = f"{self.ollama_cfg.remote_host}:{self.ollama_cfg.remote_port} (on remote)"
        print(f"[*] SSH  -> {self.ssh_cfg.describe}")
        print(f"[*] Tunnel 127.0.0.1:{self.local_port} -> {target}")

        # _spawn_ssh detaches ssh into its own process group so a Ctrl+C does
        # not reach this tunnel; that keeps it alive long enough for the
        # graceful shutdown below to run.
        self._proc = _spawn_ssh(cmd, fail_msg="ssh failed to start")

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
        _wait_port_ready(
            self.local_port, self._proc,
            timeout=timeout, interval=interval, prefix="Tunnel ",
        )


# ---------------------------------------------------------------------------
# Remote tools – deploy tools.py to the VM and manage its lifecycle
# ---------------------------------------------------------------------------

class RemoteTools:
    """
    Deploy ``tools.py`` to the VM and run it as a direct (non-nohup) RPC
    server tied to the SSH connection, so it cannot outlive the harness.

    * :meth:`start` copies ``tools.py`` to ``~`` on the VM and opens a
      single ``ssh -L`` connection that both forwards a local port and runs
      ``python3 ~/tools.py`` as the remote command.
    * :meth:`call` connects through the forward and sends a JSON-RPC
      request, returning the ``result`` value.
    * :meth:`stop` (or the context-manager ``__exit__``) first asks the
      remote ``tools.py`` to shut down and waits for its confirmation,
      then closes the SSH session as a backstop.

    Usage::

        with RemoteTools(vm_cfg, tools_cfg) as tools:
            greeting = tools.call("hello")   # -> "Hello, World!"
    """

    def __init__(
        self,
        vm_cfg: VMSSHConfig,
        tools_cfg: ToolsConfig | None = None,
    ) -> None:
        self._vm = vm_cfg
        self._tools_cfg = tools_cfg or ToolsConfig()
        self._local_port = self._tools_cfg.local_port
        self._remote_port = self._tools_cfg.remote_port
        self._tunnel: Optional[subprocess.Popen[Any]] = None
        # Path to the local tools.py that gets copied to the VM.
        self._tools_path = os.path.join(_SRC_DIR, "tools.py")

    # -- context-manager ---------------------------------------------------

    def __enter__(self) -> "RemoteTools":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- SSH helpers -------------------------------------------------------

    def _ssh(
        self,
        cmd: str,
        *,
        timeout: float = 30.0,
        check: bool = True,
    ) -> str:
        """
        Run *cmd* on the VM over SSH and return the stripped stdout.

        When *check* is True (the default) a non-zero exit raises
        ``RuntimeError``.  When *check* is False the call is best-effort
        and the returncode is simply ignored.
        """
        argv = self._vm.base_args() + [self._vm.target, cmd]
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if check and proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"[tools] remote command failed ({cmd!r}): "
                f"exit {proc.returncode}\n{err}"
            )
        out = proc.stdout.decode(errors="replace").strip()
        return out

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Copy tools.py to the VM and start it via a single SSH connection."""
        # 1. Copy the script to ~ on the VM.
        self._scp()

        # 2. Open a single SSH connection that both runs tools.py on the
        #    VM and forwards the port locally.
        self._open_tunnel()

    def _scp(self) -> None:
        """Copy the local ``tools.py`` to ``~/tools.py`` on the VM."""
        dest = f"{self._vm.target}:~/tools.py"
        cmd = self._vm.base_args(binary="scp", port_flag="-P")
        cmd += [self._tools_path, dest]
        print(f"[*] [tools] scp {os.path.basename(self._tools_path)} -> {dest}")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace")
            raise RuntimeError(f"[tools] scp failed:\n{err}")
        print(f"[OK] [tools] copied to ~.")

    def _open_tunnel(self) -> None:
        """
        Start a single ``ssh -L`` connection that both forwards a local port
        and runs ``python3 ~/tools.py`` as the remote command.  When this
        SSH process is terminated, the remote ``tools.py`` dies with it.
        """
        remote_cmd = f"python3 ~/tools.py --port {self._remote_port}"
        cmd = build_tunnel_args(
            self._vm, self._local_port, "127.0.0.1", self._remote_port,
            remote_cmd=remote_cmd,
        )
        print(f"[*] [tools] ssh -L 127.0.0.1:{self._local_port} -> "
              f"127.0.0.1:{self._remote_port} (running {remote_cmd})")

        # _spawn_ssh detaches ssh into its own process group so a Ctrl+C does
        # not reach this tunnel; that keeps it alive long enough for the
        # graceful "shutdown" handshake in stop() to complete, after which
        # _close_tunnel() tears it down explicitly.
        self._tunnel = _spawn_ssh(cmd, fail_msg="[tools] ssh exited immediately")

        _wait_port_ready(
            self._local_port, self._tunnel,
            timeout=15.0, interval=0.3, announce=False,
        )
        print(f"[OK] [tools] running on 127.0.0.1:{self._local_port}")

    # -- RPC call ----------------------------------------------------------

    def _send_request(
        self, request: dict[str, Any], *, timeout: float = 10.0,
    ) -> str:
        """
        Open a fresh connection to the local forwarded port, send one
        newline-delimited JSON *request*, and return the raw response
        line (the server's single JSON object, stripped).  Returns an
        empty string when the server closes the connection without
        replying.

        *timeout* bounds both the connect and the (blocking) read so a
        hung server cannot block the caller forever.
        """
        payload = json.dumps(request).encode("utf-8") + b"\n"
        with socket.create_connection(
            ("127.0.0.1", self._local_port), timeout=timeout,
        ) as conn:
            conn.sendall(payload)
            conn.settimeout(timeout)
            # Read the response (newline-delimited JSON).
            buf = bytearray()
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
            return buf.decode("utf-8", errors="replace").strip()

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """
        Send a JSON-RPC request to the remote ``tools.py`` server and
        return the ``result`` field.

        The wire protocol is one newline-delimited JSON object per
        connection::

            request  ->  {"method": "hello", "args": [...], "kwargs": {}}
            response ->  {"result": <value>, "error": null}

        Raises ``RuntimeError`` when the server returns an error, the
        connection fails, or the server closes without replying.
        """
        request: dict[str, Any] = {"method": method}
        if args:
            request["args"] = list(args)
        if kwargs:
            request["kwargs"] = kwargs

        raw = self._send_request(request)
        if not raw:
            raise RuntimeError(f"[tools] empty response from {method!r}")

        response = json.loads(raw)
        err = response.get("error")
        if err:
            raise RuntimeError(f"[tools] {method}() error: {err}")
        return response.get("result")

    # -- teardown ----------------------------------------------------------

    def request_shutdown(self, *, timeout: float = 10.0) -> bool:
        """
        Gracefully stop the remote ``tools.py`` by message.

        Sends the ``shutdown`` control request and waits for the tool to
        reply with a confirmation that it is about to shut down.  Returns
        ``True`` when a confirmation was received; ``False`` when it was
        not (connection error, empty/unparseable reply, or a reported
        error) — in which case the caller should fall back to killing the
        SSH session.
        """
        print("[*] [tools] requesting graceful shutdown ...")
        try:
            raw = self._send_request({"method": "shutdown"}, timeout=timeout)
        except OSError as exc:
            print(
                f"[!] [tools] shutdown request failed ({exc!r}); "
                "falling back to terminating the SSH session.",
                file=sys.stderr,
            )
            return False

        if not raw:
            print(
                "[!] [tools] no confirmation received; "
                "falling back to terminating the SSH session.",
                file=sys.stderr,
            )
            return False

        try:
            response = json.loads(raw)
        except ValueError:
            print(
                f"[!] [tools] unparseable shutdown reply ({raw!r}); "
                "falling back to terminating the SSH session.",
                file=sys.stderr,
            )
            return False

        err = response.get("error")
        if err:
            print(
                f"[!] [tools] shutdown reported an error: {err}; "
                "falling back to terminating the SSH session.",
                file=sys.stderr,
            )
            return False

        print(f"[OK] [tools] confirmed shutting down: "
              f"{response.get('result')!r}")
        return True

    def _close_tunnel(self, *, grace: float = 5.0) -> None:
        """
        Close the SSH session that runs the remote ``tools.py``.

        Prefers to *wait* for the process to exit on its own — it
        should, now that ``tools.py`` has honoured the ``shutdown``
        request and finished.  Only if it is still alive after *grace*
        seconds do we terminate it, escalating to ``kill`` if it still
        lingers.
        """
        proc = self._tunnel
        if proc is None:
            return
        exited = False
        try:
            proc.wait(timeout=grace)
            exited = True
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        tag = "exited on its own" if exited else "terminated"
        print(f"[*] [tools] ssh session closed ({tag}).")
        self._tunnel = None

    def stop(self) -> None:
        """
        Gracefully stop the remote ``tools.py``, then close the SSH
        session.

        Two steps, in order:

        1. Send a ``shutdown`` control message and wait for the tool to
           confirm it is about to shut down — the observable handshake
           that proves the tool is really stopping.
        2. Close the SSH session, preferring a natural exit and only
           force-terminating as a backstop.
        """
        self.request_shutdown()
        self._close_tunnel()

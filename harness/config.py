"""
config.py – environment configuration for the harness.

This module owns the ``local.env`` loader and the configuration dataclasses
whose defaults it reads from that file:

* ``SSHConfig`` / ``VMSSHConfig`` – the SSH connections (the Ollama host and
  the VM, built on the shared ``SSHConnection`` base);
* ``OllamaConfig`` – the remote Ollama endpoint and the local tunnel port;
* ``ToolsConfig`` – the remote ``tools.py`` RPC server (its root dir, ports,
  and the Python dependencies it needs on the VM); and
* ``LogConfig`` – the local directory chat transcripts are written to on
  quit (empty disables logging).

Both :mod:`remote` (the SSH tunnel + remote-tools lifecycle) and
:mod:`harness` (the chat REPL + CLI) import these, so the env-driven
configuration lives in one place rather than being split between the
transport layer and the application layer.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass

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


# config.py lives at <project>/harness/config.py, so the project root is one
# level up.  The tools folder (deployed to the VM) sits at the project root,
# next to the harness package.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))   # <project>/harness
_PROJECT_DIR = os.path.dirname(_PKG_DIR)                # <project>
_LOCAL_ENV = _load_local_env(os.path.join(_PROJECT_DIR, "env", "local.env"))
# The local tools folder; we deploy its single self-contained module
# (tools.py) into the VM's root dir rather than the whole folder.  The
# *_DIR / _*_ constants below are used only here; TOOLS_FILE is the deploy
# source, so it is public (no leading underscore) -- it is the one path
# constant the remote layer imports.
_TOOLS_DIR = os.path.join(_PROJECT_DIR, "tools")        # <project>/tools
TOOLS_FILE = os.path.join(_TOOLS_DIR, "tools.py")       # <project>/tools/tools.py


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
    login_shell: bool = False

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

    def wrap_remote_cmd(self, cmd: str) -> str:
        """
        Wrap *cmd* so it runs in a login shell on the remote host.

        When :attr:`login_shell` is *True* the command is wrapped in
        ``bash -lc`` so that the user's login profile
        (``.bash_profile``, ``.profile``, ``.bash_login``) is sourced
        before the command runs, giving it the same environment as an
        interactive login session.  When *False* the command is returned
        unchanged.
        """
        if self.login_shell:
            return f"bash -lc {shlex.quote(cmd)}"
        return cmd


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
    login_shell: bool = True


# ---------------------------------------------------------------------------
# Remote tools configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDependency:
    """
    A Python dependency the remote ``tools.py`` server needs *on the VM*.

    The pip distribution name and the import name are stored separately
    because they differ for some packages (here: pip installs
    ``duckduckgo-search`` but the module is imported as ``ddgs``):

    * *requirement* – the pip spec used to install it when it is missing
      (e.g. ``"duckduckgo-search"``).
    * *import_name* – the module probed to detect whether it is already
      installed (e.g. ``"ddgs"``); a successful ``import`` means "reuse the
      existing install".
    """
    requirement: str
    import_name: str


@dataclass
class ToolsConfig:
    """
    Configuration for the remote ``tools.py`` RPC server.

    * *root_dir*    – the directory on the VM where ``tools.py`` is stored and
      where the tools' working directory starts.  A leading ``~`` (the
      default) is accepted and is escaped to ``$HOME`` for the remote-shell
      commands so the VM's login shell expands it to the user's home
      (``scp`` expands a leading ``~`` on its own).
    * *remote_port* – the port ``tools.py`` listens on (on the VM).
    * *local_port*  – the local port the harness forwards to via ``ssh -L``.
    * *dependencies* – the Python packages the server imports on the VM.
      The harness ensures each is installed *on the VM* before starting the
      server, installing only the ones that are missing so a later run reuses
      the existing installation (see
      :meth:`RemoteTools._ensure_dependencies`).
    """
    root_dir: str = _LOCAL_ENV.get("VM_ROOT_DIR", "~")
    remote_port: int = int(_LOCAL_ENV.get("TOOLS_REMOTE_PORT", "18435"))
    local_port: int = int(_LOCAL_ENV.get("TOOLS_LOCAL_PORT", "18436"))
    dependencies: tuple[ToolDependency, ...] = (
        # duckduckgo-search backs the web-search tool; its import name is
        # "ddgs".  Installed on the VM on first run, reused on every run
        # afterwards.
        ToolDependency(requirement="ddgs", import_name="ddgs"),
    )


@dataclass
class LogConfig:
    """
    Chat-transcript logging.

    * *dir* – the folder the harness writes a JSON transcript of each chat
      session to when the user quits.  A leading ``~`` is expanded to the
      home directory (as with ``SSH_KEY`` / ``VM_SSH_KEY``); when empty --
      the default, i.e. no ``LOG_DIR`` in ``local.env`` -- logging is
      disabled and nothing is written.
    """
    dir: str = os.path.expanduser(_LOCAL_ENV.get("LOG_DIR", ""))

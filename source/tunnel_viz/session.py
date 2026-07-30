"""Connect to RS3, reusing a live scripting server / open model when possible."""

from __future__ import annotations

import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from rs3.Model import Model
    from rs3.RS3Modeler import RS3Modeler


# Ports commonly used by RS3 scripting / other local tools.
_PORT_SCAN_SPAN = 20
# RS3 can take several minutes to launch a second scripting-enabled instance.
_DEFAULT_OPEN_TIMEOUT_S = 600.0


@dataclass
class RS3Session:
    """Active RS3 scripting connection and ownership flags for cleanup."""

    modeler: "RS3Modeler"
    model: "Model"
    model_path: Path
    port: int
    started_application: bool
    opened_model: bool
    model_was_already_open: bool

    def close(self, *, keep_open: bool = False) -> None:
        """
        Release resources we own.

        - Never close a model the user already had open.
        - Never quit RS3 unless this session started it (and keep_open is False).
        """
        if keep_open:
            return

        if self.opened_model and not self.model_was_already_open:
            try:
                self.model.close(False)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: model.close failed: {exc}", flush=True)

        if self.started_application:
            try:
                self.modeler.closeProgram(saveModels=False)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: closeProgram failed: {exc}", flush=True)


def _call_with_timeout(fn: Callable, timeout_s: float, label: str):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            raise TimeoutError(
                f"Timed out after {timeout_s:g}s while {label}. "
                "RS3 may be busy or the scripting call is stuck."
            ) from exc


def _open_model_with_timeout(
    modeler: "RS3Modeler",
    path_str: str,
    *,
    already_open: bool,
    timeout_s: float,
    progress_callback,
) -> "Model":
    from rs3.Model import Model
    from rs3.RS3Modeler import RS3ModelerBase

    if already_open:
        project_id = _call_with_timeout(
            lambda: RS3ModelerBase.openFile(modeler, path_str),
            timeout_s,
            f"attaching to already-open model ({path_str})",
        )
        return Model(modeler._client, project_id)

    project_id = _call_with_timeout(
        lambda: RS3ModelerBase.openFile(modeler, path_str),
        timeout_s,
        f"requesting open of {path_str}",
    )

    deadline = time.monotonic() + timeout_s
    while True:
        loaded = _call_with_timeout(
            lambda: modeler._isViewLoaded(path_str),
            min(30.0, max(1.0, deadline - time.monotonic())),
            "checking whether the model view finished loading",
        )
        if loaded:
            return Model(modeler._client, project_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_s:g}s waiting for RS3 to finish "
                f"loading {path_str}."
            )
        progress_callback(
            f"Waiting for model view to load... ({remaining:.0f}s remaining)"
        )
        time.sleep(0.5)


def _server_is_running(port: int) -> bool:
    """Return True only if the RS3 scripting server answers a ping on ``port``."""
    from rs3.RS3Modeler import RS3Modeler

    try:
        return bool(RS3Modeler._isServerRunning(port))
    except Exception:  # noqa: BLE001 - connection refused / version / unavailable
        return False


def _port_is_free(port: int) -> bool:
    """True if nothing is listening/bound on localhost:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _find_free_port(preferred: int, *, scan_span: int = _PORT_SCAN_SPAN) -> int:
    """Pick ``preferred`` if free, else the next free port nearby."""
    candidates = [preferred]
    for offset in range(1, scan_span + 1):
        for port in (preferred + offset, preferred - offset):
            if 49152 <= port <= 65535 and port not in candidates:
                candidates.append(port)
    for port in candidates:
        if _port_is_free(port) and not _server_is_running(port):
            return port
    raise RuntimeError(
        f"No free scripting port near {preferred}. "
        "Stop an unused RS3 scripting server or pick another Port in the UI."
    )


def _list_rs3_pids() -> list[int]:
    """Return PIDs of running RS3.exe processes (Windows)."""
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Process -Name RS3 -ErrorAction SilentlyContinue)."
                "Id -join ','",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return []
    raw = (completed.stdout or "").strip()
    if not raw:
        return []
    pids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            pids.append(int(part))
    return pids


def _find_scripting_port(
    preferred: int,
    *,
    scan_span: int = _PORT_SCAN_SPAN,
    progress_callback=None,
) -> int | None:
    """
    Return the first port with a live RS3 scripting server.

    Checks ``preferred`` first, then nearby ports (other tools may have
    started RS3 on a non-default port).
    """
    progress = progress_callback or (lambda _msg: None)
    candidates = [preferred]
    for offset in range(1, scan_span + 1):
        for port in (preferred + offset, preferred - offset):
            if 49152 <= port <= 65535 and port not in candidates:
                candidates.append(port)

    progress(f"Looking for RS3 scripting server (preferred port {preferred})…")
    for port in candidates:
        if _server_is_running(port):
            if port != preferred:
                progress(
                    f"No scripting server on {preferred}; "
                    f"found one on port {port} — using that."
                )
            else:
                progress(f"Scripting server responding on port {port}.")
            return port
    return None


def _enable_scripting_on_open_rs3_message(port: int, pids: list[int]) -> str:
    """Instructions that do NOT require closing the user's other project."""
    pid_txt = ", ".join(str(p) for p in pids) if pids else "unknown"
    return (
        f"RS3 is already running (PID {pid_txt}) without a scripting server, "
        "and a second scripting-enabled instance could not be started "
        "(RS3 often reuses the existing window instead of launching another).\n\n"
        "You do NOT need to close your other project. On the open RS3 window:\n"
        "  1. Scripting → Manage Scripting Server\n"
        f"  2. Set port to {port} (or click Find Port)\n"
        "  3. Click Start Server\n"
        "  4. Set the Port field in this app to match, then click Extract again.\n\n"
        "That enables scripting on your current RS3 session; Extract will open "
        "the tunnel model alongside your other project."
    )


def _start_scripting_application(
    port: int,
    *,
    timeout_s: float | None,
    progress_callback,
) -> None:
    """Launch RS3 with ``-startScriptingServer`` and wait until it answers."""
    from rs3.RS3Modeler import RS3Modeler

    if timeout_s is None or timeout_s <= 0:
        progress_callback(
            f"Starting RS3 with scripting on port {port} "
            "(no timeout — waiting until the server is ready)…"
        )
        # ApplicationManager treats a falsy timeout as "wait forever".
        RS3Modeler.startApplication(port=port, timeout=None)  # type: ignore[arg-type]
    else:
        progress_callback(
            f"Starting RS3 with scripting on port {port} "
            f"(waiting up to {timeout_s:g}s)…"
        )
        RS3Modeler.startApplication(port=port, timeout=timeout_s)

    if not _server_is_running(port):
        # Extra grace poll in case the API returned slightly early.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not _server_is_running(port):
            time.sleep(0.5)
    if not _server_is_running(port):
        waited = "indefinitely" if not timeout_s else f"within {timeout_s:g}s"
        raise TimeoutError(
            f"RS3 did not expose a scripting server on port {port} {waited}."
        )
    progress_callback(f"Scripting server ready on port {port}.")


def connect_model(
    model_path: Path,
    *,
    port: int = 60064,
    force_no_start: bool = False,
    open_timeout_s: float | None = _DEFAULT_OPEN_TIMEOUT_S,
    progress_callback=print,
) -> RS3Session:
    """
    Attach to RS3 and return a handle to ``model_path``.

    1. Prefer an already-running scripting server on ``port`` (or nearby).
    2. If RS3.exe is open but scripting is off, try a second instance on a
       free port (short timeout). If that fails, tell the user how to Start
       Server on the existing RS3 (no need to close the other project).
    3. Otherwise start RS3 with scripting enabled (unless ``force_no_start``).
    4. Reuse the model view if it is already loaded; otherwise open the file.
    """
    from .rs3_bootstrap import ensure_rs3_protobuf_imports

    ensure_rs3_protobuf_imports()

    from rs3.RS3Modeler import RS3Modeler

    path = model_path.resolve()
    path_str = str(path)

    rs3_pids_before = _list_rs3_pids()
    progress_callback(
        "Connect strategy: attach if scripting is up; else try a second "
        "instance on a free port; else ask you to Start Server in RS3 "
        "(keep other projects open)."
    )
    if rs3_pids_before:
        progress_callback(
            "Detected RS3 process(es): "
            + ", ".join(str(p) for p in rs3_pids_before)
        )
    else:
        progress_callback("No RS3.exe process detected.")

    found_port = _find_scripting_port(port, progress_callback=progress_callback)
    started_application = False

    if found_port is not None:
        port = found_port
        progress_callback(
            f"RS3 scripting server already running on port {port}; attaching..."
        )
    elif force_no_start:
        raise RuntimeError(
            f"No RS3 scripting server on port {port}, and --no-start was set.\n"
            "Either start RS3 with scripting enabled, omit --no-start so this "
            "app can launch RS3, or in RS3 use Scripting → Manage Scripting Server."
        )
    elif rs3_pids_before:
        # Keep the user's other project open. Prefer a free port so we don't
        # collide, then try a second scripting-enabled instance briefly.
        launch_port = _find_free_port(port)
        progress_callback(
            "RS3 is open without scripting. "
            f"Trying a second instance on free port {launch_port}…"
        )
        try:
            _start_scripting_application(
                launch_port,
                timeout_s=open_timeout_s,
                progress_callback=progress_callback,
            )
            port = launch_port
            started_application = True
            pids_after = _list_rs3_pids()
            new_pids = [p for p in pids_after if p not in set(rs3_pids_before)]
            if new_pids:
                progress_callback(
                    "Second RS3 instance started "
                    f"(new PID {', '.join(str(p) for p in new_pids)})."
                )
            else:
                progress_callback(
                    "Scripting server is up (RS3 may have attached scripting "
                    "to the existing window)."
                )
        except Exception as exc:  # noqa: BLE001
            late = _find_scripting_port(port, progress_callback=progress_callback)
            if late is not None:
                port = late
                progress_callback(
                    f"Second-instance launch failed, but a scripting server "
                    f"is now on port {port}; continuing."
                )
            else:
                raise RuntimeError(
                    _enable_scripting_on_open_rs3_message(launch_port, rs3_pids_before)
                    + f"\n\nSecond-instance attempt detail: {exc}"
                ) from exc
    else:
        launch_port = port if _port_is_free(port) else _find_free_port(port)
        if launch_port != port:
            progress_callback(
                f"Port {port} is busy; starting RS3 on free port {launch_port}."
            )
        try:
            _start_scripting_application(
                launch_port,
                timeout_s=open_timeout_s,
                progress_callback=progress_callback,
            )
            port = launch_port
            started_application = True
        except Exception as exc:  # noqa: BLE001
            late = _find_scripting_port(port, progress_callback=progress_callback)
            if late is not None:
                port = late
                progress_callback(
                    f"startApplication reported an error, but a scripting "
                    f"server is now up on port {port}; continuing."
                )
            else:
                raise RuntimeError(
                    f"Could not start RS3 scripting server on port {launch_port}.\n"
                    f"Details: {exc}"
                ) from exc

    modeler = RS3Modeler(port=port)
    model_timeout = (
        _DEFAULT_OPEN_TIMEOUT_S
        if open_timeout_s is None or open_timeout_s <= 0
        else float(open_timeout_s)
    )

    already_open = False
    try:
        already_open = bool(
            _call_with_timeout(
                lambda: modeler._isViewLoaded(path_str),
                min(30.0, model_timeout),
                "checking if the model is already open",
            )
        )
    except Exception as exc:  # noqa: BLE001
        progress_callback(f"Could not check if model is open ({exc}); opening file...")

    if already_open:
        progress_callback(f"Model already open; reusing: {path}")
    else:
        progress_callback(f"Opening {path}...")

    model = _open_model_with_timeout(
        modeler,
        path_str,
        already_open=already_open,
        timeout_s=model_timeout,
        progress_callback=progress_callback,
    )

    return RS3Session(
        modeler=modeler,
        model=model,
        model_path=path,
        port=port,
        started_application=started_application,
        opened_model=True,
        model_was_already_open=already_open,
    )

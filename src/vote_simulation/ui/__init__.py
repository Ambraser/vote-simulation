"""UI package for vote_simulation — interface Streamlit."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _has_explicit_streamlit_port(args: list[str]) -> bool:
    return any(arg == "--server.port" or arg.startswith("--server.port=") for arg in args)


def _extract_port_from_args(args: list[str]) -> int | None:
    for i, arg in enumerate(args):
        if arg == "--server.port" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
        if arg.startswith("--server.port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _find_available_port(start_port: int = 8501, max_attempts: int = 50) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    msg = f"No free Streamlit port found between {start_port} and {start_port + max_attempts - 1}."
    raise RuntimeError(msg)


def _wait_and_open_browser(port: int, process: subprocess.Popen) -> None:
    """Wait for the Streamlit server to be ready, then open the browser."""
    health_urls = [
        f"http://localhost:{port}/_stcore/health",
        f"http://localhost:{port}/healthz",
    ]
    deadline = time.time() + 60
    while time.time() < deadline:
        if process.poll() is not None:
            return  # Process already exited, don't open browser
        for url in health_urls:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        time.sleep(0.3)  # Brief extra pause for full readiness
                        webbrowser.open(f"http://localhost:{port}")
                        return
            except Exception:
                pass
        time.sleep(0.3)
    # Timeout reached — open anyway
    webbrowser.open(f"http://localhost:{port}")


def launch() -> None:
    """Lance l'interface Streamlit vote_simulation.

    Ouvre automatiquement le navigateur quand le serveur est prêt.
    Tapez 'stop' ou 's' + Entrée dans le terminal pour arrêter le serveur.
    """
    app_path = Path(__file__).parent / "app.py"
    extra_args = sys.argv[1:]
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]

    # Determine port (once, to avoid double-call to _find_available_port)
    if "STREAMLIT_SERVER_PORT" in os.environ:
        try:
            port = int(os.environ["STREAMLIT_SERVER_PORT"])
        except ValueError:
            port = 8501
    elif _has_explicit_streamlit_port(extra_args):
        port = _extract_port_from_args(extra_args) or 8501
    else:
        port = _find_available_port()
        if port != 8501:
            print(f"Port 8501 occupe, lancement de vote-sim-ui sur le port {port}.")
        command.extend(["--server.port", str(port)])

    command.extend(extra_args)

    print(f"\n[vote-sim-ui] Demarrage du serveur sur http://localhost:{port}")
    print("[vote-sim-ui] Tapez 'stop' ou 's' puis Entree pour arreter le serveur.")
    print("[vote-sim-ui] Vous pouvez aussi fermer l'onglet du navigateur.\n")

    # Start Streamlit as a background process (non-blocking)
    process = subprocess.Popen(command)

    # Open the browser in a thread (waits for server ready first)
    browser_thread = threading.Thread(
        target=_wait_and_open_browser,
        args=(port, process),
        daemon=True,
    )
    browser_thread.start()

    # Monitor stdin for "stop" or "s" command
    def _stdin_watcher() -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:  # EOF (e.g. piped input ended)
                    break
                if line.strip().lower() in ("stop", "s"):
                    print("\n[vote-sim-ui] Arret du serveur...")
                    process.terminate()
                    break
        except Exception:
            pass

    watcher = threading.Thread(target=_stdin_watcher, daemon=True)
    watcher.start()

    # Wait for the Streamlit process to exit (triggered by stop command, tab close, or error)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[vote-sim-ui] Interruption recue. Arret du serveur...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    print("\n[vote-sim-ui] Serveur arrete.")


__all__ = ["launch"]

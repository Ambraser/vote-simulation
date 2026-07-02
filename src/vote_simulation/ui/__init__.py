"""UI package for vote_simulation — interface Streamlit."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


def _has_explicit_streamlit_port(args: list[str]) -> bool:
    return any(arg == "--server.port" or arg.startswith("--server.port=") for arg in args)


def _find_available_port(start_port: int = 8501, max_attempts: int = 50) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    msg = f"No free Streamlit port found between {start_port} and {start_port + max_attempts - 1}."
    raise RuntimeError(msg)


def launch() -> None:
    """Lance l'interface Streamlit vote_simulation.

    Equivalent à :
        streamlit run src/vote_simulation/ui/app.py
    """
    app_path = Path(__file__).parent / "app.py"
    extra_args = sys.argv[1:]
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]

    if "STREAMLIT_SERVER_PORT" not in os.environ and not _has_explicit_streamlit_port(extra_args):
        port = _find_available_port()
        if port != 8501:
            print(f"Port 8501 occupe, lancement de vote-sim-ui sur le port {port}.")
        command.extend(["--server.port", str(port)])

    command.extend(extra_args)
    subprocess.run(
        command,
        check=True,
    )


__all__ = ["launch"]

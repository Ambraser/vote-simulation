"""UI package for vote_simulation — interface Streamlit."""

from __future__ import annotations


def launch() -> None:
    """Lance l'interface Streamlit vote_simulation.

    Equivalent à :
        streamlit run src/vote_simulation/ui/app.py
    """
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


__all__ = ["launch"]

"""Subprocess worker for vote_simulation UI.

This module MUST remain importable without Streamlit (no ``st`` import, no
session_state access).  It is executed in a child process via multiprocessing:

- **POSIX** (Linux / macOS): ``fork`` context — the child inherits the parent's
  memory space, so the worker function does not need to be in a separate module.
  We keep it here anyway for symmetry.
- **Windows**: ``spawn`` context — Python starts a *fresh* interpreter that
  imports the target function from its fully-qualified module path.  The function
  therefore *must* be defined at the top level of an importable module (not inside
  ``__main__`` or a Streamlit script), which is exactly what this file provides.
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile
from typing import Any


def simulation_worker(
    config_path: str,
    reload: bool,
    mp_queue: Any,
    mp_stop: Any,
) -> None:
    """Run ``simulation_series_from_config_2`` in a subprocess and stream progress.

    Monkey-patches the ``tqdm`` symbol used by the simulation module so that
    each ``update()`` call emits a ``("progress", current, total)`` message on
    *mp_queue* instead of printing to stdout.  Internal tqdms with
    ``disable=True`` remain no-ops.

    Terminal messages sent on *mp_queue*:

    - ``("progress", current: int, total: int)`` — progress tick
    - ``("done_file", tmp_path: str)``            — success; result pickled to file
    - ``("done", None)``                          — success fallback (pickle failed)
    - ``("cancelled", reason: str)``              — annulation demandée
    - ``("error", message: str)``                 — exception non gérée
    """
    import vote_simulation.simulation.simulation as _sim_module
    from vote_simulation.simulation.simulation import simulation_series_from_config_2

    original_tqdm = _sim_module.tqdm

    class _ProcTqdm(original_tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["file"] = io.StringIO()
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> bool | None:
            if self.disable:
                return None
            self.n += n
            if mp_stop.is_set():
                self.close()
                raise InterruptedError("Annulé")
            total = self.total or 0
            mp_queue.put(("progress", min(self.n, total), total))
            return None

    _sim_module.tqdm = _ProcTqdm  # type: ignore[assignment]
    try:
        result = simulation_series_from_config_2(config_path, reload=reload, compute_metrics=True)
        # Serialize via a temp file instead of the queue (avoids mp queue size
        # limits and multiprocessing pickle overhead for large numpy-heavy objects).
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="vote_sim_result_")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            mp_queue.put(("done_file", tmp_path))
        except Exception:
            # Fallback: results are already on disk; main process will reload lazily.
            mp_queue.put(("done", None))
    except InterruptedError as exc:
        mp_queue.put(("cancelled", str(exc)))
    except Exception as exc:
        mp_queue.put(("error", str(exc)))
    finally:
        _sim_module.tqdm = original_tqdm

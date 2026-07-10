"""Tab 2 — Data Generation.

Allows selecting generative models, voters × candidates combinations,
the number of iterations, and launching generate_data() in a separate
thread with real-time progress tracking.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

from vote_simulation.ui.toml_utils import QueueWriter

# ---------------------------------------------------------------------------
# Dictionnaire code → nom lisible (data generation)
# ---------------------------------------------------------------------------

_DICT_DATAGEN_PATH = Path(__file__).with_name("dict_datagen.json")


@st.cache_resource(show_spinner=False)
def _load_datagen_labels() -> dict[str, str]:
    """Load code → human-readable label mapping for generative models."""
    with _DICT_DATAGEN_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _format_generator_code(code: str) -> str:
    """Return 'CODE — Label' for display, falling back to the raw code."""
    labels = _load_datagen_labels()
    label = labels.get(code)
    return f"{code} — {label}" if label else code


@st.cache_resource(show_spinner=False)
def _cached_generator_codes() -> list[str]:
    """Liste des codes de modèles génératifs — mise en cache pour la session."""
    from vote_simulation.models.data_generation.from_r_registry import register_r_generators
    from vote_simulation.models.data_generation.generator_registry import list_generator_codes

    register_r_generators()
    return list_generator_codes()


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

_STOP_EVENT_KEY = "gen_stop_event"
_THREAD_KEY = "gen_thread"
_LOG_QUEUE_KEY = "gen_log_queue"
_PROGRESS_KEY = "gen_progress"  # (current, total)
_RUNNING_KEY = "gen_running"
_DONE_KEY = "gen_done"
_ERROR_KEY = "gen_error"


def _init_gen_state() -> None:
    defaults = {
        _STOP_EVENT_KEY: threading.Event(),
        _THREAD_KEY: None,
        _LOG_QUEUE_KEY: queue.Queue(),
        _PROGRESS_KEY: (0, 0),
        _RUNNING_KEY: False,
        _DONE_KEY: False,
        _ERROR_KEY: None,
        "gen_files_count": 0,
        "gen_total_size_mb": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _drain_log_queue() -> list[str]:
    """Drains the log queue and returns the messages."""
    q: queue.Queue = st.session_state[_LOG_QUEUE_KEY]
    messages = []
    try:
        while True:
            messages.append(q.get_nowait())
    except queue.Empty:
        pass
    return messages


def _run_generation(config_path: str, stop_event: threading.Event, log_q: queue.Queue) -> None:
    """Thread target: launches generate_data() with log capture."""
    # Patch tqdm to emit into the queue
    import tqdm as tqdm_module

    import vote_simulation.simulation.simulation as _sim_module
    from vote_simulation.simulation.simulation import generate_data

    original_tqdm = tqdm_module.tqdm
    original_sim_tqdm = _sim_module.tqdm  # liaison locale dans simulation.py

    class _PatchedTqdm(original_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("file", None)
            super().__init__(*args, **kwargs)
            self._last_st_update: float = 0.0

        def update(self, n: int = 1) -> bool | None:
            result = super().update(n)
            if stop_event.is_set():
                self.close()
                raise InterruptedError("Generation cancelled by user.")
            now = time.monotonic()
            if now - self._last_st_update >= 0.15:
                total = self.total or 0
                current = min(self.n, total) if total else self.n
                st.session_state[_PROGRESS_KEY] = (current, total)
                self._last_st_update = now
            return result

    tqdm_module.tqdm = _PatchedTqdm  # type: ignore[assignment]
    _sim_module.tqdm = _PatchedTqdm  # type: ignore[assignment]  # fixes the `from tqdm import tqdm` binding

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = QueueWriter(log_q)

    try:
        paths = generate_data(config_path, show_progress=True)
        st.session_state["gen_files_count"] = len(paths)
        total_bytes = sum(Path(p).stat().st_size for p in paths if Path(p).is_file())
        st.session_state["gen_total_size_mb"] = total_bytes / (1024 * 1024)
        log_q.put(f"Done: {len(paths)} files generated.")
        st.session_state[_DONE_KEY] = True
        st.session_state[_ERROR_KEY] = None
    except InterruptedError as exc:
        log_q.put(f"Cancelled: {exc}")
        st.session_state[_DONE_KEY] = True
        st.session_state[_ERROR_KEY] = "Cancelled"
    except Exception as exc:
        log_q.put(f"Error: {exc}")
        st.session_state[_DONE_KEY] = True
        st.session_state[_ERROR_KEY] = str(exc)
    finally:
        sys.stdout = old_stdout
        tqdm_module.tqdm = original_tqdm
        _sim_module.tqdm = original_sim_tqdm
        st.session_state[_RUNNING_KEY] = False


# ---------------------------------------------------------------------------
# Helpers UI
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str) -> list[int]:
    """Parses a string like '11, 101, 1001' into a list of integers."""
    result = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                val = int(part)
                if val > 0:
                    result.append(val)
            except ValueError:
                pass
    return sorted(set(result))


# ---------------------------------------------------------------------------
# Rendu principal
# ---------------------------------------------------------------------------


def render_tab_generation() -> None:
    """Renders the Data Generation tab."""
    _init_gen_state()

    st.header("Data Generation")
    st.caption(
        "Configure the generative models and the voters × candidates combinations, "
        "then launch the generation of voting profiles."
    )

    cfg: dict = st.session_state["cfg"]

    # -----------------------------------------------------------------------
    # 2.1 Generative models
    # -----------------------------------------------------------------------
    st.subheader("Generative models")

    # Deferred import to avoid a circular import at module level
    all_codes = _cached_generator_codes()

    # Sync from cfg after external load (TOML upload, reset).
    # Same logic as for rules_family_*: we write session_state before rendering
    # the multiselect so that cfg takes priority over the browser-submitted value.
    if st.session_state.pop("_cfg_gen_needs_sync", False):
        st.session_state["gen_models_select"] = [m for m in cfg.get("generative_models", []) if m in all_codes]

    selected_models = st.multiselect(
        "Selected models",
        options=all_codes,
        default=None,
        key="gen_models_select",
        format_func=_format_generator_code,
        help="Codes of generative models registered in the registry.",
    )
    cfg["generative_models"] = selected_models

    # -----------------------------------------------------------------------
    # 2.2 Simulation combinations
    # -----------------------------------------------------------------------
    st.subheader("Simulation combinations")

    col_v, col_c, col_i = st.columns([2, 2, 3])

    with col_v:
        voters_str = st.text_input(
            "Number of voters (list)",
            key="gen_voters_input",
            help="Comma-separated integers. E.g.: 11, 101, 1001",
        )
        voters = _parse_int_list(voters_str)
        cfg["voters"] = voters
        if not voters:
            st.warning("Enter at least one number of voters.")

    with col_c:
        candidates_str = st.text_input(
            "Number of candidates (list)",
            key="gen_candidates_input",
            help="Comma-separated integers. E.g.: 3, 14",
        )
        candidates = _parse_int_list(candidates_str)
        cfg["candidates"] = candidates
        if not candidates:
            st.warning("Enter at least one number of candidates.")

    with col_i:
        if "gen_iterations_slider" not in st.session_state:
            st.session_state["gen_iterations_slider"] = int(cfg.get("iterations", 1000))
        iterations = st.slider(
            "Number of iterations",
            min_value=1,
            max_value=10_000,
            step=1,
            key="gen_iterations_slider",
            help="Number of iterations per combination (model × voters × candidates).",
        )
        cfg["iterations"] = iterations

    # Real-time indicator
    n_models = len(selected_models)
    n_voters = len(cfg.get("voters", []))
    n_cands = len(cfg.get("candidates", []))
    total_profiles = n_models * n_voters * n_cands * iterations
    st.info(
        f"**Profiles to generate:** {n_models} models × {n_voters} voters × "
        f"{n_cands} candidates × {iterations} iterations = **{total_profiles:,} profiles**"
    )

    # -----------------------------------------------------------------------
    # Feedback — Progression et logs
    # -----------------------------------------------------------------------
    _render_gen_feedback()


def _render_gen_feedback() -> None:
    """Displays the progress bar and log area."""
    is_running: bool = st.session_state[_RUNNING_KEY]
    is_done: bool = st.session_state[_DONE_KEY]
    error: str | None = st.session_state[_ERROR_KEY]
    progress_tuple: tuple = st.session_state[_PROGRESS_KEY]

    # Accumulate logs in session_state
    if "gen_log_messages" not in st.session_state:
        st.session_state["gen_log_messages"] = []
    new_msgs = _drain_log_queue()
    if new_msgs:
        st.session_state["gen_log_messages"].extend(new_msgs)

    # Progress bar
    current, total = progress_tuple
    if total > 0:
        pct = min(current / total, 1.0)
        st.progress(pct, text=f"Progress: {current}/{total} profiles")
    elif is_running:
        st.progress(0.0, text="Starting…")

    # Scrollable log area
    all_logs = st.session_state.get("gen_log_messages", [])
    if all_logs:
        with st.expander("Generation logs", expanded=False):
            st.text_area(
                "Logs",
                value="\n".join(all_logs[-200:]),
                height=200,
                disabled=True,
                key="gen_log_area",
                label_visibility="collapsed",
            )

    # Final summary
    if is_done and not is_running:
        if error and error != "Cancelled":
            st.error(f"Error: {error}")
        elif error == "Cancelled":
            st.warning("Generation cancelled.")
        else:
            n_files = st.session_state.get("gen_files_count", 0)
            size_mb = st.session_state.get("gen_total_size_mb", 0.0)
            st.success(f"**Generation complete** — {n_files} files generated ({size_mb:.2f} MB total).")

    # Polling: if the thread is running, rerun the script every ~0.5 s
    if is_running:
        thread: threading.Thread | None = st.session_state[_THREAD_KEY]
        if thread is not None and thread.is_alive():
            with st.spinner("Generation in progress…"):
                time.sleep(0.3)
            st.rerun()
        else:
            # Thread finished but flag not yet reset to False (slight race condition)
            st.session_state[_RUNNING_KEY] = False
            st.rerun()

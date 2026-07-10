"""Tab 3 — Simulation (voting rules).

Allows selecting voting rules organised by family,
choosing the data source, and launching simulation_from_config()
in a separate thread with progress tracking.
"""

from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

from vote_simulation.ui.toml_utils import QueueWriter

# ---------------------------------------------------------------------------
# Dictionnaire code → nom lisible (voting rules)
# ---------------------------------------------------------------------------

_DICT_VOTINGRULES_PATH = Path(__file__).with_name("dict_votingrules.json")


@st.cache_resource(show_spinner=False)
def _load_votingrule_labels() -> dict[str, str]:
    """Load code → human-readable label mapping for voting rules."""
    with _DICT_VOTINGRULES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _format_rule_code(code: str) -> str:
    """Return 'CODE — Label' for display, falling back to the raw code."""
    labels = _load_votingrule_labels()
    label = labels.get(code)
    return f"{code} — {label}" if label else code


@st.cache_resource(show_spinner=False)
def _cached_rule_codes() -> list[str]:
    """Liste des codes de règles — mise en cache pour la session."""
    from vote_simulation.models.rules import get_all_rules_codes

    return get_all_rules_codes()


@st.cache_resource(show_spinner=False)
def _cached_family_map() -> dict[str, list[str]]:
    """Dictionnaire famille → liste de codes — mis en cache pour la session."""
    return _build_family_map(_cached_rule_codes())


# ---------------------------------------------------------------------------
# Familles de règles — mapping préfixe → famille
# ---------------------------------------------------------------------------

# Displayed order in the UI
_FAMILIES: list[tuple[str, str]] = [
    ("Plurality / IRV", "PLURALITY_IRV"),
    ("Condorcet", "CONDORCET"),
    ("Score / Borda", "SCORE_BORDA"),
    ("Judgment / Continuous score", "JUDGMENT"),
    ("Approval — threshold", "AP_THRESHOLD"),
    ("Approval — K", "AP_K"),
    ("Others", "OTHER"),
]

_FAMILY_PREFIXES: dict[str, list[str]] = {
    "PLURALITY_IRV": [
        "PLU",
        "HARE",
        "IRV",
        "IRVA",
        "IRVD",
        "ICRV",
        "SIRV",
        "EXHB",
        "RPAR",
        "BUCK",
        "IBUC",
        "L4VD",
        "SPCY",
        "CAIR",
    ],
    "CONDORCET": [
        "COND",
        "COPE",
        "SCHU",
        "MMAX",
        "BLAC",
        "KEME",
        "SLAT",
        "KIMR",
        "WOOD",
        "YOUN",
        "TIDE",
        "DODG",
        "CSUM",
        "CVIR",
    ],
    "SCORE_BORDA": ["BORD", "COOM", "NANS", "PV-"],
    "JUDGMENT": ["MJ", "RV", "STAR", "VETO"],
    "AP_THRESHOLD": ["AP_T"],
    "AP_K": ["AP_K"],
}


def _classify_rule(code: str) -> str:
    """Returns the family of a rule from its code."""
    upper = code.upper()
    for family_id, prefixes in _FAMILY_PREFIXES.items():
        for pfx in prefixes:
            if upper.startswith(pfx):
                return family_id
    return "OTHER"


def _build_family_map(all_codes: list[str]) -> dict[str, list[str]]:
    """Builds a dictionary family → list of codes."""
    family_map: dict[str, list[str]] = {fid: [] for _, fid in _FAMILIES}
    for code in all_codes:
        fid = _classify_rule(code)
        if fid in family_map:
            family_map[fid].append(code)
        else:
            family_map["OTHER"].append(code)
    return family_map


# ---------------------------------------------------------------------------
# État du thread simulation
# ---------------------------------------------------------------------------

_SIM_STOP_KEY = "sim_stop_event"
_SIM_THREAD_KEY = "sim_thread"
_SIM_LOG_Q_KEY = "sim_log_queue"
_SIM_PROGRESS_KEY = "sim_progress"
_SIM_RUNNING_KEY = "sim_running"
_SIM_DONE_KEY = "sim_done"
_SIM_ERROR_KEY = "sim_error"


def _init_sim_state() -> None:
    defaults = {
        _SIM_STOP_KEY: threading.Event(),
        _SIM_THREAD_KEY: None,
        _SIM_LOG_Q_KEY: queue.Queue(),
        _SIM_PROGRESS_KEY: (0, 0),
        _SIM_RUNNING_KEY: False,
        _SIM_DONE_KEY: False,
        _SIM_ERROR_KEY: None,
        "sim_log_messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _run_simulation(config_path: str, stop_event: threading.Event, log_q: queue.Queue, reload: bool = False) -> None:
    """Thread target: launches simulation_series_from_config() with log capture."""
    import tqdm as tqdm_module

    import vote_simulation.simulation.simulation as _sim_module
    from vote_simulation.simulation.simulation import simulation_series_from_config

    original_tqdm = tqdm_module.tqdm
    original_sim_tqdm = _sim_module.tqdm  # liaison locale dans simulation.py

    class _PatchedTqdm(original_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            import io

            kwargs["file"] = io.StringIO()
            # Seules les instances non désactivées (= tqdm de séries) mettent
            # à jour la barre de progression Streamlit.
            self._track = not kwargs.get("disable", False)
            super().__init__(*args, **kwargs)
            self._last_st_update: float = 0.0

        def update(self, n: float | int = 1) -> bool | None:
            self.n += n
            if stop_event.is_set():
                self.close()
                raise InterruptedError("Simulation cancelled by user.")
            if self._track:
                now = time.monotonic()
                if now - self._last_st_update >= 0.15:
                    total = self.total or 0
                    current = min(self.n, total) if total else self.n
                    st.session_state[_SIM_PROGRESS_KEY] = (current, total)
                    self._last_st_update = now
            return None

    tqdm_module.tqdm = _PatchedTqdm  # type: ignore[assignment]
    _sim_module.tqdm = _PatchedTqdm  # type: ignore[assignment]  # fixes the `from tqdm import tqdm` binding
    old_stdout = sys.stdout
    sys.stdout = QueueWriter(log_q)

    try:
        simulation_series_from_config(config_path, reload=reload)

        log_q.put("Simulation done.")
        st.session_state[_SIM_DONE_KEY] = True
        st.session_state[_SIM_ERROR_KEY] = None
    except InterruptedError as exc:
        log_q.put(f"Cancelled: {exc}")
        st.session_state[_SIM_DONE_KEY] = True
        st.session_state[_SIM_ERROR_KEY] = "Cancelled"
    except Exception as exc:
        log_q.put(f"Error: {exc}")
        st.session_state[_SIM_DONE_KEY] = True
        st.session_state[_SIM_ERROR_KEY] = str(exc)
    finally:
        sys.stdout = old_stdout
        tqdm_module.tqdm = original_tqdm
        _sim_module.tqdm = original_sim_tqdm
        st.session_state[_SIM_RUNNING_KEY] = False


def _drain_log_queue() -> list[str]:
    q: queue.Queue = st.session_state[_SIM_LOG_Q_KEY]
    messages = []
    try:
        while True:
            messages.append(q.get_nowait())
    except queue.Empty:
        pass
    return messages


def _invalidate_results_cache(base_path: str) -> None:
    """Clear Results-tab caches tied to a base path before launching a new run."""
    # Clear in-memory objects that can keep stale rule lists between runs.
    for key in list(st.session_state.keys()):
        if not isinstance(key, str):
            continue
        if key in {"gf_m_applied", "gf_v_applied", "gf_c_applied"}:
            del st.session_state[key]
            continue
        if key.startswith(
            (
                "_res_total_",
                "_scan_struct_",
                "_filtered_",
                "_series_",
                "_total_one_",
                "_plt_",
                "_avail_rules_",
                "_g_dist_df_",
                "_g_met_df_",
            )
        ):
            if base_path in key or key.startswith("_plt_"):
                del st.session_state[key]

    # Remove worker in-memory result fallback from previous full run.
    st.session_state.pop("sim_total_result", None)

    # Clear persisted aggregated total cache so Results tab rebuilds from series files.
    total_dir = Path(base_path) / "results" / "_total_result"
    try:
        shutil.rmtree(total_dir)
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Rendu principal
# ---------------------------------------------------------------------------


def render_tab_simulation() -> None:
    """Renders the Simulation tab."""
    _init_sim_state()

    st.header("Simulation — Voting rules")
    st.caption("Select the rules to apply then launch the simulation from the global bar.")

    cfg: dict = st.session_state["cfg"]

    # -----------------------------------------------------------------------
    # 3.1 Dynamic loading of rules
    # -----------------------------------------------------------------------
    all_codes = _cached_rule_codes()
    family_map = _cached_family_map()

    # Explicit sync from cfg after an external load (TOML upload, reset).
    # Streamlit restores browser values (stale, often []) instead of using
    # default= when session_state keys have been cleared.
    # We work around this by writing the correct values directly into
    # session_state BEFORE rendering the multiselects.
    if st.session_state.pop("_cfg_rules_needs_sync", False):
        loaded_rules = cfg.get("rule_codes", [])
        for _, family_id in _FAMILIES:
            codes_in_family = family_map.get(family_id, [])
            st.session_state[f"rules_family_{family_id}"] = [c for c in codes_in_family if c in loaded_rules]

    # Quick-select buttons
    col_all, col_none = st.columns(2)
    with col_all:
        if st.button("Select all", key="sim_select_all"):
            cfg["rule_codes"] = list(all_codes)
            for _, family_id in _FAMILIES:
                st.session_state[f"rules_family_{family_id}"] = family_map.get(family_id, [])
            st.rerun()
    with col_none:
        if st.button("Deselect all", key="sim_deselect_all"):
            cfg["rule_codes"] = []
            for _, family_id in _FAMILIES:
                st.session_state[f"rules_family_{family_id}"] = []
            st.rerun()

    st.subheader("Rule selection by family")
    new_selection_family_order: list[str] = []  # accumulation in family order (for new-rule detection)

    for family_label, family_id in _FAMILIES:
        codes_in_family = family_map.get(family_id, [])
        if not codes_in_family:
            continue

        with st.expander(
            f"{family_label} ({len(codes_in_family)} rules)", expanded=(family_id in ("PLURALITY_IRV", "SCORE_BORDA"))
        ):
            # Select/deselect buttons for this family
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button(f"All — {family_label}", key=f"sel_all_{family_id}"):
                    existing_ordered = cfg.get("rule_codes", [])
                    existing_set = set(existing_ordered)
                    cfg["rule_codes"] = existing_ordered + [c for c in codes_in_family if c not in existing_set]
                    st.session_state[f"rules_family_{family_id}"] = codes_in_family
                    st.rerun()
            with fc2:
                if st.button(f"None — {family_label}", key=f"sel_none_{family_id}"):
                    to_remove = set(codes_in_family)
                    cfg["rule_codes"] = [r for r in cfg.get("rule_codes", []) if r not in to_remove]
                    st.session_state[f"rules_family_{family_id}"] = []
                    st.rerun()

            chosen = st.multiselect(
                "Rules",
                options=codes_in_family,
                default=None,
                key=f"rules_family_{family_id}",
                format_func=_format_rule_code,
                label_visibility="collapsed",
            )
            new_selection_family_order.extend(chosen)

    # ────────────────────────────────────────────────────────────────────────
    # Update cfg preserving custom order:
    #   - already present rules keep their position,
    #   - newly added rules are appended at the end (family order),
    #   - deselected rules are removed.
    new_selected_set = set(new_selection_family_order)
    old_ordered = cfg.get("rule_codes", [])
    preserved_ordered: list[str] = [r for r in old_ordered if r in new_selected_set]
    preserved_set = set(preserved_ordered)
    for r in new_selection_family_order:
        if r not in preserved_set:
            preserved_ordered.append(r)
            preserved_set.add(r)
    cfg["rule_codes"] = preserved_ordered
    selected_rules = preserved_ordered  # alias for the rest of the tab

    # Summary
    st.info(f"**{len(selected_rules)} rule(s) selected** out of {len(all_codes)} available.")

    # -----------------------------------------------------------------------
    # Feedback
    # -----------------------------------------------------------------------
    if st.session_state.get(_SIM_RUNNING_KEY) or st.session_state.get(_SIM_DONE_KEY):
        _render_sim_feedback()


@st.fragment(run_every=1)
def _render_sim_feedback() -> None:
    """Streamlit fragment (run_every=1 s) — only this zone re-executes during
    the simulation, without touching the Data/Configuration tabs or their widgets.
    A single full rerun is triggered at the end to refresh the global state.
    """
    is_running: bool = st.session_state.get(_SIM_RUNNING_KEY, False)
    is_done: bool = st.session_state.get(_SIM_DONE_KEY, False)
    error: str | None = st.session_state.get(_SIM_ERROR_KEY)
    progress_tuple: tuple = st.session_state.get(_SIM_PROGRESS_KEY, (0, 0))

    # Drain log queue into persistent list
    new_msgs = _drain_log_queue()
    if new_msgs:
        st.session_state["sim_log_messages"].extend(new_msgs)

    # Detect thread finished but flag not yet updated
    if is_running:
        thread: threading.Thread | None = st.session_state.get(_SIM_THREAD_KEY)
        if thread is not None and not thread.is_alive():
            st.session_state[_SIM_RUNNING_KEY] = False
            is_running = False

    current, total = progress_tuple
    if total > 0:
        pct = min(current / total, 1.0)
        st.progress(pct, text=f"Progress: {current}/{total} profiles simulated")
    elif is_running:
        st.progress(0.0, text="Initialising simulation…")

    all_logs = st.session_state.get("sim_log_messages", [])
    if all_logs:
        with st.expander("Simulation logs", expanded=False):
            st.text_area(
                "Logs",
                value="\n".join(all_logs[-300:]),
                height=250,
                disabled=True,
                key="sim_log_area",
                label_visibility="collapsed",
            )

    if is_done and not is_running:
        cfg: dict = st.session_state["cfg"]
        n_rules = len(cfg.get("rule_codes", []))
        if error and error != "Cancelled":
            st.error(f"Error: {error}")
        elif error == "Cancelled":
            st.warning("Simulation cancelled.")
        else:
            total_profiles = (
                len(cfg.get("generative_models", []))
                * len(cfg.get("voters", []))
                * len(cfg.get("candidates", []))
                * cfg.get("iterations", 0)
            )
            st.success(
                f"**Simulation complete** — {n_rules} rules × {total_profiles:,} profiles processed.\n\n"
                f"Results available in the Results tab."
            )

    # Single full rerun at the end — updates global status and buttons.
    if is_done and not is_running:
        if not st.session_state.get("_sim_final_rerun_done", False):
            st.session_state["_sim_final_rerun_done"] = True
            st.session_state["_cfg_post_run_restore"] = True  # restores gen_models
            st.rerun()

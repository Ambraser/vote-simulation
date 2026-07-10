"""Main Streamlit application — vote_simulation UI.

Structure:
    Global bar (status, active TOML, Full run)
    └── Tab 1: Configuration
    └── Tab 2: Data (generation)
    └── Tab 3: Simulation (rules)
    └── Tab 4: Results

Launch:
    streamlit run src/vote_simulation/ui/app.py
    # or via the entry-point:
    vote-sim-ui
"""

from __future__ import annotations

import copy
import hashlib
import multiprocessing as mp
import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from vote_simulation.ui._worker import simulation_worker
from vote_simulation.ui.tab_config import render_tab_config
from vote_simulation.ui.tab_generation import _parse_int_list, render_tab_generation
from vote_simulation.ui.tab_results import render_tab_results
from vote_simulation.ui.tab_simulation import _FAMILIES, render_tab_simulation
from vote_simulation.ui.toml_utils import DEFAULT_STATE, state_to_toml, write_temp_toml

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="vote_simulation UI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# session_state initialisation
# ---------------------------------------------------------------------------


def _init_session() -> None:
    if "cfg" not in st.session_state:
        # Start with no default configuration — the user explicitly loads a TOML.
        st.session_state["cfg"] = copy.deepcopy(DEFAULT_STATE)
        st.session_state["toml_active_path"] = None  # No config loaded
        st.session_state["cfg_base_dir"] = None

    # Global status
    if "global_status" not in st.session_state:
        st.session_state["global_status"] = "Ready"

    # Full run state
    if "full_run_running" not in st.session_state:
        st.session_state["full_run_running"] = False
    if "full_run_done" not in st.session_state:
        st.session_state["full_run_done"] = False
    if "full_run_error" not in st.session_state:
        st.session_state["full_run_error"] = None
    if "full_run_log_q" not in st.session_state:
        st.session_state["full_run_log_q"] = queue.Queue()
    if "full_run_progress" not in st.session_state:
        st.session_state["full_run_progress"] = (0, 0)
    if "full_run_thread" not in st.session_state:
        st.session_state["full_run_thread"] = None
    if "full_run_stop" not in st.session_state:
        st.session_state["full_run_stop"] = threading.Event()
    if "full_run_logs" not in st.session_state:
        st.session_state["full_run_logs"] = []
    if "full_run_start_time" not in st.session_state:
        st.session_state["full_run_start_time"] = None
    if "full_run_end_time" not in st.session_state:
        st.session_state["full_run_end_time"] = None
    # Anti-loop lock for the fragment's final rerun
    if "_full_run_final_rerun_done" not in st.session_state:
        st.session_state["_full_run_final_rerun_done"] = False


def _project_cfg_to_widgets(cfg: dict) -> None:
    """Explicitly projects cfg into widget keys sensitive to stale values."""
    st.session_state["cfg_output_base_path"] = cfg.get("output_base_path") or ""
    seed_val = cfg.get("seed")
    st.session_state["cfg_seed"] = int(seed_val) if seed_val is not None else None
    st.session_state["gen_voters_input"] = ", ".join(str(v) for v in cfg.get("voters", []))
    st.session_state["gen_candidates_input"] = ", ".join(str(c) for c in cfg.get("candidates", []))
    st.session_state["gen_iterations_slider"] = int(cfg.get("iterations", 1000))

    # These widgets are rebuilt from cfg in the Simulation tab.
    for key in ("sim_data_source", "sim_input_folder"):
        st.session_state.pop(key, None)

    # Purge dynamic keys to force value=/cfg on next render.
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (key.startswith("rules_family_") or key.startswith("gp_")):
            del st.session_state[key]

    # Multiselects synchronise at the start of their tab via these flags.
    st.session_state["_cfg_gen_needs_sync"] = True
    st.session_state["_cfg_rules_needs_sync"] = True


# ---------------------------------------------------------------------------
# Full run — Generation + Simulation chained
# ---------------------------------------------------------------------------

# On POSIX (Linux / macOS) we use "fork": the child inherits the parent's
# memory, which is fast and avoids re-importing the whole application.
# On Windows only "spawn" is available: Python starts a fresh interpreter
# and imports the worker function from its module path.  simulation_worker
# lives in vote_simulation.ui._worker (a plain importable module, not inside
# the Streamlit script) so it is safely picklable under spawn.
_MP_CONTEXT: str = "fork" if sys.platform != "win32" else "spawn"


def _run_full(
    config_path: str,
    stop_event: threading.Event,
    log_q: queue.Queue,
    reload: bool = False,
) -> None:
    """Monitor thread: launches the simulation in a forked subprocess and relays its progress.

    The subprocess has its own GIL (no contention with Streamlit's Tornado / asyncio
    event loop), allowing it to reach notebook-level execution performance —
    without any modification to the simulation code itself.
    """
    _ctx = mp.get_context(_MP_CONTEXT)
    mp_queue: mp.Queue = _ctx.Queue()
    mp_stop: object = _ctx.Event()
    proc = _ctx.Process(  # type: ignore[attr-defined]
        target=simulation_worker,
        args=(config_path, reload, mp_queue, mp_stop),
        daemon=True,
    )
    proc.start()
    last_st_update: float = 0.0

    try:
        while True:
            # Cancellation requested from the UI
            if stop_event.is_set():
                mp_stop.set()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                raise InterruptedError("Full run cancelled.")

            # Abnormal termination detection (subprocess crash without message)
            if not proc.is_alive() and mp_queue.empty():
                ec = proc.exitcode
                raise RuntimeError(f"The simulation process terminated unexpectedly (exit code {ec}).")

            try:
                msg = mp_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            msg_type = msg[0]
            if msg_type == "progress":
                _, current, total = msg
                now = time.monotonic()
                if now - last_st_update >= 0.15:
                    st.session_state["full_run_progress"] = (current, total)
                    last_st_update = now
            elif msg_type == "done_file":
                import os
                import pickle

                tmp_path = msg[1]
                try:
                    with open(tmp_path, "rb") as fh:
                        result = pickle.load(fh)
                    st.session_state["sim_total_result"] = result
                except Exception:
                    pass  # Results remain on disk; Results tab will load lazily.
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                _cur_total = st.session_state.get("full_run_progress", (0, 1))[1]
                st.session_state["full_run_progress"] = (_cur_total, _cur_total)
                log_q.put("Full run completed.")
                st.session_state["full_run_done"] = True
                st.session_state["full_run_error"] = None
                st.session_state["global_status"] = "Done"
                proc.join()
                return
            elif msg_type == "done":
                # Fallback: subprocess could not serialize the result.
                _cur_total = st.session_state.get("full_run_progress", (0, 1))[1]
                st.session_state["full_run_progress"] = (_cur_total, _cur_total)
                log_q.put("Full run completed.")
                st.session_state["full_run_done"] = True
                st.session_state["full_run_error"] = None
                st.session_state["global_status"] = "Done"
                proc.join()
                return
            elif msg_type == "cancelled":
                raise InterruptedError(msg[1])
            elif msg_type == "error":
                raise RuntimeError(msg[1])

    except InterruptedError as exc:
        log_q.put(f"Cancelled: {exc}")
        st.session_state["full_run_done"] = True
        st.session_state["full_run_error"] = "Cancelled"
        st.session_state["global_status"] = "Cancelled"
    except Exception as exc:
        log_q.put(f"Error: {exc}")
        st.session_state["full_run_done"] = True
        st.session_state["full_run_error"] = str(exc)
        st.session_state["global_status"] = "Error"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()
        st.session_state["full_run_running"] = False
        st.session_state["full_run_end_time"] = time.monotonic()


# ---------------------------------------------------------------------------
# Global bar
# ---------------------------------------------------------------------------


def _render_global_bar() -> None:
    """Renders the persistent global status bar at the top of the page."""
    cfg: dict = st.session_state["cfg"]
    status: str = st.session_state.get("global_status", "Prêt")
    toml_path: str = st.session_state.get("toml_active_path") or "Aucune config chargée"
    is_running: bool = (
        st.session_state.get("full_run_running", False)
        or st.session_state.get("gen_running", False)
        or st.session_state.get("sim_running", False)
    )

    # Compute global status from sub-states
    if st.session_state.get("gen_running") or st.session_state.get("sim_running"):
        if st.session_state.get("gen_running"):
            status = "Generating…"
        else:
            status = "Simulating…"
    elif st.session_state.get("full_run_running"):
        status = "Full run in progress…"

    st.session_state["global_status"] = status

    col_status, col_toml, col_reload, col_run = st.columns([2, 3, 2, 2])

    with col_status:
        color = (
            "#28a745"
            if "Ready" in status or "Done" in status
            else ("#dc3545" if "Error" in status or "Cancelled" in status else "#fd7e14")
        )
        st.markdown(
            f'<div style="padding:8px 12px; border-radius:6px; background:{color}20; '
            f'border-left:4px solid {color}; font-weight:bold; color:{color};">'
            f"● {status}</div>",
            unsafe_allow_html=True,
        )

    with col_toml:
        st.markdown(
            f'<div style="padding:8px 12px; border-radius:6px; background:#f0f2f6; '
            f'font-family:monospace; font-size:0.85em; color:#444;">Active TOML: {toml_path}</div>',
            unsafe_allow_html=True,
        )

    with col_reload:
        st.checkbox(
            "Force recompute",
            key="global_full_reload",
            help="Recomputes even if simulation results already exist (reload).",
            disabled=is_running,
        )

    with col_run:
        full_run_disabled = is_running or not cfg.get("generative_models") or not cfg.get("rule_codes")
        if st.button(
            "Full run",
            disabled=full_run_disabled,
            type="primary",
            key="global_full_run",
            help="Chains Generation → Simulation with the current configuration.",
            use_container_width=True,
        ):
            # Defensive validation — the button is normally disabled if these fields
            # are empty, but session_state corruption could clear them between renders.
            # We check before starting the thread to avoid an unreadable error
            # from load_simulation_config().
            if not cfg.get("rule_codes"):
                st.error("The configuration contains no voting rules — please reload the TOML.")
                st.stop()
            if not cfg.get("generative_models"):
                st.error("The configuration contains no generative models — please reload the TOML.")
                st.stop()

            # Invalidate result caches from the previous session.
            # The Results tab will use sim_total_result (built in memory)
            # instead of re-reading Parquet files from disk.
            st.session_state.pop("sim_total_result", None)
            for _k in [k for k in st.session_state if k.startswith("_res_total_")]:
                del st.session_state[_k]
            for _k in [k for k in st.session_state if k.startswith("_scan_struct_")]:
                del st.session_state[_k]

            # Save the source-of-truth state (cfg) for restoration after the run.
            st.session_state["_cfg_saved_snapshot"] = copy.deepcopy(cfg)
            st.session_state["_cfg_saved_gen_models"] = list(cfg.get("generative_models", []))
            st.session_state["_cfg_saved_rule_codes"] = list(cfg.get("rule_codes", []))

            stop_event = threading.Event()
            log_q: queue.Queue = queue.Queue()
            st.session_state["full_run_stop"] = stop_event
            st.session_state["full_run_log_q"] = log_q
            st.session_state["full_run_running"] = True
            st.session_state["full_run_done"] = False
            st.session_state["full_run_error"] = None
            st.session_state["full_run_logs"] = []
            st.session_state["_full_run_final_rerun_done"] = False  # permet le rerun final
            # Pre-compute the number of combinations (model × voters × candidates)
            # to display the progress bar before the thread's first tick.
            _pre_total = max(
                len(cfg.get("generative_models", [])) * len(cfg.get("voters", [])) * len(cfg.get("candidates", [])),
                1,
            )
            st.session_state["full_run_progress"] = (0, _pre_total)
            st.session_state["global_status"] = "Full run in progress…"

            # Reuse the temp TOML file kept up to date by _refresh_active_toml().
            # If absent (first run without prior modification), create one.
            tmp_path = st.session_state.get("_active_tmp_toml_path") or write_temp_toml(
                cfg, base_dir=st.session_state.get("cfg_base_dir")
            )

            _reload = st.session_state.get("global_full_reload", False)
            t = threading.Thread(
                target=_run_full,
                args=(tmp_path, stop_event, log_q, _reload),
                daemon=True,
            )
            add_script_run_ctx(t, get_script_run_ctx())
            t.start()
            st.session_state["full_run_thread"] = t
            st.session_state["full_run_start_time"] = time.monotonic()
            st.session_state["full_run_end_time"] = None
            st.rerun()

    # Full run feedback
    if st.session_state.get("full_run_running") or st.session_state.get("full_run_done"):
        _render_full_run_feedback()


def _fmt_duration(seconds: float) -> str:
    """Formats a duration in seconds as a human-readable string (Xm Ys or Xs)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


@st.fragment(run_every=1)
def _render_full_run_feedback() -> None:
    """Inline feedback for the Full run.

    Streamlit fragment (run_every=1 s): only this section re-executes every
    second, without re-running the 4 tabs.  This nearly fully releases the GIL
    for the simulation thread and eliminates the CPU contention responsible for
    the ×10 slowdown observed with the old time.sleep+rerun approach.

    A single full rerun is triggered at the end of the run to update the global
    status, re-enable the button, etc.
    """
    is_running: bool = st.session_state.get("full_run_running", False)
    is_done: bool = st.session_state.get("full_run_done", False)
    error: str | None = st.session_state.get("full_run_error")
    progress_tuple: tuple = st.session_state.get("full_run_progress", (0, 0))

    log_q: queue.Queue = st.session_state["full_run_log_q"]
    new_msgs = []
    try:
        while True:
            new_msgs.append(log_q.get_nowait())
    except queue.Empty:
        pass
    if new_msgs:
        st.session_state["full_run_logs"].extend(new_msgs)

    # ── Progress bar + timer ──────────────────────────────────────────────
    current, total = progress_tuple
    start_t: float | None = st.session_state.get("full_run_start_time")
    end_t: float | None = st.session_state.get("full_run_end_time")

    if total > 0:
        frac = min(current / total, 1.0)
        # Build the label with timer
        if is_running and start_t is not None:
            elapsed = time.monotonic() - start_t
            elapsed_str = _fmt_duration(elapsed)
            if current > 0:
                eta = elapsed / current * (total - current)
                label = f"Full run: {current}/{total} — ⏱ {elapsed_str} elapsed · ETA ~{_fmt_duration(eta)}"
            else:
                label = f"Full run: {current}/{total} — ⏱ {elapsed_str} elapsed"
        elif is_done and start_t is not None and end_t is not None:
            elapsed_str = _fmt_duration(end_t - start_t)
            label = f"Full run: {current}/{total} — completed in {elapsed_str}"
        else:
            label = f"Full run: {current}/{total}"
        st.progress(frac, text=label)
    elif is_running:
        # Indeterminate bar immediately after click (no tqdm tick yet)
        elapsed_str = _fmt_duration(time.monotonic() - start_t) if start_t else ""
        st.progress(0.0, text=f"Full run in progress… {elapsed_str}")

    if st.session_state["full_run_logs"]:
        with st.expander("Full run logs", expanded=False):
            st.text_area(
                "Logs",
                value="\n".join(st.session_state["full_run_logs"][-200:]),
                height=150,
                disabled=True,
                label_visibility="collapsed",
                key="full_run_log_area",
            )

    if is_done and not is_running:
        if error and error != "Cancelled":
            st.error(f"Full run — Error: {error}")
        elif error == "Cancelled":
            st.warning("Full run cancelled.")
        else:
            st.success("Full run completed — see the Results tab.")

    if is_running:
        t: threading.Thread | None = st.session_state.get("full_run_thread")
        if t is not None and not t.is_alive():
            # Thread finished but flag not yet reset to False — defensive fix.
            st.session_state["full_run_running"] = False

    # ── Single full rerun at the end of the run ───────────────────────────────
    # Triggered once when is_done becomes True so the rest of the page
    # (global status, button, Results tab) is refreshed.
    if is_done and not is_running:
        if not st.session_state.get("_full_run_final_rerun_done", False):
            st.session_state["_full_run_final_rerun_done"] = True
            st.session_state["_cfg_post_run_restore"] = True  # restores gen_models
            st.rerun()  # full rerun (outside fragment) — once only


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _sync_cfg_from_widgets() -> None:
    """Pre-applies the current widget values to cfg before rendering tab_config.

    Streamlit executes tabs in declaration order.  tab_config (tab 1) computes
    the TOML preview BEFORE tab_generation and tab_simulation have updated cfg
    via their own widgets.  This function reads values already stored in
    session_state (available from the start of the script on each rerun) and
    applies them to cfg, so the preview always reflects the current widget state.
    """
    cfg: dict | None = st.session_state.get("cfg")
    if cfg is None:
        return

    # ── Data tab ─────────────────────────────────────────────────────────
    # output_base_path (tab_config widget, also applied here for consistency)
    if "cfg_output_base_path" in st.session_state:
        cfg["output_base_path"] = st.session_state["cfg_output_base_path"]

    # Seed (number_input → int or None)
    if "cfg_seed" in st.session_state:
        val = st.session_state["cfg_seed"]
        cfg["seed"] = int(val) if val is not None else None

    # Generative models
    # If a resync from cfg is requested, do not overwrite cfg
    # with a potentially stale browser state on this rerun.
    if not st.session_state.get("_cfg_gen_needs_sync", False) and "gen_models_select" in st.session_state:
        cfg["generative_models"] = list(st.session_state["gen_models_select"])

    # Voters / Candidates (text_input → parsing)
    if "gen_voters_input" in st.session_state:
        cfg["voters"] = _parse_int_list(st.session_state["gen_voters_input"])

    if "gen_candidates_input" in st.session_state:
        cfg["candidates"] = _parse_int_list(st.session_state["gen_candidates_input"])

    # Iterations (slider)
    if "gen_iterations_slider" in st.session_state:
        val = st.session_state["gen_iterations_slider"]
        if isinstance(val, (int, float)) and int(val) > 0:
            cfg["iterations"] = int(val)

    # ── Simulation tab — Voting rules ────────────────────────────────────
    # Aggregate all families present in session_state.
    # We only touch rule_codes if at least one family key exists,
    # to avoid overwriting a config loaded via TOML before the first render.
    rule_keys = [f"rules_family_{fid}" for _, fid in _FAMILIES]
    if (not st.session_state.get("_cfg_rules_needs_sync", False)) and any(k in st.session_state for k in rule_keys):
        # Build new selection in family order (for detecting newly added rules)
        new_family_order: list[str] = []
        for key in rule_keys:
            new_family_order.extend(st.session_state.get(key, []))
        # Preserve custom order: keep existing order, append newly added rules at end
        new_set = set(new_family_order)
        old_ordered = cfg.get("rule_codes", [])
        preserved: list[str] = [r for r in old_ordered if r in new_set]
        preserved_set = set(preserved)
        for r in new_family_order:
            if r not in preserved_set:
                preserved.append(r)
                preserved_set.add(r)
        cfg["rule_codes"] = preserved


def _refresh_active_toml() -> None:
    """Keeps a temp TOML file up to date reflecting the current cfg state.

    Called on every rerun, after _sync_cfg_from_widgets().  Computes a hash of
    the serialised TOML content and only rewrites to disk if cfg has changed
    since the last write (avoids unnecessary I/O).

    Updates:
    - ``session_state["_active_tmp_toml_path"]``: absolute path of the temp file
    - ``session_state["toml_active_path"]``: label displayed in the global bar
      (e.g. "simulation.toml", "simulation.toml (modified)", "UI (unsaved)")
    """
    cfg: dict | None = st.session_state.get("cfg")
    if cfg is None:
        return

    # Hash du contenu TOML sérialisé — seule la sérialisation finale compte.
    toml_str = state_to_toml(cfg)
    cfg_hash = hashlib.md5(toml_str.encode()).hexdigest()  # noqa: S324 — non-cryptographic

    if st.session_state.get("_active_tmp_cfg_hash") == cfg_hash:
        # Nothing changed: the existing temp file is still valid.
        return

    # Write the new temp file (resolving output_base_path from cfg_base_dir).
    tmp_path = write_temp_toml(cfg, base_dir=st.session_state.get("cfg_base_dir"))

    # Clean up the old temp file to avoid accumulation in /tmp/.
    old_path: str | None = st.session_state.get("_active_tmp_toml_path")
    if old_path and old_path != tmp_path:
        try:
            Path(old_path).unlink(missing_ok=True)
        except OSError:
            pass

    st.session_state["_active_tmp_toml_path"] = tmp_path
    st.session_state["_active_tmp_cfg_hash"] = cfg_hash

    # ── Label displayed in the global bar ────────────────────────────────
    original_name: str | None = st.session_state.get("_original_toml_name")
    original_hash: str | None = st.session_state.get("_original_cfg_hash")

    if original_name:
        # A file was loaded: indicate if the config has been modified since.
        if cfg_hash == original_hash:
            label: str | None = original_name
        else:
            label = f"{original_name} (modified)"
    else:
        # No file loaded: config built entirely from the UI.
        has_content = bool(cfg.get("generative_models") or cfg.get("rule_codes"))
        label = "UI (unsaved)" if has_content else None

    st.session_state["toml_active_path"] = label


def main() -> None:
    _init_session()

    # ────────────────────────────────────────────────────────────────────────
    # Widget state restoration after a full run / simulation.
    #
    # Problem: when the fragment (@st.fragment run_every=1) triggers st.rerun(),
    # Streamlit reconciles the browser's WebSocket state (which was not updated
    # during fragment reruns) with the server session_state.  The browser's
    # "cold" value for the gen_models_select multiselect can overwrite the server
    # value, dropping models added between runs.
    # Solution: save gen_models_select before each run and restore it here,
    # before _sync_cfg_from_widgets, on the final rerun.
    # ────────────────────────────────────────────────────────────────────────
    if st.session_state.pop("_cfg_post_run_restore", False):
        # Restore the cfg state captured at run launch: guards against any
        # reset / overwrite that occurred during fragment reruns.
        saved_snapshot = st.session_state.pop("_cfg_saved_snapshot", None)
        if saved_snapshot is not None:
            st.session_state["cfg"] = copy.deepcopy(saved_snapshot)

        _cfg_now = st.session_state.get("cfg")
        st.session_state.pop("_cfg_saved_gen_models", None)
        st.session_state.pop("_cfg_saved_rule_codes", None)

        if _cfg_now is not None:
            _project_cfg_to_widgets(_cfg_now)
            # On this precise rerun, block widget→cfg sync to avoid a stale
            # browser value overwriting the current cfg.
            st.session_state["_skip_widget_sync_once"] = True

    # Pre-sync cfg from widgets so that the tab_config TOML preview
    # immediately reflects every change in the Data and Simulation tabs.
    if st.session_state.pop("_skip_widget_sync_once", False):
        pass
    else:
        _sync_cfg_from_widgets()

    # Keep the temp TOML file and the "Active TOML" label up to date.
    _refresh_active_toml()

    # Header
    st.title("vote_simulation")
    st.markdown("---")

    # Global bar
    _render_global_bar()
    st.markdown("---")

    # 4 main tabs
    tab_cfg, tab_gen, tab_sim, tab_res = st.tabs(
        [
            "Configuration",
            "Data",
            "Simulation",
            "Results",
        ]
    )

    with tab_cfg:
        render_tab_config()

    with tab_gen:
        render_tab_generation()

    with tab_sim:
        render_tab_simulation()

    with tab_res:
        render_tab_results()


if __name__ == "__main__":
    main()

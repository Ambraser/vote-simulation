"""Tab 1 — TOML Configuration.

Allows loading, editing and exporting the simulation configuration
without touching the code. Any change is propagated to st.session_state.
"""

from __future__ import annotations

import copy
import hashlib

import streamlit as st

from vote_simulation.ui.toml_utils import (
    DEFAULT_STATE,
    state_to_toml,
    toml_bytes_to_state,
)


def _cfg_hash(state: dict) -> str:
    """Returns an MD5 hash of the serialised TOML from *state*."""
    return hashlib.md5(state_to_toml(state).encode()).hexdigest()  # noqa: S324


def _clear_cfg_widget_keys() -> None:
    """Synchronises all cfg-driven widgets with the values of the current cfg.

    Must be called before st.rerun() after any external cfg load
    (TOML upload, reset).

    IMPORTANT: clearing session_state keys is not enough — Streamlit restores
    the value submitted by the browser (stale) instead of using value=/default=.
    We therefore write each value explicitly into session_state from cfg, which
    takes absolute priority over the browser value.
    """
    cfg = st.session_state.get("cfg", {})

    # ---- Simple widgets (text_input, number_input, slider) ----
    # Write the cfg value directly into session_state → absolute priority.
    st.session_state["cfg_output_base_path"] = cfg.get("output_base_path") or ""
    seed_val = cfg.get("seed")
    st.session_state["cfg_seed"] = int(seed_val) if seed_val is not None else None
    st.session_state["gen_voters_input"] = ", ".join(str(v) for v in cfg.get("voters", []))
    st.session_state["gen_candidates_input"] = ", ".join(str(c) for c in cfg.get("candidates", []))
    st.session_state["gen_iterations_slider"] = int(cfg.get("iterations", 1000))

    # ──── Optional widgets (radio/text data source) — simple deletion is sufficient
    for key in ("sim_data_source", "sim_input_folder"):
        st.session_state.pop(key, None)

    # ---- Multiselects (depend on external registries loaded in the tabs) ----
    # Use flags consumed by each tab at the start of its render.
    st.session_state["_cfg_gen_needs_sync"] = True  # → gen_models_select in render_tab_generation
    st.session_state["_cfg_rules_needs_sync"] = True  # → rules_family_* in render_tab_simulation

    # Clear old rules_family_* keys to avoid type conflicts
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("rules_family_"):
            del st.session_state[key]


def render_tab_config() -> None:
    """Renders the Configuration tab."""
    st.header("Configuration")
    st.caption(
        "Load an existing TOML file or configure the simulation manually. "
        "Changes here are reflected across all tabs."
    )

    cfg: dict = st.session_state["cfg"]

    # -----------------------------------------------------------------------
    # Section 1 — Import / Export
    # -----------------------------------------------------------------------
    st.subheader("TOML File")

    if "_toml_uploader_nonce" not in st.session_state:
        st.session_state["_toml_uploader_nonce"] = 0
    uploader_key = f"toml_uploader_{st.session_state['_toml_uploader_nonce']}"

    col_import, col_export, col_reset = st.columns(3)

    with col_import:
        uploaded = st.file_uploader(
            "Load a TOML",
            type=["toml"],
            key=uploader_key,
            help="Import a simulation.toml file and fill in all fields.",
        )
        if uploaded is not None:
            # getvalue() always reads from the start of the buffer, unlike read()
            # which advances the read position and returns b"" on subsequent rerenders
            # (Streamlit may reuse the same BytesIO object between rerenders).
            raw = uploaded.getvalue()
            if not raw:
                # Empty read — buffer consumed on this render, silently ignore.
                pass
            else:
                # Deduplication based on content (hash) — not on name/size.
                # Allows re-uploading the same file after modification.
                upload_id = hashlib.md5(raw).hexdigest()  # noqa: S324 — non-cryptographic use
                if st.session_state.get("_last_upload_id") != upload_id:
                    try:
                        new_state, warnings = toml_bytes_to_state(raw)
                        st.session_state["cfg"] = new_state
                        st.session_state["toml_active_path"] = uploaded.name
                        st.session_state["_last_upload_id"] = upload_id
                        # Uploaded file: no path on disk → no base_dir
                        st.session_state["cfg_base_dir"] = None
                        # Store the original name and hash to detect subsequent
                        # modifications and display "(modified)" in the bar.
                        st.session_state["_original_toml_name"] = uploaded.name
                        st.session_state["_original_cfg_hash"] = _cfg_hash(new_state)
                        # Invalidate the temp cache to force an immediate rewrite.
                        st.session_state.pop("_active_tmp_cfg_hash", None)
                        st.session_state.pop("_active_tmp_toml_path", None)
                        _clear_cfg_widget_keys()
                        for w in warnings:
                            st.warning(w)
                        st.success(f"Loaded: **{uploaded.name}**")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error loading TOML: {exc}")
        else:
            # File removed from the uploader: clear the cache to allow
            # a future re-upload of the same file.
            st.session_state.pop("_last_upload_id", None)

    with col_export:
        toml_content = state_to_toml(cfg)
        st.download_button(
            label="Export config",
            data=toml_content.encode("utf-8"),
            file_name="simulation.toml",
            mime="text/plain",
            help="Downloads simulation.toml from the current settings.",
        )

    with col_reset:
        if st.button("Reset", help="Resets the configuration to an empty state (no loaded config)."):
            st.session_state["cfg"] = copy.deepcopy(DEFAULT_STATE)
            st.session_state["toml_active_path"] = None
            st.session_state["cfg_base_dir"] = None
            st.session_state.pop("_last_upload_id", None)  # allow re-upload
            # Force a new widget identifier to clear the file_uploader on the browser side.
            st.session_state["_toml_uploader_nonce"] += 1
            st.session_state.pop(uploader_key, None)
            # Clear all traces of the old loaded/temp config.
            for _k in ("_original_toml_name", "_original_cfg_hash", "_active_tmp_cfg_hash", "_active_tmp_toml_path"):
                st.session_state.pop(_k, None)
            # Effacer les caches de résultats de simulation.
            st.session_state.pop("sim_total_result", None)
            for _k in [
                k
                for k in st.session_state
                if isinstance(k, str) and (k.startswith("_res_total_") or k.startswith("_scan_struct_"))
            ]:
                del st.session_state[_k]
            # Reset run states (generation, simulation, full run).
            for _k in (
                "gen_running",
                "gen_done",
                "gen_error",
                "gen_log_messages",
                "gen_files_count",
                "gen_total_size_mb",
                "sim_running",
                "sim_done",
                "sim_error",
                "sim_log_messages",
                "full_run_running",
                "full_run_done",
                "full_run_error",
                "full_run_logs",
                "_full_run_final_rerun_done",
                "_sim_final_rerun_done",
                "_cfg_saved_snapshot",
                "_cfg_saved_gen_models",
                "_cfg_post_run_restore",
                "_cfg_saved_rule_codes",
            ):
                st.session_state.pop(_k, None)
            _clear_cfg_widget_keys()
            st.rerun()

    active_path = st.session_state.get("toml_active_path") or "—"
    if active_path != "—":
        st.caption(f"Active config: `{active_path}`")
    else:
        st.caption("No configuration loaded — use the import above.")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2 — Main parameters
    # -----------------------------------------------------------------------
    st.subheader("Simulation parameters")

    # Apply the folder picker result BEFORE instantiating the widget
    # (we cannot write to session_state[key] after a widget with that key
    # has already been rendered in the same rerun).
    if "_folder_picker_result" in st.session_state:
        st.session_state["cfg_output_base_path"] = st.session_state.pop("_folder_picker_result")

    col_path, col_browse, col_seed = st.columns([5, 1, 3])

    with col_path:
        new_path = st.text_input(
            "Output folder (output_base_path)",
            key="cfg_output_base_path",
            placeholder="E.g.: ../data/ or /home/user/results",
            help="Root directory for gen/ and sim_result/. Leave empty to use the current directory.",
        )
    cfg["output_base_path"] = new_path

    with col_browse:
        # Vertical space to align the button with the text_input
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("📁", help="Open a folder picker", key="cfg_browse_btn"):
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", 1)
                folder = filedialog.askdirectory(title="Choose the output folder")
                root.destroy()
                if folder:
                    # Store in a temporary key — will be applied on the next rerun
                    # BEFORE instantiating the cfg_output_base_path widget.
                    st.session_state["_folder_picker_result"] = folder
                    st.rerun()
            except Exception as exc:
                st.error(f"Folder picker unavailable: {exc}")

    with col_seed:
        new_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=None,
            step=1,
            key="cfg_seed",
            placeholder="Random if empty",
            help="Seed for reproducibility. Leave empty for a random seed.",
        )
    cfg["seed"] = int(new_seed) if new_seed is not None else None

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3 — TOML preview (reuses toml_content already computed above)
    # -----------------------------------------------------------------------
    st.subheader("Current TOML preview")
    st.code(toml_content, language="toml")

    # -----------------------------------------------------------------------
    # Section 4 — Advanced per-model parameters
    # -----------------------------------------------------------------------
    with st.expander("Advanced per-model parameters (generator_params)", expanded=False):
        st.caption(
            "The parameters here correspond to `[generator_params.<MODEL>]` sub-tables in the TOML. "
            "They are optional — leave empty to use the default values."
        )

        gen_models = cfg.get("generative_models", [])
        if not gen_models:
            st.info("Select generative models first in the **Data** tab.")
        else:
            gp = cfg.get("generator_params", {})
            for model in gen_models:
                st.markdown(f"**{model}**")
                # Afficher les clés connues selon le modèle
                model_params = gp.get(model, {})
                _render_model_params(model, model_params, gp)
            cfg["generator_params"] = gp


def _render_model_params(model: str, current: dict, gp: dict) -> None:
    """Renders the parameter fields for a given model."""
    KNOWN_PARAMS: dict[str, dict] = {
        "VMF_HC": {"vmf_concentration": ("Concentration VMF", 10.0, 0.01, 1000.0)},
        "VMF_HS": {
            "vmf_concentration": ("Concentration VMF", 10.0, 0.01, 1000.0),
            "stretching": ("Stretching", 1.0, 0.01, 100.0),
        },
        "EUCLID": {},  # box_dimensions est une liste, géré différemment
        "EUCLID_5D": {},
        "GAUSS": {"sigma": ("Sigma", 1.0, 0.001, 100.0)},
        "SPHEROID": {"stretching": ("Stretching", 1.0, 0.01, 100.0)},
        "PERTURB": {"theta": ("Theta", 0.1, 0.0, 1.0)},
        "UFR": {"n_max_rankings": ("Max rankings", 4, 1, 100)},
        "LADDER": {"n_rungs": ("Number of rungs", 21, 2, 200)},
    }

    params_def = KNOWN_PARAMS.get(model, {})
    if not params_def:
        st.caption(f"No known advanced parameters for {model}.")
        return

    updated: dict = dict(current)
    cols = st.columns(min(len(params_def), 3))
    for idx, (param_key, param_info) in enumerate(params_def.items()):
        label, default, min_val, max_val = param_info
        col = cols[idx % len(cols)]
        if isinstance(default, float):
            val = col.number_input(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(current.get(param_key, default)),
                step=0.01,
                key=f"gp_{model}_{param_key}",
            )
        else:
            val = col.number_input(
                label,
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(current.get(param_key, default)),
                step=1,
                key=f"gp_{model}_{param_key}",
            )
        updated[param_key] = val

    gp[model] = updated

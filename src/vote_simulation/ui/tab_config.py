"""Onglet 1 — Configuration TOML.

Permet de charger, modifier et exporter la configuration simulation
sans toucher au code. Toute modification se répercute dans st.session_state.
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
    """Retourne un hash MD5 du TOML sérialisé depuis *state*."""
    return hashlib.md5(state_to_toml(state).encode()).hexdigest()  # noqa: S324


def _clear_cfg_widget_keys() -> None:
    """Synchronise tous les widgets pilotés par cfg avec les valeurs du cfg courant.

    Doit être appelé avant st.rerun() après tout chargement externe de cfg
    (upload TOML, réinitialisation).

    IMPORTANT : effacer les clés session_state ne suffit pas — Streamlit restaure
    la valeur soumise par le navigateur (stale) au lieu d'utiliser value=/default=.
    On écrit donc explicitement chaque valeur dans session_state depuis cfg, ce qui
    prend la priorité absolue sur la valeur du navigateur.
    """
    cfg = st.session_state.get("cfg", {})

    # ---- Widgets simples (text_input, number_input, slider) ----
    # Écrire directement la valeur cfg dans session_state → priorité absolue.
    st.session_state["cfg_output_base_path"] = cfg.get("output_base_path") or ""
    seed_val = cfg.get("seed")
    st.session_state["cfg_seed"] = int(seed_val) if seed_val is not None else None
    st.session_state["gen_voters_input"] = ", ".join(str(v) for v in cfg.get("voters", []))
    st.session_state["gen_candidates_input"] = ", ".join(str(c) for c in cfg.get("candidates", []))
    st.session_state["gen_iterations_slider"] = int(cfg.get("iterations", 1000))

    # ──── Widgets optionnels (radio/text source données) — simple suppression suffit
    for key in ("sim_data_source", "sim_input_folder"):
        st.session_state.pop(key, None)

    # ---- Multiselects (dépendent de registres externes chargés dans les onglets) ----
    # Utiliser des flags consommés par chaque onglet au début de son rendu.
    st.session_state["_cfg_gen_needs_sync"] = True  # → gen_models_select dans render_tab_generation
    st.session_state["_cfg_rules_needs_sync"] = True  # → rules_family_* dans render_tab_simulation

    # Vider les anciennes clés rules_family_* pour éviter des conflits de type
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("rules_family_"):
            del st.session_state[key]


def render_tab_config() -> None:
    """Rend l'onglet Configuration."""
    st.header("Configuration")
    st.caption(
        "Chargez un fichier TOML existant ou paramétrez la simulation manuellement. "
        "Les changements ici se reflètent dans tous les onglets."
    )

    cfg: dict = st.session_state["cfg"]

    # -----------------------------------------------------------------------
    # Section 1 — Import / Export
    # -----------------------------------------------------------------------
    st.subheader("Fichier TOML")

    col_import, col_export, col_reset = st.columns(3)

    with col_import:
        uploaded = st.file_uploader(
            "Charger un TOML",
            type=["toml"],
            key="toml_uploader",
            help="Importe un fichier simulation.toml et remplit tous les champs.",
        )
        if uploaded is not None:
            # getvalue() lit toujours depuis le début du buffer, contrairement à read()
            # qui avance la position de lecture et retourne b"" sur les rerenders suivants
            # (Streamlit peut réutiliser le même objet BytesIO entre rerenders).
            raw = uploaded.getvalue()
            if not raw:
                # Lecture vide — buffer consommé sur ce render, ignorer silencieusement.
                pass
            else:
                # Dédoublonnage basé sur le contenu (hash) — pas sur le nom/taille.
                # Permet de ré-uploader le même fichier après modification.
                upload_id = hashlib.md5(raw).hexdigest()  # noqa: S324 — non-cryptographic use
                if st.session_state.get("_last_upload_id") != upload_id:
                    try:
                        new_state, warnings = toml_bytes_to_state(raw)
                        st.session_state["cfg"] = new_state
                        st.session_state["toml_active_path"] = uploaded.name
                        st.session_state["_last_upload_id"] = upload_id
                        # Fichier uploadé : pas de chemin sur disque → pas de base_dir
                        st.session_state["cfg_base_dir"] = None
                        # Mémoriser le nom et le hash d'origine pour détecter les
                        # modifications ultérieures et afficher "(modifié)" dans la barre.
                        st.session_state["_original_toml_name"] = uploaded.name
                        st.session_state["_original_cfg_hash"] = _cfg_hash(new_state)
                        # Invalider le cache temp pour forcer une réécriture immédiate.
                        st.session_state.pop("_active_tmp_cfg_hash", None)
                        st.session_state.pop("_active_tmp_toml_path", None)
                        _clear_cfg_widget_keys()
                        for w in warnings:
                            st.warning(w)
                        st.success(f"Chargé : **{uploaded.name}**")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Erreur lors du chargement du TOML : {exc}")
        else:
            # Fichier retiré de l'uploader : effacer le cache pour autoriser
            # un futur re-upload du même fichier.
            st.session_state.pop("_last_upload_id", None)

    with col_export:
        toml_content = state_to_toml(cfg)
        st.download_button(
            label="Exporter la config",
            data=toml_content.encode("utf-8"),
            file_name="simulation.toml",
            mime="text/plain",
            help="Télécharge simulation.toml à partir des réglages courants.",
        )

    with col_reset:
        if st.button("Réinitialiser", help="Remet la configuration à zéro (état vide, sans config chargée)."):
            st.session_state["cfg"] = copy.deepcopy(DEFAULT_STATE)
            st.session_state["toml_active_path"] = None
            st.session_state["cfg_base_dir"] = None
            st.session_state.pop("_last_upload_id", None)  # autorise un re-upload
            # Effacer toutes les traces de l'ancienne config chargée/temp.
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
            # Réinitialiser les états de run (génération, simulation, run complet).
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
                "_cfg_saved_gen_models",
                "_cfg_post_run_restore",
                "_cfg_saved_rule_codes",
            ):
                st.session_state.pop(_k, None)
            _clear_cfg_widget_keys()
            st.rerun()

    active_path = st.session_state.get("toml_active_path") or "—"
    if active_path != "—":
        st.caption(f"Config active : `{active_path}`")
    else:
        st.caption("Aucune configuration chargée — utilisez l'import ci-dessus.")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2 — Paramètres principaux
    # -----------------------------------------------------------------------
    st.subheader("Paramètres de simulation")

    # Appliquer le résultat du sélecteur de dossier AVANT l'instanciation du widget
    # (on ne peut pas écrire dans session_state[key] après qu'un widget avec ce key
    # a déjà été rendu dans ce même rerun).
    if "_folder_picker_result" in st.session_state:
        st.session_state["cfg_output_base_path"] = st.session_state.pop("_folder_picker_result")

    col_path, col_browse, col_seed = st.columns([5, 1, 3])

    with col_path:
        new_path = st.text_input(
            "Dossier de sortie (output_base_path)",
            key="cfg_output_base_path",
            placeholder="Ex : ../data/ ou /home/user/results",
            help="Répertoire racine pour gen/ et sim_result/. Laissez vide pour utiliser le répertoire courant.",
        )
    cfg["output_base_path"] = new_path

    with col_browse:
        # Espace vertical pour aligner le bouton avec le text_input
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("📁", help="Ouvrir un sélecteur de dossier", key="cfg_browse_btn"):
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", 1)
                folder = filedialog.askdirectory(title="Choisir le dossier de sortie")
                root.destroy()
                if folder:
                    # Stocker dans une clé temporaire — sera appliquée au prochain rerun
                    # AVANT l'instanciation du widget cfg_output_base_path.
                    st.session_state["_folder_picker_result"] = folder
                    st.rerun()
            except Exception as exc:
                st.error(f"Sélecteur de dossier indisponible : {exc}")

    with col_seed:
        new_seed = st.number_input(
            "Seed aléatoire",
            min_value=0,
            max_value=2**31 - 1,
            value=None,
            step=1,
            key="cfg_seed",
            placeholder="Aléatoire si vide",
            help="Graine pour la reproductibilité. Laissez vide pour une seed aléatoire.",
        )
    cfg["seed"] = int(new_seed) if new_seed is not None else None

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3 — Aperçu TOML (réutilise toml_content déjà calculé ci-dessus)
    # -----------------------------------------------------------------------
    st.subheader("Aperçu du TOML courant")
    st.code(toml_content, language="toml")

    # -----------------------------------------------------------------------
    # Section 4 — Paramètres avancés par modèle
    # -----------------------------------------------------------------------
    with st.expander("Paramètres avancés par modèle (generator_params)", expanded=False):
        st.caption(
            "Les paramètres ici correspondent aux sous-tables `[generator_params.<MODEL>]` du TOML. "
            "Ils sont facultatifs — laissez vide pour utiliser les valeurs par défaut."
        )

        gen_models = cfg.get("generative_models", [])
        if not gen_models:
            st.info("Sélectionnez d'abord des modèles génératifs dans l'onglet **Données**.")
        else:
            gp = cfg.get("generator_params", {})
            for model in gen_models:
                st.markdown(f"**{model}**")
                # Afficher les clés connues selon le modèle
                model_params = gp.get(model, {})
                _render_model_params(model, model_params, gp)
            cfg["generator_params"] = gp


def _render_model_params(model: str, current: dict, gp: dict) -> None:
    """Affiche les champs de paramètres pour un modèle donné."""
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
        "UFR": {"n_max_rankings": ("Nb max rankings", 4, 1, 100)},
        "LADDER": {"n_rungs": ("Nb de barreaux", 21, 2, 200)},
    }

    params_def = KNOWN_PARAMS.get(model, {})
    if not params_def:
        st.caption(f"Aucun paramètre avancé connu pour {model}.")
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

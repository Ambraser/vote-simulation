"""Onglet 3 — Simulation (règles de vote).

Permet de sélectionner les règles de vote organisées par famille,
de choisir la source des données, et de lancer simulation_from_config()
dans un thread séparé avec suivi de progression.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from vote_simulation.ui.toml_utils import write_temp_toml, QueueWriter


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

# Ordre affiché dans l'UI
_FAMILIES: list[tuple[str, str]] = [
    ("Pluralité / IRV", "PLURALITY_IRV"),
    ("Condorcet", "CONDORCET"),
    ("Score / Borda", "SCORE_BORDA"),
    ("Jugement / Score continu", "JUDGMENT"),
    ("Approbation — seuil", "AP_THRESHOLD"),
    ("Approbation — K", "AP_K"),
    ("Autres", "OTHER"),
]

_FAMILY_PREFIXES: dict[str, list[str]] = {
    "PLURALITY_IRV": ["PLU", "HARE", "IRV", "IRVA", "IRVD", "ICRV", "SIRV", "EXHB", "RPAR", "BUCK", "IBUC", "L4VD", "SPCY", "CAIR"],
    "CONDORCET": ["COPE", "SCHU", "MMAX", "BLAC", "KEME", "SLAT", "KIMR", "WOOD", "YOUN", "TIDE", "DODG", "CSUM", "CVIR"],
    "SCORE_BORDA": ["BORD", "COOM", "NANS", "PV-"],
    "JUDGMENT": ["MJ", "RV", "STAR", "VETO"],
    "AP_THRESHOLD": ["AP_T"],
    "AP_K": ["AP_K"],
}


def _classify_rule(code: str) -> str:
    """Retourne la famille d'une règle à partir de son code."""
    upper = code.upper()
    for family_id, prefixes in _FAMILY_PREFIXES.items():
        for pfx in prefixes:
            if upper.startswith(pfx):
                return family_id
    return "OTHER"


def _build_family_map(all_codes: list[str]) -> dict[str, list[str]]:
    """Construit un dictionnaire famille → liste de codes."""
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
    """Cible du thread : lance simulation_series_from_config() avec capture des logs."""
    from vote_simulation.simulation.simulation import simulation_series_from_config
    import vote_simulation.simulation.simulation as _sim_module

    import tqdm as tqdm_module
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

        def update(self, n: int = 1) -> bool | None:
            self.n += n
            if stop_event.is_set():
                self.close()
                raise InterruptedError("Simulation annulée par l'utilisateur.")
            if self._track:
                now = time.monotonic()
                if now - self._last_st_update >= 0.15:
                    total = self.total or 0
                    current = min(self.n, total) if total else self.n
                    st.session_state[_SIM_PROGRESS_KEY] = (current, total)
                    self._last_st_update = now
            return None

    tqdm_module.tqdm = _PatchedTqdm  # type: ignore[assignment]
    _sim_module.tqdm = _PatchedTqdm  # corrige la liaison `from tqdm import tqdm`
    old_stdout = sys.stdout
    sys.stdout = QueueWriter(log_q)

    try:
        simulation_series_from_config(config_path, reload=reload)

        log_q.put("Simulation terminee.")
        st.session_state[_SIM_DONE_KEY] = True
        st.session_state[_SIM_ERROR_KEY] = None
    except InterruptedError as exc:
        log_q.put(f"Annule : {exc}")
        st.session_state[_SIM_DONE_KEY] = True
        st.session_state[_SIM_ERROR_KEY] = "Annulé"
    except Exception as exc:
        log_q.put(f"Erreur : {exc}")
        st.session_state[_SIM_DONE_KEY] = True
        st.session_state[_SIM_ERROR_KEY] = str(exc)
    finally:
        sys.stdout = old_stdout
        tqdm_module.tqdm = original_tqdm  # type: ignore[assignment]
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


# ---------------------------------------------------------------------------
# Rendu principal
# ---------------------------------------------------------------------------

def render_tab_simulation() -> None:
    """Rend l'onglet Simulation."""
    _init_sim_state()

    st.header("Simulation — Règles de vote")
    st.caption(
        "Sélectionnez les règles à appliquer, choisissez la source des données, "
        "puis lancez la simulation."
    )

    cfg: dict = st.session_state["cfg"]

    # -----------------------------------------------------------------------
    # 3.1 Chargement dynamique des règles
    # -----------------------------------------------------------------------
    all_codes = _cached_rule_codes()
    family_map = _cached_family_map()

    # Synchronisation explicite depuis cfg après un chargement externe (upload TOML,
    # réinitialisation). Streamlit restaure les valeurs du navigateur (stale, souvent [])
    # au lieu d'utiliser default= quand les clés session_state ont été effacées.
    # On contourne cela en écrivant directement les bonnes valeurs dans session_state
    # AVANT de rendre les multiselects.
    if st.session_state.pop("_cfg_rules_needs_sync", False):
        loaded_rules = cfg.get("rule_codes", [])
        for _, family_id in _FAMILIES:
            codes_in_family = family_map.get(family_id, [])
            st.session_state[f"rules_family_{family_id}"] = [
                c for c in codes_in_family if c in loaded_rules
            ]

    current_rules: list[str] = cfg.get("rule_codes", [])

    # Boutons rapides
    col_all, col_none = st.columns(2)
    with col_all:
        if st.button("Tout sélectionner", key="sim_select_all"):
            cfg["rule_codes"] = list(all_codes)
            for _, family_id in _FAMILIES:
                st.session_state[f"rules_family_{family_id}"] = family_map.get(family_id, [])
            st.rerun()
    with col_none:
        if st.button("Tout désélectionner", key="sim_deselect_all"):
            cfg["rule_codes"] = []
            for _, family_id in _FAMILIES:
                st.session_state[f"rules_family_{family_id}"] = []
            st.rerun()

    st.subheader("Sélection des règles par famille")
    selected_rules: list[str] = []

    for family_label, family_id in _FAMILIES:
        codes_in_family = family_map.get(family_id, [])
        if not codes_in_family:
            continue

        with st.expander(f"{family_label} ({len(codes_in_family)} règles)", expanded=(family_id in ("PLURALITY_IRV", "SCORE_BORDA"))):
            # Boutons select/deselect pour cette famille
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button(f"Tout — {family_label}", key=f"sel_all_{family_id}"):
                    existing = set(cfg.get("rule_codes", []))
                    existing.update(codes_in_family)
                    cfg["rule_codes"] = list(existing)
                    st.session_state[f"rules_family_{family_id}"] = codes_in_family
                    st.rerun()
            with fc2:
                if st.button(f"Aucun — {family_label}", key=f"sel_none_{family_id}"):
                    existing = set(cfg.get("rule_codes", []))
                    existing.difference_update(codes_in_family)
                    cfg["rule_codes"] = list(existing)
                    st.session_state[f"rules_family_{family_id}"] = []
                    st.rerun()

            chosen = st.multiselect(
                "Règles",
                options=codes_in_family,
                default=None,
                key=f"rules_family_{family_id}",
                label_visibility="collapsed",
            )
            selected_rules.extend(chosen)

    # Mettre à jour cfg avec toutes les règles sélectionnées
    cfg["rule_codes"] = selected_rules

    # Résumé
    st.info(f"**{len(selected_rules)} règle(s) sélectionnée(s)** sur {len(all_codes)} disponibles.")

    st.divider()

    # -----------------------------------------------------------------------
    # 3.2 Source des données
    # -----------------------------------------------------------------------
    st.subheader("Source des données")

    data_source = st.radio(
        "Données à utiliser",
        options=["Données générées (onglet Données)", "Dossier existant (chemin personnalisé)"],
        index=0 if not cfg.get("input_folder_path") else 1,
        key="sim_data_source",
        horizontal=True,
    )

    if data_source == "Dossier existant (chemin personnalisé)":
        folder_path = st.text_input(
            "Chemin du dossier de données",
            value=cfg.get("input_folder_path") or cfg.get("output_base_path", "../data/"),
            key="sim_input_folder",
            help="Chemin vers le dossier contenant les sous-dossiers gen/ avec les .parquet.",
        )
        cfg["input_folder_path"] = folder_path
    else:
        cfg["input_folder_path"] = None

    st.divider()

    # -----------------------------------------------------------------------
    # 3.3 Actions
    # -----------------------------------------------------------------------
    st.subheader("Actions")

    reload_flag: bool = st.checkbox(
        "Forcer le recalcul (reload)",
        value=False,
        key="sim_reload",
        help=(
            "Coché : ignore les fichiers de résultats existants et recalcule"
            " chaque itération depuis zéro.\n"
            "Décoché (défaut) : passe les itérations dont le fichier existe déjà."
        ),
    )

    is_running: bool = st.session_state[_SIM_RUNNING_KEY]

    col_run, col_cancel = st.columns(2)

    with col_run:
        run_disabled = is_running or not selected_rules or not cfg.get("generative_models")
        if st.button(
            "Lancer la simulation",
            disabled=run_disabled,
            type="primary",
            key="sim_run_btn",
        ):
            if not selected_rules:
                st.error("Sélectionnez au moins une règle de vote.")
            elif not cfg.get("generative_models"):
                st.error("Aucun modèle génératif configuré (onglet Données).")
            else:
                stop_event = threading.Event()
                log_q: queue.Queue = queue.Queue()
                st.session_state[_SIM_STOP_KEY] = stop_event
                st.session_state[_SIM_LOG_Q_KEY] = log_q
                st.session_state[_SIM_PROGRESS_KEY] = (0, 0)
                st.session_state[_SIM_RUNNING_KEY] = True
                st.session_state[_SIM_DONE_KEY] = False
                st.session_state[_SIM_ERROR_KEY] = None
                st.session_state["sim_log_messages"] = []

                tmp_path = write_temp_toml(cfg, base_dir=st.session_state.get("cfg_base_dir"))
                st.session_state["sim_tmp_toml"] = tmp_path

                t = threading.Thread(
                    target=_run_simulation,
                    args=(tmp_path, stop_event, log_q, reload_flag),
                    daemon=True,
                )
                add_script_run_ctx(t, get_script_run_ctx())
                t.start()
                st.session_state[_SIM_THREAD_KEY] = t
                st.rerun()

    with col_cancel:
        if st.button(
            "Annuler",
            disabled=not is_running,
            key="sim_cancel_btn",
        ):
            stop: threading.Event = st.session_state[_SIM_STOP_KEY]
            stop.set()
            st.warning("Arrêt demandé — la simulation s'interrompra à la prochaine itération.")

    # -----------------------------------------------------------------------
    # Feedback
    # -----------------------------------------------------------------------
    _render_sim_feedback()


def _render_sim_feedback() -> None:
    """Affiche barre de progression et logs pour la simulation."""
    is_running: bool = st.session_state[_SIM_RUNNING_KEY]
    is_done: bool = st.session_state[_SIM_DONE_KEY]
    error: str | None = st.session_state[_SIM_ERROR_KEY]
    progress_tuple: tuple = st.session_state[_SIM_PROGRESS_KEY]

    new_msgs = _drain_log_queue()
    if new_msgs:
        st.session_state["sim_log_messages"].extend(new_msgs)

    current, total = progress_tuple
    if total > 0:
        pct = min(current / total, 1.0)
        st.progress(pct, text=f"Progression : {current}/{total} profils simulés")
    elif is_running:
        st.progress(0.0, text="Initialisation de la simulation…")

    all_logs = st.session_state.get("sim_log_messages", [])
    if all_logs:
        with st.expander("Logs de simulation", expanded=False):
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
        if error and error != "Annulé":
            st.error(f"Erreur : {error}")
        elif error == "Annulé":
            st.warning("Simulation annulée.")
        else:
            total_profiles = (
                len(cfg.get("generative_models", []))
                * len(cfg.get("voters", []))
                * len(cfg.get("candidates", []))
                * cfg.get("iterations", 0)
            )
            st.success(
                f"**Simulation terminée** — {n_rules} règles x {total_profiles:,} profils traités.\n\n"
                f"Résultats disponibles dans l'onglet Résultats."
            )

    if is_running:
        thread: threading.Thread | None = st.session_state[_SIM_THREAD_KEY]
        if thread is not None and thread.is_alive():
            with st.spinner("Simulation en cours…"):
                time.sleep(1.0)
            st.rerun()
        else:
            st.session_state[_SIM_RUNNING_KEY] = False
            st.rerun()

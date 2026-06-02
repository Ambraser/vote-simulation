"""Onglet 2 — Génération de données.

Permet de sélectionner les modèles génératifs, les combinaisons voters×candidats,
le nombre d'itérations, et de lancer generate_data() dans un thread séparé
avec suivi de progression en temps réel.
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

from vote_simulation.ui.toml_utils import QueueWriter, write_temp_toml


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
    """Vide la queue de logs et retourne les messages."""
    q: queue.Queue = st.session_state[_LOG_QUEUE_KEY]
    messages = []
    try:
        while True:
            messages.append(q.get_nowait())
    except queue.Empty:
        pass
    return messages


def _run_generation(config_path: str, stop_event: threading.Event, log_q: queue.Queue) -> None:
    """Cible du thread : lance generate_data() avec capture des logs."""
    # Patch tqdm pour émettre dans la queue
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
                raise InterruptedError("Génération annulée par l'utilisateur.")
            now = time.monotonic()
            if now - self._last_st_update >= 0.15:
                total = self.total or 0
                current = min(self.n, total) if total else self.n
                st.session_state[_PROGRESS_KEY] = (current, total)
                self._last_st_update = now
            return result

    tqdm_module.tqdm = _PatchedTqdm  # type: ignore[assignment]
    _sim_module.tqdm = _PatchedTqdm  # type: ignore[assignment]  # corrige la liaison `from tqdm import tqdm`

    # Capturer stdout
    old_stdout = sys.stdout
    sys.stdout = QueueWriter(log_q)

    try:
        paths = generate_data(config_path, show_progress=True)
        st.session_state["gen_files_count"] = len(paths)
        total_bytes = sum(Path(p).stat().st_size for p in paths if Path(p).is_file())
        st.session_state["gen_total_size_mb"] = total_bytes / (1024 * 1024)
        log_q.put(f"Termine : {len(paths)} fichiers generes.")
        st.session_state[_DONE_KEY] = True
        st.session_state[_ERROR_KEY] = None
    except InterruptedError as exc:
        log_q.put(f"Annule : {exc}")
        st.session_state[_DONE_KEY] = True
        st.session_state[_ERROR_KEY] = "Annulé"
    except Exception as exc:
        log_q.put(f"Erreur : {exc}")
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
    """Parse une chaîne comme '11, 101, 1001' en liste d'entiers."""
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
    """Rend l'onglet Génération de données."""
    _init_gen_state()

    st.header("Génération de données")
    st.caption(
        "Configurez les modèles génératifs et les combinaisons voters × candidats, "
        "puis lancez la génération des profils de vote."
    )

    cfg: dict = st.session_state["cfg"]

    # -----------------------------------------------------------------------
    # 2.1 Modèles génératifs
    # -----------------------------------------------------------------------
    st.subheader("Modèles génératifs")

    # Import différé pour éviter un import circulaire au niveau module
    all_codes = _cached_generator_codes()

    # Synchronisation depuis cfg après chargement externe (upload TOML, réinitialisation).
    # Même logique que pour les rules_family_* : on écrit session_state avant de rendre
    # le multiselect pour que cfg ait la priorité sur la valeur soumise par le navigateur.
    if st.session_state.pop("_cfg_gen_needs_sync", False):
        st.session_state["gen_models_select"] = [m for m in cfg.get("generative_models", []) if m in all_codes]

    selected_models = st.multiselect(
        "Modèles sélectionnés",
        options=all_codes,
        default=None,
        key="gen_models_select",
        help="Codes des modèles génératifs enregistrés dans le registre.",
    )
    cfg["generative_models"] = selected_models

    # -----------------------------------------------------------------------
    # 2.2 Combinaisons de simulation
    # -----------------------------------------------------------------------
    st.subheader("Combinaisons de simulation")

    col_v, col_c, col_i = st.columns([2, 2, 3])

    with col_v:
        voters_str = st.text_input(
            "Nombre de votants (liste)",
            key="gen_voters_input",
            help="Entiers séparés par des virgules. Ex : 11, 101, 1001",
        )
        voters = _parse_int_list(voters_str)
        if voters:
            cfg["voters"] = voters
        else:
            st.warning("Entrez au moins un nombre de votants.")

    with col_c:
        candidates_str = st.text_input(
            "Nombre de candidats (liste)",
            key="gen_candidates_input",
            help="Entiers séparés par des virgules. Ex : 3, 14",
        )
        candidates = _parse_int_list(candidates_str)
        if candidates:
            cfg["candidates"] = candidates
        else:
            st.warning("Entrez au moins un nombre de candidats.")

    with col_i:
        if "gen_iterations_slider" not in st.session_state:
            st.session_state["gen_iterations_slider"] = int(cfg.get("iterations", 1000))
        iterations = st.slider(
            "Nombre d'itérations",
            min_value=1,
            max_value=10_000,
            step=1,
            key="gen_iterations_slider",
            help="Nombre d'itérations par combinaison (modèle × voters × candidats).",
        )
        cfg["iterations"] = iterations

    # Indicateur temps réel
    n_models = len(selected_models)
    n_voters = len(cfg.get("voters", []))
    n_cands = len(cfg.get("candidates", []))
    total_profiles = n_models * n_voters * n_cands * iterations
    st.info(
        f"**Profils à générer :** {n_models} modèles × {n_voters} voters × "
        f"{n_cands} candidats × {iterations} itérations = **{total_profiles:,} profils**"
    )

    # -----------------------------------------------------------------------
    # 2.3 Actions — Générer / Annuler
    # -----------------------------------------------------------------------
    st.subheader("Actions")

    is_running: bool = st.session_state[_RUNNING_KEY]

    col_run, col_cancel = st.columns(2)

    with col_run:
        run_disabled = is_running or not selected_models or total_profiles == 0
        if st.button(
            "Generer les donnees",
            disabled=run_disabled,
            type="primary",
            key="gen_run_btn",
        ):
            if not selected_models:
                st.error("Sélectionnez au moins un modèle génératif.")
            elif not cfg.get("voters") or not cfg.get("candidates"):
                st.error("Configurez les listes de voters et de candidats.")
            else:
                # Réinitialiser l'état
                stop_event = threading.Event()
                log_q: queue.Queue = queue.Queue()
                st.session_state[_STOP_EVENT_KEY] = stop_event
                st.session_state[_LOG_QUEUE_KEY] = log_q
                st.session_state[_PROGRESS_KEY] = (0, total_profiles)
                st.session_state[_RUNNING_KEY] = True
                st.session_state[_DONE_KEY] = False
                st.session_state[_ERROR_KEY] = None
                st.session_state["gen_log_messages"] = []

                # Écriture du TOML temporaire
                tmp_path = write_temp_toml(cfg, base_dir=st.session_state.get("cfg_base_dir"))
                st.session_state["gen_tmp_toml"] = tmp_path

                t = threading.Thread(
                    target=_run_generation,
                    args=(tmp_path, stop_event, log_q),
                    daemon=True,
                )
                add_script_run_ctx(t, get_script_run_ctx())
                t.start()
                st.session_state[_THREAD_KEY] = t
                st.rerun()

    with col_cancel:
        if st.button(
            "Annuler",
            disabled=not is_running,
            key="gen_cancel_btn",
        ):
            stop: threading.Event = st.session_state[_STOP_EVENT_KEY]
            stop.set()
            st.warning("Arrêt demandé — la génération s'interrompra à la prochaine itération.")

    # -----------------------------------------------------------------------
    # Feedback — Progression et logs
    # -----------------------------------------------------------------------
    _render_gen_feedback()


def _render_gen_feedback() -> None:
    """Affiche la barre de progression et la zone de logs."""
    is_running: bool = st.session_state[_RUNNING_KEY]
    is_done: bool = st.session_state[_DONE_KEY]
    error: str | None = st.session_state[_ERROR_KEY]
    progress_tuple: tuple = st.session_state[_PROGRESS_KEY]

    # Accumuler les logs dans session_state
    if "gen_log_messages" not in st.session_state:
        st.session_state["gen_log_messages"] = []
    new_msgs = _drain_log_queue()
    if new_msgs:
        st.session_state["gen_log_messages"].extend(new_msgs)

    # Barre de progression
    current, total = progress_tuple
    if total > 0:
        pct = min(current / total, 1.0)
        st.progress(pct, text=f"Progression : {current}/{total} profils")
    elif is_running:
        st.progress(0.0, text="Démarrage…")

    # Zone de logs scrollable
    all_logs = st.session_state.get("gen_log_messages", [])
    if all_logs:
        with st.expander("Logs de génération", expanded=False):
            st.text_area(
                "Logs",
                value="\n".join(all_logs[-200:]),
                height=200,
                disabled=True,
                key="gen_log_area",
                label_visibility="collapsed",
            )

    # Résumé final
    if is_done and not is_running:
        if error and error != "Annulé":
            st.error(f"Erreur : {error}")
        elif error == "Annulé":
            st.warning("Génération annulée.")
        else:
            n_files = st.session_state.get("gen_files_count", 0)
            size_mb = st.session_state.get("gen_total_size_mb", 0.0)
            st.success(f"**Génération terminée** — {n_files} fichiers générés ({size_mb:.2f} Mo au total).")

    # Polling : si le thread tourne, relancer le script toutes les ~0.5 s
    if is_running:
        thread: threading.Thread | None = st.session_state[_THREAD_KEY]
        if thread is not None and thread.is_alive():
            with st.spinner("Génération en cours…"):
                time.sleep(0.3)
            st.rerun()
        else:
            # Thread fini mais flag pas encore remis à False (race condition légère)
            st.session_state[_RUNNING_KEY] = False
            st.rerun()

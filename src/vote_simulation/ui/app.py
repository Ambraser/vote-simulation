"""Application Streamlit principale — vote_simulation UI.

Structure :
    Barre globale (statut, TOML actif, Run complet)
    └── Onglet 1 : Configuration
    └── Onglet 2 : Données (génération)
    └── Onglet 3 : Simulation (règles)
    └── Onglet 4 : Résultats

Lancement :
    streamlit run src/vote_simulation/ui/app.py
    # ou via l'entry-point :
    vote-sim-ui
"""

from __future__ import annotations

import copy
import hashlib
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from vote_simulation.ui.tab_config import render_tab_config
from vote_simulation.ui.tab_generation import _parse_int_list, render_tab_generation
from vote_simulation.ui.tab_results import render_tab_results
from vote_simulation.ui.tab_simulation import _FAMILIES, render_tab_simulation
from vote_simulation.ui.toml_utils import DEFAULT_STATE, state_to_toml, write_temp_toml

# ---------------------------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="vote_simulation UI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Initialisation de session_state
# ---------------------------------------------------------------------------


def _init_session() -> None:
    if "cfg" not in st.session_state:
        # Démarrer sans configuration par défaut — l'utilisateur charge explicitement un TOML.
        st.session_state["cfg"] = copy.deepcopy(DEFAULT_STATE)
        st.session_state["toml_active_path"] = None  # Aucune config chargée
        st.session_state["cfg_base_dir"] = None

    # Statuts globaux
    if "global_status" not in st.session_state:
        st.session_state["global_status"] = "Prêt"

    # État du Run complet
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
    # Verrou anti-boucle pour le rerun final du fragment
    if "_full_run_final_rerun_done" not in st.session_state:
        st.session_state["_full_run_final_rerun_done"] = False


# ---------------------------------------------------------------------------
# Run complet — Génération + Simulation enchaînées
# ---------------------------------------------------------------------------


def _simulation_target(
    config_path: str,
    reload: bool,
    mp_queue: mp.Queue[tuple],
    mp_stop: Any,
) -> None:
    """Exécuté dans un sous-processus forké — GIL propre, sans concurrence Streamlit.

    Patche le tqdm du module simulation pour émettre des messages de progression
    via mp_queue. Les tqdms internes (disable=True) restent des no-ops.
    Envoie enfin ("done", result) ou ("error", msg).
    """
    import io

    import vote_simulation.simulation.simulation as _sim_module
    from vote_simulation.simulation.simulation import simulation_series_from_config_2

    original_tqdm = _sim_module.tqdm

    class _ProcTqdm(original_tqdm):
        def __init__(self, *args, **kwargs) -> None:
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
        # Serialize via a temp file instead of the queue (avoids mp queue size limits
        # and multiprocessing pickle overhead for large numpy-heavy objects).
        import os
        import pickle
        import tempfile
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="vote_sim_result_")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            mp_queue.put(("done_file", tmp_path))
        except Exception:
            # Fallback: results are already on disk; main process will re-read them.
            mp_queue.put(("done", None))
    except InterruptedError as exc:
        mp_queue.put(("cancelled", str(exc)))
    except Exception as exc:
        mp_queue.put(("error", str(exc)))
    finally:
        _sim_module.tqdm = original_tqdm


def _run_full(
    config_path: str,
    stop_event: threading.Event,
    log_q: queue.Queue,
    reload: bool = False,
) -> None:
    """Thread moniteur : lance la simulation dans un sous-processus forké et relaie sa progression.

    Le sous-processus dispose de son propre GIL (aucune concurrence avec l'event
    loop Tornado / asyncio de Streamlit), ce qui lui permet d'atteindre les
    performances d'exécution d'un notebook — sans aucune modification du code
    de simulation lui-même.
    """
    _ctx = mp.get_context("fork")
    mp_queue: mp.Queue = _ctx.Queue()
    mp_stop: object = _ctx.Event()
    proc = _ctx.Process(
        target=_simulation_target,
        args=(config_path, reload, mp_queue, mp_stop),
        daemon=True,
    )
    proc.start()
    last_st_update: float = 0.0

    try:
        while True:
            # Annulation demandée depuis l'UI
            if stop_event.is_set():
                mp_stop.set()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                raise InterruptedError("Run complet annulé.")

            # Détection de fin anormale (crash du sous-processus sans message)
            if not proc.is_alive() and mp_queue.empty():
                ec = proc.exitcode
                raise RuntimeError(f"Le processus de simulation s'est terminé inopinément (code {ec}).")

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
                log_q.put("Run complet terminé.")
                st.session_state["full_run_done"] = True
                st.session_state["full_run_error"] = None
                st.session_state["global_status"] = "Terminé"
                proc.join()
                return
            elif msg_type == "done":
                # Fallback: subprocess could not serialize the result.
                _cur_total = st.session_state.get("full_run_progress", (0, 1))[1]
                st.session_state["full_run_progress"] = (_cur_total, _cur_total)
                log_q.put("Run complet terminé.")
                st.session_state["full_run_done"] = True
                st.session_state["full_run_error"] = None
                st.session_state["global_status"] = "Terminé"
                proc.join()
                return
            elif msg_type == "cancelled":
                raise InterruptedError(msg[1])
            elif msg_type == "error":
                raise RuntimeError(msg[1])

    except InterruptedError as exc:
        log_q.put(f"Annulé : {exc}")
        st.session_state["full_run_done"] = True
        st.session_state["full_run_error"] = "Annulé"
        st.session_state["global_status"] = "Annulé"
    except Exception as exc:
        log_q.put(f"Erreur : {exc}")
        st.session_state["full_run_done"] = True
        st.session_state["full_run_error"] = str(exc)
        st.session_state["global_status"] = "Erreur"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()
        st.session_state["full_run_running"] = False
        st.session_state["full_run_end_time"] = time.monotonic()


# ---------------------------------------------------------------------------
# Barre globale
# ---------------------------------------------------------------------------


def _render_global_bar() -> None:
    """Affiche la barre de statut globale persistante en haut de page."""
    cfg: dict = st.session_state["cfg"]
    status: str = st.session_state.get("global_status", "Prêt")
    toml_path: str = st.session_state.get("toml_active_path") or "Aucune config chargée"
    is_running: bool = (
        st.session_state.get("full_run_running", False)
        or st.session_state.get("gen_running", False)
        or st.session_state.get("sim_running", False)
    )

    # Calcul du statut global à partir des sous-états
    if st.session_state.get("gen_running") or st.session_state.get("sim_running"):
        if st.session_state.get("gen_running"):
            status = "Génération en cours…"
        else:
            status = "Simulation en cours…"
    elif st.session_state.get("full_run_running"):
        status = "Run complet en cours…"

    st.session_state["global_status"] = status

    col_status, col_toml, col_reload, col_run = st.columns([2, 3, 2, 2])

    with col_status:
        color = (
            "#28a745"
            if "Prêt" in status or "Terminé" in status
            else ("#dc3545" if "Erreur" in status or "Annulé" in status else "#fd7e14")
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
            f'font-family:monospace; font-size:0.85em; color:#444;">TOML actif : {toml_path}</div>',
            unsafe_allow_html=True,
        )

    with col_reload:
        st.checkbox(
            "Recalcul forcé",
            key="global_full_reload",
            help="Recalcule même si les résultats de simulation existent déjà (reload).",
            disabled=is_running,
        )

    with col_run:
        full_run_disabled = is_running or not cfg.get("generative_models") or not cfg.get("rule_codes")
        if st.button(
            "Run complet",
            disabled=full_run_disabled,
            type="primary",
            key="global_full_run",
            help="Enchaîne Génération → Simulation avec la configuration courante.",
            use_container_width=True,
        ):
            # Validation défensive — le bouton est normalement désactivé si ces champs
            # sont vides, mais une corruption de session_state peut les vider entre
            # deux renders. On vérifie avant de lancer le thread pour éviter une erreur
            # peu lisible depuis load_simulation_config().
            if not cfg.get("rule_codes"):
                st.error("La configuration ne contient aucune règle de vote — rechargez le TOML.")
                st.stop()
            if not cfg.get("generative_models"):
                st.error("La configuration ne contient aucun modèle génératif — rechargez le TOML.")
                st.stop()

            # Invalider les caches de résultats de la session précédente.
            # L'onglet Résultats utilisera sim_total_result (construit en mémoire)
            # au lieu de relire les Parquet depuis le disque.
            st.session_state.pop("sim_total_result", None)
            for _k in [k for k in st.session_state if k.startswith("_res_total_")]:
                del st.session_state[_k]
            for _k in [k for k in st.session_state if k.startswith("_scan_struct_")]:
                del st.session_state[_k]

            stop_event = threading.Event()
            log_q: queue.Queue = queue.Queue()
            st.session_state["full_run_stop"] = stop_event
            st.session_state["full_run_log_q"] = log_q
            st.session_state["full_run_running"] = True
            st.session_state["full_run_done"] = False
            st.session_state["full_run_error"] = None
            st.session_state["full_run_logs"] = []
            st.session_state["_full_run_final_rerun_done"] = False  # permet le rerun final
            # Pré-calculer le nombre de combinaisons (model × voters × candidates)
            # pour afficher la barre avant le premier tick du thread.
            _pre_total = max(
                len(cfg.get("generative_models", [])) * len(cfg.get("voters", [])) * len(cfg.get("candidates", [])),
                1,
            )
            st.session_state["full_run_progress"] = (0, _pre_total)
            st.session_state["global_status"] = "Run complet en cours…"

            # Réutiliser le fichier TOML temp maintenu à jour par _refresh_active_toml().
            # En cas d'absence (première exécution sans modification préalable), en créer un.
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

    # Feedback du Run complet
    if st.session_state.get("full_run_running") or st.session_state.get("full_run_done"):
        _render_full_run_feedback()


def _fmt_duration(seconds: float) -> str:
    """Formate une durée en secondes en chaîne lisible (Xm Ys ou Xs)."""
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
    """Feedback inline pour le Run complet.

    Fragment Streamlit (run_every=1 s) : seule cette section se ré-exécute
    chaque seconde, sans relancer les 4 onglets.  Cela libère quasi-totalement
    le GIL au profit du thread de simulation et supprime la compétition CPU
    responsable du ralentissement ×10 observé avec l'ancien time.sleep+rerun.

    Un unique rerun complet est déclenché en fin de run pour mettre à jour
    le statut global, réactiver le bouton, etc.
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

    # ── Barre de progression + timer ──────────────────────────────────────
    current, total = progress_tuple
    start_t: float | None = st.session_state.get("full_run_start_time")
    end_t: float | None = st.session_state.get("full_run_end_time")

    if total > 0:
        frac = min(current / total, 1.0)
        # Construire le label avec timer
        if is_running and start_t is not None:
            elapsed = time.monotonic() - start_t
            elapsed_str = _fmt_duration(elapsed)
            if current > 0:
                eta = elapsed / current * (total - current)
                label = f"Run complet : {current}/{total} — ⏱ {elapsed_str} écoulé · ETA ~{_fmt_duration(eta)}"
            else:
                label = f"Run complet : {current}/{total} — ⏱ {elapsed_str} écoulé"
        elif is_done and start_t is not None and end_t is not None:
            elapsed_str = _fmt_duration(end_t - start_t)
            label = f"Run complet : {current}/{total} — terminé en {elapsed_str}"
        else:
            label = f"Run complet : {current}/{total}"
        st.progress(frac, text=label)
    elif is_running:
        # Barre indéterminée immédiatement après le clic (pas encore de tqdm tick)
        elapsed_str = _fmt_duration(time.monotonic() - start_t) if start_t else ""
        st.progress(0.0, text=f"Run complet en cours… {elapsed_str}")

    if st.session_state["full_run_logs"]:
        with st.expander("Logs du Run complet", expanded=False):
            st.text_area(
                "Logs",
                value="\n".join(st.session_state["full_run_logs"][-200:]),
                height=150,
                disabled=True,
                label_visibility="collapsed",
                key="full_run_log_area",
            )

    if is_done and not is_running:
        if error and error != "Annulé":
            st.error(f"Run complet — Erreur : {error}")
        elif error == "Annulé":
            st.warning("Run complet annulé.")
        else:
            st.success("Run complet terminé — consultez l'onglet Résultats.")

    if is_running:
        t: threading.Thread | None = st.session_state.get("full_run_thread")
        if t is not None and not t.is_alive():
            # Thread terminé mais flag pas encore remis à False — correction défensive.
            st.session_state["full_run_running"] = False

    # ── Rerun complet unique à la fin du run ─────────────────────────────────
    # Déclenché une seule fois quand is_done passe à True pour que le reste de
    # la page (statut global, bouton, onglet Résultats) soit rafraîchi.
    if is_done and not is_running:
        if not st.session_state.get("_full_run_final_rerun_done", False):
            st.session_state["_full_run_final_rerun_done"] = True
            st.rerun()  # rerun complet (hors fragment) — une seule fois


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def _sync_cfg_from_widgets() -> None:
    """Pré-applique les valeurs courantes des widgets à cfg avant le rendu de tab_config.

    Streamlit exécute les onglets dans l'ordre de déclaration.  tab_config (onglet 1)
    calcule l'aperçu TOML AVANT que tab_generation et tab_simulation n'aient mis à jour
    cfg via leurs propres widgets.  Cette fonction lit les valeurs déjà stockées dans
    session_state (disponibles dès le début du script sur chaque rerun) et les applique
    à cfg, de sorte que l'aperçu reflète toujours l'état courant des widgets.
    """
    cfg: dict | None = st.session_state.get("cfg")
    if cfg is None:
        return

    # ── Onglet Données ────────────────────────────────────────────────────
    # output_base_path (tab_config widget, mais appliqué ici aussi pour cohérence)
    if "cfg_output_base_path" in st.session_state:
        cfg["output_base_path"] = st.session_state["cfg_output_base_path"]

    # Seed (number_input → int ou None)
    if "cfg_seed" in st.session_state:
        val = st.session_state["cfg_seed"]
        cfg["seed"] = int(val) if val is not None else None

    # Modèles génératifs
    if "gen_models_select" in st.session_state:
        cfg["generative_models"] = list(st.session_state["gen_models_select"])

    # Voters / Candidates (text_input → parsing)
    if "gen_voters_input" in st.session_state:
        parsed = _parse_int_list(st.session_state["gen_voters_input"])
        if parsed:
            cfg["voters"] = parsed

    if "gen_candidates_input" in st.session_state:
        parsed = _parse_int_list(st.session_state["gen_candidates_input"])
        if parsed:
            cfg["candidates"] = parsed

    # Iterations (slider)
    if "gen_iterations_slider" in st.session_state:
        val = st.session_state["gen_iterations_slider"]
        if isinstance(val, (int, float)) and int(val) > 0:
            cfg["iterations"] = int(val)

    # ── Onglet Simulation — Règles de vote ───────────────────────────────
    # Agréger toutes les familles présentes dans session_state.
    # On ne touche à rule_codes que si au moins une clé de famille existe,
    # pour éviter d'écraser une config chargée via TOML avant le premier rendu.
    rule_keys = [f"rules_family_{fid}" for _, fid in _FAMILIES]
    if any(k in st.session_state for k in rule_keys):
        aggregated: list[str] = []
        for key in rule_keys:
            aggregated.extend(st.session_state.get(key, []))
        cfg["rule_codes"] = aggregated


def _refresh_active_toml() -> None:
    """Maintient un fichier TOML temp à jour reflétant l'état courant de cfg.

    Appelé à chaque rerun, après _sync_cfg_from_widgets().  Calcule un hash du
    contenu TOML sérialisé et ne réécrit sur disque que si cfg a changé depuis
    la dernière écriture (évite des I/O inutiles).

    Met à jour :
    - ``session_state["_active_tmp_toml_path"]`` : chemin absolu du fichier temp
    - ``session_state["toml_active_path"]`` : libellé affiché dans la barre globale
      (ex. "simulation.toml", "simulation.toml (modifié)", "Interface (non sauvegardé)")
    """
    cfg: dict | None = st.session_state.get("cfg")
    if cfg is None:
        return

    # Hash du contenu TOML sérialisé — seule la sérialisation finale compte.
    toml_str = state_to_toml(cfg)
    cfg_hash = hashlib.md5(toml_str.encode()).hexdigest()  # noqa: S324 — non-cryptographic

    if st.session_state.get("_active_tmp_cfg_hash") == cfg_hash:
        # Rien n'a changé : le fichier temp existant est encore valide.
        return

    # Écriture du nouveau fichier temp (resolve output_base_path depuis cfg_base_dir).
    tmp_path = write_temp_toml(cfg, base_dir=st.session_state.get("cfg_base_dir"))

    # Nettoyage de l'ancien fichier temp pour éviter l'accumulation dans /tmp/.
    old_path: str | None = st.session_state.get("_active_tmp_toml_path")
    if old_path and old_path != tmp_path:
        try:
            Path(old_path).unlink(missing_ok=True)
        except OSError:
            pass

    st.session_state["_active_tmp_toml_path"] = tmp_path
    st.session_state["_active_tmp_cfg_hash"] = cfg_hash

    # ── Libellé affiché dans la barre globale ────────────────────────────
    original_name: str | None = st.session_state.get("_original_toml_name")
    original_hash: str | None = st.session_state.get("_original_cfg_hash")

    if original_name:
        # Un fichier a été chargé : signaler si la config a été modifiée depuis.
        if cfg_hash == original_hash:
            label: str | None = original_name
        else:
            label = f"{original_name} (modifié)"
    else:
        # Pas de fichier chargé : config construite entièrement depuis l'interface.
        has_content = bool(cfg.get("generative_models") or cfg.get("rule_codes"))
        label = "Interface (non sauvegardé)" if has_content else None

    st.session_state["toml_active_path"] = label


def main() -> None:
    _init_session()

    # Pré-synchroniser cfg depuis les widgets pour que l'aperçu TOML de tab_config
    # reflète immédiatement chaque modification dans les onglets Données et Simulation.
    _sync_cfg_from_widgets()

    # Maintenir le fichier TOML temp à jour et le libellé "TOML actif".
    _refresh_active_toml()

    # En-tête
    st.title("vote_simulation")
    st.markdown("---")

    # Barre globale
    _render_global_bar()
    st.markdown("---")

    # 4 onglets principaux
    tab_cfg, tab_gen, tab_sim, tab_res = st.tabs(
        [
            "Configuration",
            "Données",
            "Simulation",
            "Résultats",
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

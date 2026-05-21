"""Onglet 4 — Résultats.

Utilise exclusivement les méthodes natives du projet :
  - SimulationSeriesResult : plot_mean_distance_matrix, plot_rules_2d, plot_rules_3d,
                             mean_distance_matrix_frame, metrics_summary_frame
  - SimulationTotalResult  : plot_mean_distance_matrix, plot_metric_heatmap,
                             plot_comparison_grid, plot_rule_pair_heatmap,
                             summary_frame
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Helpers : scan + chargement
# ---------------------------------------------------------------------------

def _scan_sim_result(base_path: str) -> dict[str, dict[str, list[int]]]:
    """Retourne {model: {str(n_v): [n_c, …]}} depuis sim_result/."""
    root = Path(base_path) / "sim_result"
    structure: dict[str, dict[str, list[int]]] = {}
    if not root.is_dir():
        return structure
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            model, rest = subdir.name.split("_v", 1)
            nv_str, nc_str = rest.split("_c", 1)
            n_v, n_c = int(nv_str), int(nc_str)
        except (ValueError, TypeError):
            continue
        if not any(subdir.glob("*.parquet")):
            continue
        structure.setdefault(model, {}).setdefault(str(n_v), [])
        if n_c not in structure[model][str(n_v)]:
            structure[model][str(n_v)].append(n_c)
    return structure


def _load_series(base_path: str, model: str, n_v: int, n_c: int) -> Any:
    """Charge les Parquet d'un dossier comme SimulationSeriesResult."""
    from vote_simulation.models.results.result_config import ResultConfig
    from vote_simulation.models.results.series_result import SimulationSeriesResult
    from vote_simulation.models.results.step_result import SimulationStepResult

    folder = Path(base_path) / "sim_result" / f"{model}_v{n_v}_c{n_c}"
    series = SimulationSeriesResult()
    for fpath in sorted(folder.glob("*.parquet")):
        step = SimulationStepResult(data_source=fpath.stem)
        try:
            step.load_from_file(str(fpath))
            series.add_step(step)
        except Exception:
            pass
    # Config explicite pour que SimulationTotalResult puisse l'indexer
    if series.step_count > 0:
        series.config = ResultConfig.single(
            gen_model=model,
            n_voters=n_v,
            n_candidates=n_c,
            n_iterations=series.step_count,
        )
    return series


def _load_total(
    base_path: str,
    structure: dict[str, dict[str, list[int]]],
    *,
    session_cache: dict | None = None,
) -> Any:
    """Charge toutes les séries disponibles dans un SimulationTotalResult.

    Si *session_cache* est fourni (typiquement ``st.session_state``), chaque
    série chargée est aussi stockée individuellement sous la clé
    ``_series_{base_path}_{model}_{nv_str}_{n_c}`` afin que le rendu de
    l'onglet Résultats n'ait pas à relire les parquets.
    """
    from vote_simulation.models.results.total_result import SimulationTotalResult

    total = SimulationTotalResult()
    for model, voters_map in structure.items():
        for nv_str, cands in voters_map.items():
            for n_c in cands:
                try:
                    s = _load_series(base_path, model, int(nv_str), n_c)
                    if s.step_count > 0:
                        total.add_series(s)
                        if session_cache is not None:
                            series_key = f"_series_{base_path}_{model}_{nv_str}_{n_c}"
                            session_cache[series_key] = s
                except Exception:
                    pass
    return total


def _rebuild_filtered(
    total: Any,
    models: list[str],
    voters: list[int],
    cands: list[int],
) -> Any:
    """Reconstruit un SimulationTotalResult filtré."""
    from vote_simulation.models.results.total_result import SimulationTotalResult

    result = SimulationTotalResult()
    for key, s in total:
        if (
            key.gen_model in models
            and key.n_voters in voters
            and key.n_candidates in cands
        ):
            try:
                result.add_series(s)
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Helpers : affichage
# ---------------------------------------------------------------------------

def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _show_and_export(ax_or_axes: Any, key: str, filename: str) -> None:
    """Affiche la figure dans Streamlit et propose l'export PNG."""
    try:
        if hasattr(ax_or_axes, "get_figure"):
            fig = ax_or_axes.get_figure()
        else:
            flat = [a for a in np.asarray(ax_or_axes, dtype=object).flatten()
                    if hasattr(a, "get_figure")]
            fig = flat[0].get_figure() if flat else plt.gcf()
    except Exception:
        fig = plt.gcf()

    st.pyplot(fig, use_container_width=True)
    st.download_button(
        "Exporter PNG",
        data=_fig_to_bytes(fig),
        file_name=filename,
        mime="image/png",
        key=key,
    )
    plt.close(fig)


def _third_param_filter(total_f: Any, row_p: str, col_p: str, key_prefix: str) -> Any:
    """Filtre le total sur le 3ème paramètre si nécessaire pour plot_metric_heatmap."""
    PARAMS = ["gen_model", "n_voters", "n_candidates"]
    third_p = next(p for p in PARAMS if p not in (row_p, col_p))
    vals_map: dict[str, list] = {
        "gen_model": total_f.gen_models,
        "n_voters": total_f.voter_counts,
        "n_candidates": total_f.candidate_counts,
    }
    third_vals = vals_map[third_p]
    if len(third_vals) > 1:
        sel = st.selectbox(
            f"Fixer `{third_p}` (requis — 2 axes déjà pris)",
            third_vals,
            key=f"{key_prefix}_third",
        )
        return total_f.filter(**{third_p: sel})
    return total_f


# ---------------------------------------------------------------------------
# Rendu principal
# ---------------------------------------------------------------------------

def render_tab_results() -> None:
    st.header("Résultats")
    st.caption(
        "Exploration via les méthodes natives "
        "`SimulationSeriesResult` et `SimulationTotalResult`."
    )

    # Pendant un Run complet, on évite de lire les 18 000 parquets existants :
    # _load_total() bloquerait le thread principal 20-60 s et figerait la barre.
    if st.session_state.get("full_run_running"):
        st.info(
            "⏳ Run complet en cours — les résultats seront disponibles à la fin."
        )
        return

    cfg: dict = st.session_state["cfg"]
    base_path = str(Path(cfg.get("output_base_path", "../data/")).resolve())

    # Cache du scan filesystem : évite de reglobler les dossiers à chaque rerun.
    # La clé est effacée par le handler du bouton Run complet, donc après
    # chaque simulation, le premier affichage re-scanne pour inclure les
    # nouveaux résultats.
    _scan_key = f"_scan_struct_{base_path}"
    if _scan_key not in st.session_state:
        st.session_state[_scan_key] = _scan_sim_result(base_path)
    structure = st.session_state[_scan_key]

    if not structure:
        st.info(
            f"Aucun résultat dans `{base_path}/sim_result/`.\n\n"
            "Lancez une simulation dans l'onglet **Simulation**."
        )
        return

    # ── Cache du total dans la session ──────────────────────────────────────
    cache_key = f"_res_total_{base_path}"

    # Priorité 1 : résultat en mémoire construit pendant la simulation courante.
    # Évite de lire tous les Parquet depuis le disque après un Run complet.
    if cache_key not in st.session_state and "sim_total_result" in st.session_state:
        st.session_state[cache_key] = st.session_state["sim_total_result"]

    # Priorité 2 : chargement depuis le disque (premier affichage ou données existantes).
    if cache_key not in st.session_state:
        with st.spinner("Chargement initial de tous les résultats…"):
            st.session_state[cache_key] = _load_total(base_path, structure)

    total: Any = st.session_state[cache_key]

    tab_serie, tab_global = st.tabs(["Analyse d'une série", "Vue globale"])

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet A — Analyse d'une série
    # ══════════════════════════════════════════════════════════════════════════
    with tab_serie:
        c1, c2, c3 = st.columns(3)
        sel_model = c1.selectbox("Modèle", sorted(structure.keys()), key="res_model")
        voters_opts = sorted(structure.get(sel_model, {}).keys(), key=int)
        sel_voters = c2.selectbox("Votants", voters_opts, key="res_voters")
        cands_opts = sorted(structure.get(sel_model, {}).get(sel_voters, []))
        sel_cands = c3.selectbox("Candidats", cands_opts, key="res_cands")

        series_key = f"_series_{base_path}_{sel_model}_{sel_voters}_{sel_cands}"
        if series_key not in st.session_state:
            with st.spinner("Chargement de la série…"):
                st.session_state[series_key] = _load_series(
                    base_path, sel_model, int(sel_voters), int(sel_cands)
                )
        series = st.session_state[series_key]

        if series.step_count == 0:
            st.warning("Aucune donnée valide pour cette sélection.")
            return

        rules = series.mean_distance_matrix_frame.columns.tolist()
        tag = f"{sel_model}_v{sel_voters}_c{sel_cands}"
        st.success(
            f"**{series.step_count} itérations** · **{len(rules)} règles** : {', '.join(rules)}"
        )

        s_tabs = st.tabs([
            "Matrice de distance",
            "Projection 2D",
            "Projection 3D",
            "Résumé",
        ])

        # ── Matrice de distance ─────────────────────────────────────────────
        with s_tabs[0]:
            st.caption(
                "`series.plot_mean_distance_matrix()` — distance de Jaccard moyenne, "
                "0 = toujours d'accord · 100 = jamais"
            )
            try:
                _show_and_export(
                    series.plot_mean_distance_matrix(show=False),
                    f"dl_dist_{tag}", f"dist_{tag}.png",
                )
            except Exception as exc:
                st.error(str(exc))

        # ── Projection 2D ──────────────────────────────────────────────────
        with s_tabs[1]:
            if len(rules) < 3:
                st.info("Au moins 3 règles nécessaires pour la projection 2D.")
            else:
                st.caption("`series.plot_rules_2d()` — distances projetées par MDS")
                try:
                    _show_and_export(
                        series.plot_rules_2d(show=False),
                        f"dl_2d_{tag}", f"mds2d_{tag}.png",
                    )
                except Exception as exc:
                    st.error(str(exc))

        # ── Projection 3D ──────────────────────────────────────────────────
        with s_tabs[2]:
            if len(rules) < 4:
                st.info("Au moins 4 règles nécessaires pour la projection 3D.")
            else:
                st.caption("`series.plot_rules_3d()` — distances projetées par MDS 3D")
                try:
                    _show_and_export(
                        series.plot_rules_3d(show=False),
                        f"dl_3d_{tag}", f"mds3d_{tag}.png",
                    )
                except Exception as exc:
                    st.error(str(exc))

        # ── Résumé ─────────────────────────────────────────────────────────
        with s_tabs[3]:
            ca, cb = st.columns(2)
            ca.metric("Distance moyenne", f"{series.mean_distance:.2f} %")
            ra, rb, d = series.most_distant_rules
            if ra:
                cb.metric("Paire la + distante", f"{ra} ↔ {rb}", f"{d:.2f} %")

            st.markdown("**Matrice de distance** (`mean_distance_matrix_frame`)")
            st.dataframe(
                series.mean_distance_matrix_frame
                    .style.background_gradient(cmap="Reds", vmin=0, vmax=100)
                    .format("{:.1f}"),
                use_container_width=True,
            )

            msf = series.metrics_summary_frame
            if not msf.empty:
                st.markdown("**Métriques des gagnants** (`metrics_summary_frame`)")
                st.dataframe(msf.style.format("{:.4f}"), use_container_width=True)
            else:
                st.info(
                    "Les métriques des gagnants ne sont pas disponibles depuis le disque "
                    "(elles sont calculées en temps réel lors de la simulation)."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet B — Vue globale
    # ══════════════════════════════════════════════════════════════════════════
    with tab_global:
        if total.series_count == 0:
            st.info("Aucune série chargée.")
            return

        col_info, col_btn = st.columns([4, 1])
        col_info.caption(
            f"{total.series_count} série(s) · modèles : {', '.join(total.gen_models)} · "
            f"votants : {total.voter_counts} · candidats : {total.candidate_counts}"
        )
        if col_btn.button("Recharger", key="btn_reload_total"):
            del st.session_state[cache_key]
            st.rerun()

        # Filtres
        with st.expander("Filtres", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            f_models = fc1.multiselect(
                "Modèles", total.gen_models, default=total.gen_models, key="gf_m"
            )
            f_voters = fc2.multiselect(
                "Votants", total.voter_counts, default=total.voter_counts, key="gf_v"
            )
            f_cands = fc3.multiselect(
                "Candidats", total.candidate_counts, default=total.candidate_counts, key="gf_c"
            )

        total_f = _rebuild_filtered(
            total,
            f_models or total.gen_models,
            f_voters or total.voter_counts,
            f_cands or total.candidate_counts,
        )

        if total_f.series_count == 0:
            st.warning("Aucune série ne correspond aux filtres.")
            return

        PARAMS = ["gen_model", "n_voters", "n_candidates"]

        g_tabs = st.tabs([
            "Résumé",
            "Matrice de distance",
            "Heatmap métrique",
            "Grille de comparaison",
            "Paire de règles",
        ])

        # ── Résumé ──────────────────────────────────────────────────────────
        with g_tabs[0]:
            st.caption("`total.summary_frame()` — une ligne par série")
            st.dataframe(total_f.summary_frame(), use_container_width=True)

        # ── Matrice de distance globale ──────────────────────────────────────
        with g_tabs[1]:
            st.caption(
                "`total.plot_mean_distance_matrix()` — distance moyenne "
                "inter-règles sur toutes les séries filtrées"
            )
            try:
                _show_and_export(
                    total_f.plot_mean_distance_matrix(show=False),
                    "dl_g_dist", "global_distance_matrix.png",
                )
            except Exception as exc:
                st.error(str(exc))

        # ── Heatmap métrique ─────────────────────────────────────────────────
        with g_tabs[2]:
            st.caption(
                "`total.plot_metric_heatmap()` — distance moyenne croisée "
                "sur 2 paramètres (le 3ème doit être fixé)"
            )
            hc1, hc2 = st.columns(2)
            row_p = hc1.selectbox("Axe lignes", PARAMS, index=0, key="hm_row")
            col_p = hc2.selectbox(
                "Axe colonnes", [p for p in PARAMS if p != row_p], index=0, key="hm_col"
            )
            total_for_hm = _third_param_filter(total_f, row_p, col_p, "hm")
            try:
                _show_and_export(
                    total_for_hm.plot_metric_heatmap(
                        row_param=row_p, col_param=col_p,
                        metric="mean_distance", show=False,
                    ),
                    "dl_g_hm", "metric_heatmap.png",
                )
            except Exception as exc:
                st.error(str(exc))

        # ── Grille de comparaison ────────────────────────────────────────────
        with g_tabs[3]:
            st.caption(
                "`total.plot_comparison_grid()` — matrices de distance côte à côte, "
                "une par valeur du paramètre choisi"
            )
            vary_p = st.selectbox(
                "Paramètre à faire varier", PARAMS, index=2, key="grid_vary"
            )
            try:
                _show_and_export(
                    total_f.plot_comparison_grid(vary_param=vary_p, show=False),
                    "dl_g_grid", "comparison_grid.png",
                )
            except Exception as exc:
                st.error(str(exc))

        # ── Paire de règles ──────────────────────────────────────────────────
        with g_tabs[4]:
            st.caption(
                "`total.plot_rule_pair_heatmap()` — distance Jaccard entre deux règles "
                "croisée sur 2 paramètres"
            )
            avail_rules: list[str] = []
            for _k, s in total_f:
                for r in s.mean_distance_matrix_frame.columns.tolist():
                    if r not in avail_rules:
                        avail_rules.append(r)
            avail_rules = sorted(avail_rules)

            if len(avail_rules) < 2:
                st.info("Au moins 2 règles nécessaires.")
            else:
                pc1, pc2 = st.columns(2)
                rule_a = pc1.selectbox("Règle A", avail_rules, index=0, key="pair_a")
                rule_b_opts = [r for r in avail_rules if r != rule_a]
                rule_b = pc2.selectbox("Règle B", rule_b_opts, index=0, key="pair_b")

                pr1, pr2 = st.columns(2)
                rp_pair = pr1.selectbox("Axe lignes", PARAMS, index=0, key="pair_row")
                cp_pair = pr2.selectbox(
                    "Axe colonnes", [p for p in PARAMS if p != rp_pair],
                    index=0, key="pair_col",
                )
                try:
                    _show_and_export(
                        total_f.plot_rule_pair_heatmap(
                            rule_a, rule_b,
                            row_param=rp_pair, col_param=cp_pair,
                            show=False,
                        ),
                        "dl_g_pair", f"pair_{rule_a}_{rule_b}.png",
                    )
                except Exception as exc:
                    st.error(str(exc))


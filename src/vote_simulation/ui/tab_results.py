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
    session_cache: Any | None = None,
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


def _apply_filter(
    total: Any,
    models: list[str],
    voters: list[int],
    cands: list[int],
) -> Any:
    """Filtre via copie superficielle de _entries — O(n) références, zéro recopie de données."""
    from vote_simulation.models.results.total_result import SimulationTotalResult

    models_s = set(models)
    voters_s = set(voters)
    cands_s = set(cands)
    result = SimulationTotalResult()
    for key, s in total:
        if key.gen_model in models_s and key.n_voters in voters_s and key.n_candidates in cands_s:
            result._entries[key] = s
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
            flat = [a for a in np.asarray(ax_or_axes, dtype=object).flatten() if hasattr(a, "get_figure")]
            fig = flat[0].get_figure() if flat else plt.gcf()
    except Exception:
        fig = plt.gcf()

    png = _fig_to_bytes(fig)
    plt.close(fig)
    _img_col, _ = st.columns([1, 1])
    _img_col.image(png, width="stretch")
    st.download_button(
        "Exporter PNG",
        data=png,
        file_name=filename,
        mime="image/png",
        key=key,
    )


def _cached_plot(
    cache_key: str,
    plot_fn,
    export_key: str,
    filename: str,
) -> None:
    """Exécute plot_fn() une seule fois et met le PNG en cache session_state."""
    if cache_key not in st.session_state:
        try:
            ax_or_axes = plot_fn()
            try:
                if hasattr(ax_or_axes, "get_figure"):
                    fig = ax_or_axes.get_figure()
                else:
                    flat = [a for a in np.asarray(ax_or_axes, dtype=object).flatten() if hasattr(a, "get_figure")]
                    fig = flat[0].get_figure() if flat else plt.gcf()
            except Exception:
                fig = plt.gcf()
            png = _fig_to_bytes(fig)
            plt.close(fig)
            st.session_state[cache_key] = png
        except Exception as exc:
            st.session_state[cache_key] = exc

    val = st.session_state[cache_key]
    if isinstance(val, Exception):
        # ValueError : données manquantes (ex. métriques non disponibles)
        if isinstance(val, ValueError):
            st.info(
                f"{val}\n\nLancez un **Run complet** (avec *compute\\_metrics=True*) "
                "pour rendre ces données disponibles."
            )
        else:
            st.error(str(val))
    else:
        _img_col, _ = st.columns([1, 1])
        _img_col.image(val, width="stretch")
        st.download_button(
            "Exporter PNG",
            data=val,
            file_name=filename,
            mime="image/png",
            key=export_key,
        )


def _df_csv_export(df: Any, key: str, filename: str) -> None:
    """Bouton de téléchargement CSV pour un DataFrame (ou Styler)."""
    if hasattr(df, "data"):  # pandas Styler → DataFrame sous-jacent
        df = df.data
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Exporter CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def _build_global_dist_df(total_f: Any) -> Any:
    """Construit la matrice de distance moyenne inter-règles pour un SimulationTotalResult."""
    import pandas as pd

    all_rules: list[str] = []
    rule_index: dict[str, int] = {}
    for s in total_f._entries.values():
        for r in s._rule_order:
            if r not in rule_index:
                rule_index[r] = len(all_rules)
                all_rules.append(r)
    if not all_rules:
        return pd.DataFrame()
    n = len(all_rules)
    acc = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)
    for s in total_f._entries.values():
        mat = s.mean_distance_matrix
        if mat.size == 0:
            continue
        perm = np.array([rule_index[r] for r in s._rule_order], dtype=np.intp)
        ix = np.ix_(perm, perm)
        acc[ix] += mat.astype(np.float64)
        counts[ix] += 1
    with np.errstate(invalid="ignore"):
        avg = np.where(counts > 0, acc / counts, np.nan)
    return pd.DataFrame(avg, index=np.array(all_rules), columns=np.array(all_rules))


def _build_global_metrics_df(total_f: Any, metrics: list[str] | None = None, stat: str = "mean") -> Any:
    """Construit la matrice (règles × métriques) moyenne pour un SimulationTotalResult."""
    import pandas as pd

    from vote_simulation.models.rules.winner_metrics import METRIC_FIELDS

    fields = list(metrics) if metrics is not None else list(METRIC_FIELDS)
    all_rules: list[str] = []
    for s in total_f._entries.values():
        frame = s.metrics_summary_frame
        if not frame.empty:
            for r in frame.index:
                if r not in all_rules:
                    all_rules.append(r)
    if not all_rules:
        return pd.DataFrame()
    n_r, n_m = len(all_rules), len(fields)
    acc = np.zeros((n_r, n_m), dtype=np.float64)
    cnt = np.zeros((n_r, n_m), dtype=np.int64)
    for s in total_f._entries.values():
        frame = s.metrics_summary_frame
        if frame.empty:
            continue
        for ri, rule in enumerate(all_rules):
            if rule not in frame.index:
                continue
            for mi, field in enumerate(fields):
                col = f"{field}_{stat}"
                if col in frame.columns:
                    v = float(frame.loc[rule, col])
                    if not np.isnan(v):
                        acc[ri, mi] += v
                        cnt[ri, mi] += 1
    with np.errstate(invalid="ignore"):
        avg = np.where(cnt > 0, acc / cnt, np.nan)
    return pd.DataFrame(avg, index=np.array(all_rules), columns=np.array(fields))


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
    st.caption("Exploration via les méthodes natives `SimulationSeriesResult` et `SimulationTotalResult`.")

    # Pendant un Run complet, on évite de lire les 18 000 parquets existants :
    # _load_total() bloquerait le thread principal 20-60 s et figerait la barre.
    if st.session_state.get("full_run_running"):
        st.info("⏳ Run complet en cours — les résultats seront disponibles à la fin.")
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
            f"Aucun résultat dans `{base_path}/sim_result/`.\n\nLancez une simulation dans l'onglet **Simulation**."
        )
        return

    # ── Cache du total dans la session ──────────────────────────────────────
    cache_key = f"_res_total_{base_path}"

    # Priorité 1 : résultat en mémoire construit pendant la simulation courante.
    # Évite de lire tous les Parquet depuis le disque après un Run complet.
    if cache_key not in st.session_state and "sim_total_result" in st.session_state:
        st.session_state[cache_key] = st.session_state["sim_total_result"]

    # Priorité 2 : chargement depuis le disque (premier affichage ou données existantes).
    # session_cache=st.session_state : chaque série chargée est aussi mise en cache
    # individuellement → zéro double-lecture parquet dans l'onglet "Analyse d'une série".
    if cache_key not in st.session_state:
        with st.spinner("Chargement initial de tous les résultats…"):
            st.session_state[cache_key] = _load_total(base_path, structure, session_cache=st.session_state)

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
                st.session_state[series_key] = _load_series(base_path, sel_model, int(sel_voters), int(sel_cands))
        series = st.session_state[series_key]

        if series.step_count == 0:
            st.warning("Aucune donnée valide pour cette sélection.")
            return

        rules = series.mean_distance_matrix_frame.columns.tolist()
        tag = f"{sel_model}_v{sel_voters}_c{sel_cands}"
        st.success(f"**{series.step_count} itérations** · **{len(rules)} règles** : {', '.join(rules)}")

        s_tabs = st.tabs(
            [
                "Matrice de distance",
                "Projection 2D",
                "Projection 3D",
                "Métriques",
                "Résumé",
            ]
        )

        # ── Matrice de distance ─────────────────────────────────────────────
        with s_tabs[0]:
            st.caption(
                "`series.plot_mean_distance_matrix()` — distance de Jaccard moyenne, "
                "0 = toujours d'accord · 100 = jamais"
            )
            _cached_plot(
                f"_plt_dist_{tag}",
                lambda: series.plot_mean_distance_matrix(show=False),
                f"dl_dist_{tag}",
                f"dist_{tag}.png",
            )

        # ── Projection 2D ──────────────────────────────────────────────────
        with s_tabs[1]:
            if len(rules) < 3:
                st.info("Au moins 3 règles nécessaires pour la projection 2D.")
            else:
                st.caption("`series.plot_rules_2d()` — distances projetées par MDS")
                _cached_plot(
                    f"_plt_2d_{tag}",
                    lambda: series.plot_rules_2d(show=False),
                    f"dl_2d_{tag}",
                    f"mds2d_{tag}.png",
                )

        # ── Projection 3D ──────────────────────────────────────────────────
        with s_tabs[2]:
            if len(rules) < 4:
                st.info("Au moins 4 règles nécessaires pour la projection 3D.")
            else:
                st.caption("`series.plot_rules_3d()` — distances projetées par MDS 3D")
                _cached_plot(
                    f"_plt_3d_{tag}",
                    lambda: series.plot_rules_3d(show=False),
                    f"dl_3d_{tag}",
                    f"mds3d_{tag}.png",
                )

        # ── Métriques des gagnants ──────────────────────────────────────────
        with s_tabs[3]:
            from vote_simulation.models.results.total_result import SimulationTotalResult as _STR
            from vote_simulation.models.rules.winner_metrics import METRIC_FIELDS as _MF

            msf = series.metrics_summary_frame
            if msf.empty:
                st.info(
                    "Les métriques des gagnants ne sont pas disponibles pour cette série.\n\n"
                    "Elles sont calculées lors d'un **Run complet** "
                    "(onglet Configuration → bouton **Run complet**)."
                )
            else:
                # Enveloppe la série dans un SimulationTotalResult pour utiliser
                # plot_metrics_rules_matrix
                _total_one_key = f"_total_one_{tag}"
                if _total_one_key not in st.session_state:
                    _t1 = _STR()
                    _t1.add_series(series)
                    st.session_state[_total_one_key] = _t1
                _total_one = st.session_state[_total_one_key]

                avail_mf = [f for f in _MF if any(c.startswith(f + "_") for c in msf.columns)]
                mc1, mc2 = st.columns([3, 1])
                sel_mf_sm = mc1.multiselect(
                    "Métriques à afficher",
                    options=avail_mf,
                    default=avail_mf,
                    key=f"sm_mf_{tag}",
                )
                stat_sm = mc2.radio("Statistique", ["mean", "std"], horizontal=True, key=f"sm_stat_{tag}")
                if not sel_mf_sm:
                    st.info("Sélectionnez au moins une métrique.")
                else:
                    _sm_key = f"_plt_sm_{tag}_{sorted(sel_mf_sm)}_{stat_sm}"
                    _cached_plot(
                        _sm_key,
                        lambda _m=sel_mf_sm, _s=stat_sm: _total_one.plot_metrics_rules_matrix(
                            metrics=_m, stat=_s, show=False
                        ),
                        f"dl_sm_{tag}",
                        f"metrics_{tag}.png",
                    )

                    st.dataframe(
                        msf.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                        width="stretch",
                    )
                    _df_csv_export(msf, f"dl_csv_sm_{tag}", f"metrics_{tag}.csv")

        # ── Résumé ─────────────────────────────────────────────────────────
        with s_tabs[4]:
            ca, cb = st.columns(2)
            ca.metric("Distance moyenne", f"{series.mean_distance:.2f} %")
            ra, rb, d = series.most_distant_rules
            if ra:
                cb.metric("Paire la + distante", f"{ra} ↔ {rb}", f"{d:.2f} %")

            st.markdown("**Matrice de distance** (`mean_distance_matrix_frame`)")
            st.dataframe(
                series.mean_distance_matrix_frame.style.background_gradient(cmap="Reds", vmin=0, vmax=100).format(
                    "{:.1f}"
                ),
                width="stretch",
            )
            _df_csv_export(
                series.mean_distance_matrix_frame,
                f"dl_csv_dist_{tag}",
                f"distance_matrix_{tag}.csv",
            )

            msf = series.metrics_summary_frame
            if not msf.empty:
                st.markdown("**Métriques des gagnants** (`metrics_summary_frame`)")
                st.dataframe(msf.style.format("{:.4f}"), width="stretch")
                _df_csv_export(msf, f"dl_csv_msf_{tag}", f"metrics_{tag}.csv")
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
            # Supprimer le total, les filtres appliqu\u00e9s et tous les caches de plots
            for _k in list(st.session_state.keys()):
                if (
                    _k in (cache_key, "gf_m_applied", "gf_v_applied", "gf_c_applied")
                    or (isinstance(_k, str) and _k.startswith("_filtered_"))
                    or (isinstance(_k, str) and _k.startswith("_plt_g_"))
                    or (isinstance(_k, str) and _k.startswith("_avail_rules_"))
                ):
                    del st.session_state[_k]
            st.rerun()

        # Filtres avec bouton Appliquer pour éviter toute recomputation à chaque widget
        _fk_models = "gf_m_applied"
        _fk_voters = "gf_v_applied"
        _fk_cands = "gf_c_applied"
        # Valeurs appliquées courantes (defaults : tout sélectionné)
        applied_models = st.session_state.get(_fk_models, list(total.gen_models))
        applied_voters = st.session_state.get(_fk_voters, list(total.voter_counts))
        applied_cands = st.session_state.get(_fk_cands, list(total.candidate_counts))

        with st.expander("Filtres", expanded=True):
            fc1, fc2, fc3, fc4 = st.columns([3, 3, 3, 1])
            f_models = fc1.multiselect("Modèles", total.gen_models, default=applied_models, key="gf_m")
            f_voters = fc2.multiselect("Votants", total.voter_counts, default=applied_voters, key="gf_v")
            f_cands = fc3.multiselect("Candidats", total.candidate_counts, default=applied_cands, key="gf_c")
            if fc4.button("Appliquer", key="btn_apply_filters", width="stretch"):
                new_models = f_models or list(total.gen_models)
                new_voters = f_voters or list(total.voter_counts)
                new_cands = f_cands or list(total.candidate_counts)
                # Invalider le cache filtré si les filtres changent
                if new_models != applied_models or new_voters != applied_voters or new_cands != applied_cands:
                    # Supprimer tous les caches de plots associés
                    old_fk = (
                        f"_filtered_{cache_key}_"
                        f"{sorted(applied_models)}_{sorted(applied_voters)}_{sorted(applied_cands)}"
                    )
                    for k in list(st.session_state.keys()):
                        if (isinstance(k, str) and k.startswith("_plt_g_")) or k == old_fk:
                            del st.session_state[k]
                st.session_state[_fk_models] = new_models
                st.session_state[_fk_voters] = new_voters
                st.session_state[_fk_cands] = new_cands
                applied_models, applied_voters, applied_cands = new_models, new_voters, new_cands

        # Cache de la vue filtrée (shallow copy — zéro copie de données)
        _filter_cache_key = (
            f"_filtered_{cache_key}_{sorted(applied_models)}_{sorted(applied_voters)}_{sorted(applied_cands)}"
        )
        if _filter_cache_key not in st.session_state:
            st.session_state[_filter_cache_key] = _apply_filter(total, applied_models, applied_voters, applied_cands)
        total_f = st.session_state[_filter_cache_key]

        if total_f.series_count == 0:
            st.warning("Aucune série ne correspond aux filtres.")
            return

        PARAMS = ["gen_model", "n_voters", "n_candidates"]

        g_tabs = st.tabs(
            [
                "Résumé",
                "Matrice de distance",
                "Métriques des gagnants",
                "Heatmap métrique",
                "Grille de comparaison",
                "Paire de règles",
            ]
        )

        # ── Résumé ──────────────────────────────────────────────────────────
        with g_tabs[0]:
            st.caption("`total.summary_frame()` — une ligne par série")
            _summary_df = total_f.summary_frame()
            st.dataframe(_summary_df, width="stretch")
            _df_csv_export(_summary_df, "dl_csv_g_summary", "summary_global.csv")

        # ── Matrice de distance globale ──────────────────────────────────────
        with g_tabs[1]:
            st.caption(
                "`total.plot_mean_distance_matrix()` — distance moyenne inter-règles sur toutes les séries filtrées"
            )
            _cached_plot(
                f"_plt_g_dist_{_filter_cache_key}",
                lambda: total_f.plot_mean_distance_matrix(show=False),
                "dl_g_dist",
                "global_distance_matrix.png",
            )
            _g_dist_df_key = f"_g_dist_df_{_filter_cache_key}"
            if _g_dist_df_key not in st.session_state:
                st.session_state[_g_dist_df_key] = _build_global_dist_df(total_f)
            _g_dist_df = st.session_state[_g_dist_df_key]
            if not _g_dist_df.empty:
                st.markdown("**Matrice de distance moyenne** (%)")
                st.dataframe(
                    _g_dist_df.style.background_gradient(cmap="Reds", vmin=0, vmax=100).format("{:.1f}"),
                    width="stretch",
                )
                _df_csv_export(_g_dist_df, "dl_csv_g_dist", "global_distance_matrix.csv")

        # ── Métriques des gagnants ───────────────────────────────────────────
        with g_tabs[2]:
            st.caption(
                "`total.plot_metrics_rules_matrix()` — valeur agrégée de chaque métrique "
                "par règle, normalisée par ligne pour comparer les règles entre elles"
            )
            from vote_simulation.models.rules.winner_metrics import METRIC_FIELDS as _GMF

            gmc1, gmc2 = st.columns([3, 1])
            sel_gm = gmc1.multiselect(
                "Métriques à afficher",
                options=list(_GMF),
                default=list(_GMF),
                key="gm_metrics",
            )
            stat_gm = gmc2.radio("Statistique", ["mean", "std"], horizontal=True, key="gm_stat")
            if not sel_gm:
                st.info("Sélectionnez au moins une métrique.")
            else:
                _cached_plot(
                    f"_plt_g_gm_{_filter_cache_key}_{sorted(sel_gm)}_{stat_gm}",
                    lambda _m=sel_gm, _s=stat_gm: total_f.plot_metrics_rules_matrix(metrics=_m, stat=_s, show=False),
                    "dl_gm_matrix",
                    "metrics_rules_matrix.png",
                )
                _g_met_df_key = f"_g_met_df_{_filter_cache_key}_{sorted(sel_gm)}_{stat_gm}"
                if _g_met_df_key not in st.session_state:
                    st.session_state[_g_met_df_key] = _build_global_metrics_df(total_f, metrics=sel_gm, stat=stat_gm)
                _g_met_df = st.session_state[_g_met_df_key]
                if not _g_met_df.empty:
                    st.markdown("**Données brutes** (règles × métriques)")
                    st.dataframe(
                        _g_met_df.style.background_gradient(cmap="Blues", axis=0).format("{:.4f}"),
                        width="stretch",
                    )
                    _df_csv_export(
                        _g_met_df,
                        f"dl_csv_g_met_{sorted(sel_gm)}_{stat_gm}",
                        f"metrics_rules_matrix_{stat_gm}.csv",
                    )

        # ── Heatmap métrique ─────────────────────────────────────────────────
        with g_tabs[3]:
            st.caption(
                "`total.plot_metric_heatmap()` — distance moyenne croisée sur 2 paramètres (le 3ème doit être fixé)"
            )
            hc1, hc2 = st.columns(2)
            row_p = hc1.selectbox("Axe lignes", PARAMS, index=0, key="hm_row")
            col_p = hc2.selectbox("Axe colonnes", [p for p in PARAMS if p != row_p], index=0, key="hm_col")
            total_for_hm = _third_param_filter(total_f, row_p, col_p, "hm")
            # La valeur du 3ème param est déjà encodée dans total_for_hm._entries
            # mais on l'encode aussi dans la clé via sa taille (suffisant pour invalider)
            _hm_third_tag = len(total_for_hm._entries)
            _cached_plot(
                f"_plt_g_hm_{_filter_cache_key}_{row_p}_{col_p}_{_hm_third_tag}",
                lambda _rp=row_p, _cp=col_p: total_for_hm.plot_metric_heatmap(
                    row_param=_rp, col_param=_cp, metric="mean_distance", show=False
                ),
                "dl_g_hm",
                "metric_heatmap.png",
            )

        # ── Grille de comparaison ────────────────────────────────────────────
        with g_tabs[4]:
            st.caption(
                "`total.plot_comparison_grid()` — matrices de distance côte à côte, une par valeur du paramètre choisi"
            )
            vary_p = st.selectbox("Paramètre à faire varier", PARAMS, index=2, key="grid_vary")
            _cached_plot(
                f"_plt_g_grid_{_filter_cache_key}_{vary_p}",
                lambda _vp=vary_p: total_f.plot_comparison_grid(vary_param=_vp, show=False),
                "dl_g_grid",
                "comparison_grid.png",
            )

        # ── Paire de règles ──────────────────────────────────────────────────
        with g_tabs[5]:
            st.caption("`total.plot_rule_pair_heatmap()` — distance Jaccard entre deux règles croisée sur 2 paramètres")
            # Cache avail_rules pour éviter d'itérer total_f à chaque rerun
            _rules_cache_key = f"_avail_rules_{_filter_cache_key}"
            if _rules_cache_key not in st.session_state:
                _rules: list[str] = []
                for _k, _s in total_f:
                    for _r in _s.mean_distance_matrix_frame.columns.tolist():
                        if _r not in _rules:
                            _rules.append(_r)
                st.session_state[_rules_cache_key] = sorted(_rules)
            avail_rules: list[str] = st.session_state[_rules_cache_key]

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
                    "Axe colonnes",
                    [p for p in PARAMS if p != rp_pair],
                    index=0,
                    key="pair_col",
                )
                _cached_plot(
                    f"_plt_g_pair_{_filter_cache_key}_{rule_a}_{rule_b}_{rp_pair}_{cp_pair}",
                    lambda _ra=rule_a, _rb=rule_b, _rp=rp_pair, _cp=cp_pair: total_f.plot_rule_pair_heatmap(
                        _ra, _rb, row_param=_rp, col_param=_cp, show=False
                    ),
                    "dl_g_pair",
                    f"pair_{rule_a}_{rule_b}.png",
                )

"""Tab 4 — Results.

Uses exclusively the project's native methods:
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
# Helpers: scan + loading
# ---------------------------------------------------------------------------


def _scan_sim_result(base_path: str) -> dict[str, dict[str, list[int]]]:
    """Returns {model: {str(n_v): [n_c, …]}} from sim_result/ and results/.

    - sim_result/{MODEL}_v{NV}_c{NC}/*.parquet  (legacy per-iteration format)
    - results/{MODEL}_v{NV}_c{NC}_i*.parquet    (series cache written by simulation_instance)
    """
    structure: dict[str, dict[str, list[int]]] = {}

    # ── sim_result/ (legacy per-iteration files) ─────────────────────────
    sim_root = Path(base_path) / "sim_result"
    if sim_root.is_dir():
        for subdir in sorted(sim_root.iterdir()):
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

    # ── results/ (series cache: {MODEL}_v{NV}_c{NC}_i{N}.parquet) ────────
    res_root = Path(base_path) / "results"
    if res_root.is_dir():
        import re

        _pat = re.compile(r"^([A-Za-z0-9_]+)_v(\d+)_c(\d+)_i\d+\.parquet$")
        for fpath in sorted(res_root.glob("*.parquet")):
            m = _pat.match(fpath.name)
            if not m:
                continue
            model, nv_str, nc_str = m.group(1), m.group(2), m.group(3)
            n_v, n_c = int(nv_str), int(nc_str)
            structure.setdefault(model, {}).setdefault(str(n_v), [])
            if n_c not in structure[model][str(n_v)]:
                structure[model][str(n_v)].append(n_c)

    return structure


def _load_series(base_path: str, model: str, n_v: int, n_c: int) -> Any:
    """Loads Parquet files from a folder or cache file as a SimulationSeriesResult.

    Tries in order:
    1. sim_result/{model}_v{n_v}_c{n_c}/*.parquet  (per-iteration files)
    2. results/{model}_v{n_v}_c{n_c}_i*.parquet    (full-series cache file)
    """
    from vote_simulation.models.results.result_config import ResultConfig
    from vote_simulation.models.results.series_result import SimulationSeriesResult
    from vote_simulation.models.results.step_result import SimulationStepResult

    # ── 1. sim_result/ per-iteration format ──────────────────────────────
    folder = Path(base_path) / "sim_result" / f"{model}_v{n_v}_c{n_c}"
    series = SimulationSeriesResult()
    if folder.is_dir():
        for fpath in sorted(folder.glob("*.parquet")):
            step = SimulationStepResult(data_source=fpath.stem)
            try:
                step.load_from_file(str(fpath))
                series.add_step(step)
            except Exception:
                pass
        if series.step_count > 0:
            series.config = ResultConfig.single(
                gen_model=model,
                n_voters=n_v,
                n_candidates=n_c,
                n_iterations=series.step_count,
            )
            return series

    # ── 2. results/ series-cache format ──────────────────────────────────
    res_root = Path(base_path) / "results"
    candidates = sorted(res_root.glob(f"{model}_v{n_v}_c{n_c}_i*.parquet"))
    if candidates:
        # Use the most recent (largest n_iterations) cache file.
        cache_file = candidates[-1]
        series = SimulationSeriesResult()
        series.load_from_file(str(cache_file))
        if series.step_count > 0 and not series.config.gen_models:
            series.config = ResultConfig.single(
                gen_model=model,
                n_voters=n_v,
                n_candidates=n_c,
                n_iterations=series.step_count,
            )
        return series

    return series


def _load_total(
    base_path: str,
    structure: dict[str, dict[str, list[int]]],
    *,
    session_cache: Any | None = None,
) -> Any:
    """Loads all available series into a SimulationTotalResult.

    If *session_cache* is provided (typically ``st.session_state``), each
    loaded series is also stored individually under the key
    ``_series_{base_path}_{model}_{nv_str}_{n_c}`` so that the Results tab
    rendering does not need to re-read the parquets.
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


def _total_result_dir(base_path: str) -> str:
    """Path to the save directory of the aggregated SimulationTotalResult."""
    return str(Path(base_path) / "results" / "_total_result")


def _save_total_to_disk(total: Any, base_path: str) -> None:
    """Persists the total to disk (silently ignores errors)."""
    try:
        total.save_to_dir(_total_result_dir(base_path))
    except Exception:
        pass


def _load_total_from_disk(base_path: str) -> Any | None:
    """Loads the total from the persisted directory, or returns None."""
    from vote_simulation.models.results.total_result import SimulationTotalResult

    d = _total_result_dir(base_path)
    if not Path(d).is_dir() or not any(Path(d).glob("*.parquet")):
        return None
    try:
        return SimulationTotalResult.load_from_dir(d)
    except Exception:
        return None


def _delete_total_from_disk(base_path: str) -> None:
    """Deletes the persisted total directory."""
    from vote_simulation.models.results.total_result import SimulationTotalResult

    SimulationTotalResult.delete_dir(_total_result_dir(base_path))


def _apply_filter(
    total: Any,
    models: list[str],
    voters: list[int],
    cands: list[int],
) -> Any:
    """Filters via a shallow copy of _entries — O(n) references, zero data copy."""
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
    """Executes plot_fn() once and caches the PNG in session_state."""
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
            "Export PNG",
            data=val,
            file_name=filename,
            mime="image/png",
            key=export_key,
        )


def _df_csv_export(df: Any, key: str, filename: str) -> None:
    """CSV download button for a DataFrame (or Styler)."""
    if hasattr(df, "data"):  # pandas Styler → DataFrame sous-jacent
        df = df.data
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Export CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def _build_global_dist_df(total_f: Any) -> Any:
    """Builds the mean inter-rule distance matrix for a SimulationTotalResult."""
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
    """Builds the (rules × metrics) mean matrix for a SimulationTotalResult."""
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
    """Filters the total on the 3rd parameter if needed for plot_metric_heatmap."""
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
            f"Fix `{third_p}` (required — 2 axes already taken)",
            third_vals,
            key=f"{key_prefix}_third",
        )
        return total_f.filter(**{third_p: sel})
    return total_f


# ---------------------------------------------------------------------------
# Rendu principal
# ---------------------------------------------------------------------------


def render_tab_results() -> None:
    st.header("Results")
    st.caption("Exploration via the native `SimulationSeriesResult` and `SimulationTotalResult` methods.")

    # During a Full run, avoid reading all existing parquets:
    # _load_total() would block the main thread for 20-60 s and freeze the progress bar.
    if st.session_state.get("full_run_running"):
        st.info("⏳ Full run in progress — results will be available at the end.")
        return

    cfg: dict = st.session_state["cfg"]
    base_path = str(Path(cfg.get("output_base_path", "../data/")).resolve())

    # Cache of the filesystem scan: avoids re-globbing folders on every rerun.
    # The key is cleared by the Full run button handler, so after each simulation
    # the first display re-scans to include new results.
    _scan_key = f"_scan_struct_{base_path}"
    if _scan_key not in st.session_state:
        st.session_state[_scan_key] = _scan_sim_result(base_path)
    structure = st.session_state[_scan_key]

    if not structure:
        st.info(
            f"No results in `{base_path}/sim_result/`.\n\nLaunch a simulation in the **Simulation** tab."
        )
        return

    # ── Session cache of the total ──────────────────────────────────────────
    cache_key = f"_res_total_{base_path}"

    # Priority 1: in-memory result built during the current simulation.
    # Avoids reading all Parquet files from disk after a Full run.
    if cache_key not in st.session_state and "sim_total_result" in st.session_state:
        total_from_sim = st.session_state["sim_total_result"]
        st.session_state[cache_key] = total_from_sim
        # Persist to disk so next session loads directly from the saved total.
        _save_total_to_disk(total_from_sim, base_path)

    # Priority 2: load from the persisted total directory (fast — one parquet
    # per series with metrics included, no need to re-read all individual iterations).
    if cache_key not in st.session_state:
        saved = _load_total_from_disk(base_path)
        if saved is not None and saved.series_count > 0:
            st.session_state[cache_key] = saved

    # Priority 3: load from disk (first display or existing data).
    # session_cache=st.session_state: each loaded series is also cached
    # individually → zero double parquet reads in the "Series analysis" tab.
    if cache_key not in st.session_state:
        with st.spinner("Initial loading of all results…"):
            loaded = _load_total(base_path, structure, session_cache=st.session_state)
            st.session_state[cache_key] = loaded
            # Persist the aggregated total (with metrics) for faster future loads.
            if loaded.series_count > 0:
                _save_total_to_disk(loaded, base_path)

    total: Any = st.session_state[cache_key]

    tab_serie, tab_global, tab_manage = st.tabs(["Series analysis", "Global view", "Data management"])

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
            # Fast path: the total is already in memory — no disk I/O needed.
            try:
                st.session_state[series_key] = total.get_series(sel_model, int(sel_voters), int(sel_cands))
            except KeyError:
                with st.spinner("Chargement de la série…"):
                    st.session_state[series_key] = _load_series(base_path, sel_model, int(sel_voters), int(sel_cands))
        series = st.session_state[series_key]

        if series.step_count == 0:
            st.warning("Aucune donnée valide pour cette sélection.")
            return

        rules = series.mean_distance_matrix_frame.columns.tolist()
        tag = f"{sel_model}_v{sel_voters}_c{sel_cands}"
        st.success(f"**{series.step_count} itérations** · **{len(rules)} règles** : {', '.join(rules)}")

        # ── Filtre de règles ─────────────────────────────────────────────────
        sel_rules_s = st.multiselect(
            "Règles à afficher",
            options=rules,
            default=rules,
            key=f"rule_sel_s_{tag}",
        )
        _active_rules_s = sel_rules_s if sel_rules_s else rules
        if _active_rules_s != rules:
            _series_view_key = f"_series_view_{tag}_{'_'.join(sorted(_active_rules_s))}"
            if _series_view_key not in st.session_state:
                st.session_state[_series_view_key] = series.filter_rules(_active_rules_s)
            series_v = st.session_state[_series_view_key]
            view_tag = f"{tag}_r{'_'.join(sorted(_active_rules_s))}"
        else:
            series_v = series
            view_tag = tag
        view_rules = series_v.mean_distance_matrix_frame.columns.tolist() if series_v.step_count > 0 else []

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
                f"_plt_dist_{view_tag}",
                lambda: series_v.plot_mean_distance_matrix(show=False),
                f"dl_dist_{view_tag}",
                f"dist_{view_tag}.png",
            )

        # ── Projection 2D ──────────────────────────────────────────────────
        with s_tabs[1]:
            if len(view_rules) < 3:
                st.info("Au moins 3 règles nécessaires pour la projection 2D.")
            else:
                st.caption("`series.plot_rules_2d()` — distances projetées par MDS")
                _cached_plot(
                    f"_plt_2d_{view_tag}",
                    lambda: series_v.plot_rules_2d(show=False),
                    f"dl_2d_{view_tag}",
                    f"mds2d_{view_tag}.png",
                )

        # ── Projection 3D ──────────────────────────────────────────────────
        with s_tabs[2]:
            if len(view_rules) < 4:
                st.info("Au moins 4 règles nécessaires pour la projection 3D.")
            else:
                st.caption("`series.plot_rules_3d()` — distances projetées par MDS 3D")
                _cached_plot(
                    f"_plt_3d_{view_tag}",
                    lambda: series_v.plot_rules_3d(show=False),
                    f"dl_3d_{view_tag}",
                    f"mds3d_{view_tag}.png",
                )

        # ── Métriques des gagnants ──────────────────────────────────────────
        with s_tabs[3]:
            from vote_simulation.models.results.total_result import SimulationTotalResult as _STR
            from vote_simulation.models.rules.winner_metrics import METRIC_FIELDS as _MF

            msf = series_v.metrics_summary_frame
            if msf.empty:
                st.info(
                    "Winner metrics are not available for this series.\n\n"
                    "They are computed during a **Full run** "
                    "(Configuration tab → **Full run** button)."
                )
            else:
                # Wrap the filtered series in a SimulationTotalResult to use
                # plot_metrics_rules_matrix
                _total_one_key = f"_total_one_{view_tag}"
                if _total_one_key not in st.session_state:
                    _t1 = _STR()
                    _t1.add_series(series_v)
                    st.session_state[_total_one_key] = _t1
                _total_one = st.session_state[_total_one_key]

                avail_mf = [f for f in _MF if any(c.startswith(f + "_") for c in msf.columns)]
                mc1, mc2 = st.columns([3, 1])
                sel_mf_sm = mc1.multiselect(
                    "Metrics to display",
                    options=avail_mf,
                    default=avail_mf,
                    key=f"sm_mf_{view_tag}",
                )
                stat_sm = mc2.radio("Statistic", ["mean", "std"], horizontal=True, key=f"sm_stat_{view_tag}")
                if not sel_mf_sm:
                    st.info("Select at least one metric.")
                else:
                    _sm_key = f"_plt_sm_{view_tag}_{sorted(sel_mf_sm)}_{stat_sm}"
                    _cached_plot(
                        _sm_key,
                        lambda _m=sel_mf_sm, _s=stat_sm: _total_one.plot_metrics_rules_matrix(
                            metrics=_m, stat=_s, show=False
                        ),
                        f"dl_sm_{view_tag}",
                        f"metrics_{view_tag}.png",
                    )

                    st.dataframe(
                        msf.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                        width="stretch",
                    )
                    _df_csv_export(msf, f"dl_csv_sm_{view_tag}", f"metrics_{view_tag}.csv")

        # ── Résumé ─────────────────────────────────────────────────────────
        with s_tabs[4]:
            ca, cb = st.columns(2)
            ca.metric("Distance moyenne", f"{series_v.mean_distance:.2f} %")
            ra, rb, d = series_v.most_distant_rules
            if ra:
                cb.metric("Paire la + distante", f"{ra} ↔ {rb}", f"{d:.2f} %")

            st.markdown("**Matrice de distance** (`mean_distance_matrix_frame`)")
            st.dataframe(
                series_v.mean_distance_matrix_frame.style.background_gradient(cmap="Reds", vmin=0, vmax=100).format(
                    "{:.1f}"
                ),
                width="stretch",
            )
            _df_csv_export(
                series_v.mean_distance_matrix_frame,
                f"dl_csv_dist_{view_tag}",
                f"distance_matrix_{view_tag}.csv",
            )

            msf = series_v.metrics_summary_frame
            if not msf.empty:
                st.markdown("**Métriques des gagnants** (`metrics_summary_frame`)")
                st.dataframe(msf.style.format("{:.4f}"), width="stretch")
                _df_csv_export(msf, f"dl_csv_msf_{view_tag}", f"metrics_{view_tag}.csv")
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
                    _k in (cache_key, _scan_key, "gf_m_applied", "gf_v_applied", "gf_c_applied")
                    or (isinstance(_k, str) and _k.startswith("_filtered_"))
                    or (isinstance(_k, str) and _k.startswith("_rfilt_"))
                    or (isinstance(_k, str) and _k.startswith("_all_rules_g_"))
                    or (isinstance(_k, str) and _k.startswith("gf_r_applied_"))
                    or (isinstance(_k, str) and _k.startswith("_plt_g_"))
                    or (isinstance(_k, str) and _k.startswith("_avail_rules_"))
                    or (isinstance(_k, str) and _k.startswith("_series_"))
                    or (isinstance(_k, str) and _k.startswith("_total_one_"))
                    or (isinstance(_k, str) and _k.startswith("_g_dist_df_"))
                    or (isinstance(_k, str) and _k.startswith("_g_met_df_"))
                ):
                    del st.session_state[_k]
            # Also delete the disk cache of the aggregated total to force
            # reconstruction from individual series (includes new series).
            _delete_total_from_disk(base_path)
            st.rerun()

        # Filters with an Apply button to avoid recomputation on every widget change
        _fk_models = "gf_m_applied"
        _fk_voters = "gf_v_applied"
        _fk_cands = "gf_c_applied"
        # Currently applied values (defaults: all selected)
        applied_models = st.session_state.get(_fk_models, list(total.gen_models))
        applied_voters = st.session_state.get(_fk_voters, list(total.voter_counts))
        applied_cands = st.session_state.get(_fk_cands, list(total.candidate_counts))

        with st.expander("Filters", expanded=True):
            fc1, fc2, fc3, fc4 = st.columns([3, 3, 3, 1])
            f_models = fc1.multiselect("Models", total.gen_models, default=applied_models, key="gf_m")
            f_voters = fc2.multiselect("Voters", total.voter_counts, default=applied_voters, key="gf_v")
            f_cands = fc3.multiselect("Candidates", total.candidate_counts, default=applied_cands, key="gf_c")
            if fc4.button("Apply", key="btn_apply_filters", width="stretch"):
                new_models = f_models or list(total.gen_models)
                new_voters = f_voters or list(total.voter_counts)
                new_cands = f_cands or list(total.candidate_counts)
                # Invalider le cache filtré si les filtres changent
                if new_models != applied_models or new_voters != applied_voters or new_cands != applied_cands:
                    # Supprimer tous les caches de plots et de règles associés
                    old_fk = (
                        f"_filtered_{cache_key}_"
                        f"{sorted(applied_models)}_{sorted(applied_voters)}_{sorted(applied_cands)}"
                    )
                    for k in list(st.session_state.keys()):
                        if (
                            (isinstance(k, str) and k.startswith("_plt_g_"))
                            or (isinstance(k, str) and k.startswith("_rfilt_"))
                            or (isinstance(k, str) and k.startswith("_all_rules_g_"))
                            or (isinstance(k, str) and k.startswith("gf_r_applied_"))
                            or k == old_fk
                        ):
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

        # ── Filtre de règles (vue globale) ───────────────────────────────────
        # Collect the union of all rule labels across the (param-)filtered total.
        _all_rules_g_key = f"_all_rules_g_{_filter_cache_key}"
        if _all_rules_g_key not in st.session_state:
            _rls: list[str] = []
            for _, _s in total_f:
                for _r in _s._rule_order:
                    if _r not in _rls:
                        _rls.append(_r)
            st.session_state[_all_rules_g_key] = sorted(_rls)
        _all_rules_g: list[str] = st.session_state[_all_rules_g_key]

        _fk_rules = f"gf_r_applied_{_filter_cache_key}"
        _applied_rules_g = st.session_state.get(_fk_rules, _all_rules_g)
        # Drop stale rule selections that no longer exist after a param-filter change
        _applied_rules_g = [r for r in _applied_rules_g if r in _all_rules_g] or _all_rules_g

        sel_rules_g = st.multiselect(
            "Rules to display",
            options=_all_rules_g,
            default=_applied_rules_g,
            key=f"gf_r_{_filter_cache_key}",
        )
        _active_rules_g = sel_rules_g if sel_rules_g else _all_rules_g
        if _active_rules_g != _applied_rules_g:
            # Invalidate rule-filtered plot caches when selection changes
            for _k in list(st.session_state.keys()):
                if isinstance(_k, str) and _k.startswith("_plt_g_"):
                    del st.session_state[_k]
            st.session_state[_fk_rules] = _active_rules_g

        # Lightweight rule-filtered view — O(n_rules²) numpy copy per series
        _rule_cache_tag = "_".join(sorted(_active_rules_g))
        _rule_filter_key = f"_rfilt_{_filter_cache_key}_{_rule_cache_tag}"
        if _rule_filter_key not in st.session_state:
            if _active_rules_g == _all_rules_g:
                st.session_state[_rule_filter_key] = total_f
            else:
                st.session_state[_rule_filter_key] = total_f.filter_rules(_active_rules_g)
        total_fr: Any = st.session_state[_rule_filter_key]

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
                f"_plt_g_dist_{_rule_filter_key}",
                lambda: total_fr.plot_mean_distance_matrix(show=False),
                "dl_g_dist",
                "global_distance_matrix.png",
            )
            _g_dist_df_key = f"_g_dist_df_{_rule_filter_key}"
            if _g_dist_df_key not in st.session_state:
                st.session_state[_g_dist_df_key] = _build_global_dist_df(total_fr)
            _g_dist_df = st.session_state[_g_dist_df_key]
            if not _g_dist_df.empty:
                st.markdown("**Mean distance matrix** (%)")
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
                    f"_plt_g_gm_{_rule_filter_key}_{sorted(sel_gm)}_{stat_gm}",
                    lambda _m=sel_gm, _s=stat_gm: total_fr.plot_metrics_rules_matrix(metrics=_m, stat=_s, show=False),
                    "dl_gm_matrix",
                    "metrics_rules_matrix.png",
                )
                _g_met_df_key = f"_g_met_df_{_rule_filter_key}_{sorted(sel_gm)}_{stat_gm}"
                if _g_met_df_key not in st.session_state:
                    st.session_state[_g_met_df_key] = _build_global_metrics_df(total_fr, metrics=sel_gm, stat=stat_gm)
                _g_met_df = st.session_state[_g_met_df_key]
                if not _g_met_df.empty:
                    st.markdown("**Raw data** (rules × metrics)")
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
            total_for_hm = _third_param_filter(total_fr, row_p, col_p, "hm")
            # Utilise la valeur sélectionnée du 3ème param comme discriminant de cache
            # (len(_entries) ne suffit pas : deux valeurs différentes peuvent avoir le même count)
            _hm_third_tag = st.session_state.get("hm_third", "")
            _cached_plot(
                f"_plt_g_hm_{_rule_filter_key}_{row_p}_{col_p}_{_hm_third_tag}",
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
                f"_plt_g_grid_{_rule_filter_key}_{vary_p}",
                lambda _vp=vary_p: total_fr.plot_comparison_grid(vary_param=_vp, show=False),
                "dl_g_grid",
                "comparison_grid.png",
            )

        # ── Paire de règles ──────────────────────────────────────────────────
        with g_tabs[5]:
            st.caption("`total.plot_rule_pair_heatmap()` — distance Jaccard entre deux règles croisée sur 2 paramètres")
            # avail_rules comes from the rule-filtered view so the pair selector
            # only shows rules that are currently visible.
            _rules_cache_key = f"_avail_rules_{_rule_filter_key}"
            if _rules_cache_key not in st.session_state:
                _rules: list[str] = []
                for _k, _s in total_fr:
                    for _r in _s._rule_order:
                        if _r not in _rules:
                            _rules.append(_r)
                st.session_state[_rules_cache_key] = sorted(_rules)
            avail_rules: list[str] = st.session_state[_rules_cache_key]

            if len(avail_rules) < 2:
                st.info("At least 2 rules required.")
            else:
                pc1, pc2 = st.columns(2)
                rule_a = pc1.selectbox("Rule A", avail_rules, index=0, key="pair_a")
                rule_b_opts = [r for r in avail_rules if r != rule_a]
                rule_b = pc2.selectbox("Rule B", rule_b_opts, index=0, key="pair_b")

                pr1, pr2 = st.columns(2)
                rp_pair = pr1.selectbox("Row axis", PARAMS, index=0, key="pair_row")
                cp_pair = pr2.selectbox(
                    "Column axis",
                    [p for p in PARAMS if p != rp_pair],
                    index=0,
                    key="pair_col",
                )
                _cached_plot(
                    f"_plt_g_pair_{_rule_filter_key}_{rule_a}_{rule_b}_{rp_pair}_{cp_pair}",
                    lambda _ra=rule_a, _rb=rule_b, _rp=rp_pair, _cp=cp_pair: total_fr.plot_rule_pair_heatmap(
                        _ra, _rb, row_param=_rp, col_param=_cp, show=False
                    ),
                    "dl_g_pair",
                    f"pair_{rule_a}_{rule_b}.png",
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet C — Gestion des données
    # ══════════════════════════════════════════════════════════════════════════
    with tab_manage:
        import shutil

        st.subheader("Gestion des données")

        # ── Cache total agrégé ───────────────────────────────────────────────
        st.markdown("### Cache total agrégé")
        total_dir = Path(_total_result_dir(base_path))
        total_files = list(total_dir.glob("*.parquet")) if total_dir.is_dir() else []
        if total_files:
            st.info(
                f"Cache total présent : **{len(total_files)} série(s)** dans `{total_dir}`.\n\n"
                "Ce cache est utilisé pour charger les résultats rapidement sans relire toutes les itérations."
            )
            if st.button(
                "🗑️ Supprimer le cache total",
                key="btn_del_total_cache",
                help="Force le rechargement depuis les séries individuelles à la prochaine ouverture.",
            ):
                _delete_total_from_disk(base_path)
                # Clear session-state references
                for _k in list(st.session_state.keys()):
                    if _k == cache_key or (isinstance(_k, str) and _k.startswith("_filtered_")):
                        del st.session_state[_k]
                st.success("Cache total supprimé. Les résultats seront rechargés depuis les séries individuelles.")
                st.rerun()
        else:
            st.info("Aucun cache total présent.")
            if st.button("💾 Enregistrer le total actuel sur disque", key="btn_save_total_now"):
                _save_total_to_disk(total, base_path)
                st.success(f"Total sauvegardé dans `{total_dir}`.")

        st.divider()

        # ── Suppression de séries individuelles ──────────────────────────────
        st.markdown("### Supprimer des séries individuelles")
        st.caption("Supprime les fichiers parquet de la série sélectionnée dans `results/` et `sim_result/`.")

        if not structure:
            st.info("Aucune série disponible.")
        else:
            dm1, dm2, dm3 = st.columns(3)
            del_model = dm1.selectbox("Modèle", sorted(structure.keys()), key="del_model")
            del_voters_opts = sorted(structure.get(del_model, {}).keys(), key=int)
            del_voters = dm2.selectbox("Votants", del_voters_opts, key="del_voters")
            del_cands_opts = sorted(structure.get(del_model, {}).get(del_voters, []))
            del_cands = dm3.selectbox("Candidats", del_cands_opts, key="del_cands")

            # Show what would be deleted
            _del_res_files = sorted(
                (Path(base_path) / "results").glob(f"{del_model}_v{del_voters}_c{del_cands}_i*.parquet")
            )
            _del_sim_dir = Path(base_path) / "sim_result" / f"{del_model}_v{del_voters}_c{del_cands}"
            _del_sim_files = sorted(_del_sim_dir.glob("*.parquet")) if _del_sim_dir.is_dir() else []
            _del_total_file = total_dir / f"{del_model}_v{del_voters}_c{del_cands}.parquet"

            n_files = len(_del_res_files) + len(_del_sim_files) + (1 if _del_total_file.is_file() else 0)
            if n_files > 0:
                st.warning(
                    f"Files that will be deleted: **{n_files}** "
                    f"({len(_del_res_files)} results, {len(_del_sim_files)} individual iterations, "
                    f"{'1 total cache entry' if _del_total_file.is_file() else '0 total cache entries'})"
                )
            else:
                st.info("No files found for this combination.")

            if st.button(
                f"🗑️ Delete {del_model} v{del_voters} c{del_cands}",
                key="btn_del_series",
                disabled=(n_files == 0),
                type="primary",
            ):
                deleted = 0
                for f in _del_res_files + _del_sim_files:
                    try:
                        f.unlink()
                        deleted += 1
                    except OSError:
                        pass
                if _del_total_file.is_file():
                    try:
                        _del_total_file.unlink()
                        deleted += 1
                    except OSError:
                        pass
                if _del_sim_dir.is_dir() and not any(_del_sim_dir.iterdir()):
                    try:
                        _del_sim_dir.rmdir()
                    except OSError:
                        pass
                # Invalidate caches
                _series_del_key = f"_series_{base_path}_{del_model}_{del_voters}_{del_cands}"
                for _k in list(st.session_state.keys()):
                    if _k in (cache_key, _scan_key, _series_del_key) or (
                        isinstance(_k, str) and (_k.startswith("_filtered_") or _k.startswith("_plt_g_"))
                    ):
                        del st.session_state[_k]
                st.success(f"{deleted} fichier(s) supprimé(s).")
                st.rerun()

        st.divider()

        # ── Suppression de toutes les séries ─────────────────────────────────
        st.markdown("### ⚠️ Supprimer tous les résultats de simulation")
        st.caption(
            "Supprime le contenu de `results/`, `sim_result/` et le cache total. "
            "Les données générées (`gen/`) ne sont pas affectées."
        )
        _confirm_del_all = st.checkbox(
            "Je confirme vouloir supprimer tous les résultats de simulation",
            key="del_all_confirm",
        )
        if st.button(
            "🗑️ Supprimer tous les résultats",
            key="btn_del_all",
            disabled=not _confirm_del_all,
            type="primary",
        ):
            _del_count = 0
            for _dir_name in ("results", "sim_result"):
                _dir = Path(base_path) / _dir_name
                if _dir.is_dir():
                    try:
                        shutil.rmtree(str(_dir))
                        _del_count += 1
                    except OSError:
                        pass
            # Clear all result caches in session state
            for _k in list(st.session_state.keys()):
                if any(
                    isinstance(_k, str) and _k.startswith(pfx)
                    for pfx in (
                        "_res_total_",
                        "_scan_struct_",
                        "_series_",
                        "_filtered_",
                        "_plt_",
                        "_total_one_",
                        "_g_dist_df_",
                        "_g_met_df_",
                        "_avail_rules_",
                    )
                ):
                    del st.session_state[_k]
            st.session_state.pop("sim_total_result", None)
            st.success(f"Deletion done ({_del_count} folder(s) removed).")
            st.rerun()

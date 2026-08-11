"""candidate_distribution.py

Summary table and distribution plots for each candidate from an election CSV.

Usage:
    # single file
    python candidate_distribution.py data/election_AUT_2024.csv --save-dir res/

    # crawl a whole folder (one PNG per CSV)
    python candidate_distribution.py data/ --save-dir res/ --no-show

The CSV must have a row-index column and one column per candidate with numeric ratings.
Cleaning: rows with any NaN are dropped (same logic as in the notebook).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── cleaning ──────────────────────────────────────────────────────────────────


def clean_incomplete_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="any")


# ── summary table ─────────────────────────────────────────────────────────────


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate table: mean, median, std, min/max."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "candidate": col,
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows).set_index("candidate")


# ── plotting ──────────────────────────────────────────────────────────────────


def plot_distributions(
    df: pd.DataFrame,
    *,
    bins: int = 10,
    show: bool = True,
    save_dir: Path | None = None,
    election_name: str = "",
) -> None:
    """One subplot per candidate: histogram + mean/median lines."""
    n_candidates = len(df.columns)
    n_cols = min(3, n_candidates)
    n_rows = int(np.ceil(n_candidates / n_cols))

    # A4 landscape ≈ 11.7 × 8.3 in; keep subplots compact
    fig_w = min(11.7, 3.8 * n_cols)
    fig_h = min(8.3, 2.8 * n_rows + 0.6)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    title = f"{election_name} — rating distribution" if election_name else "Rating distribution"
    fig.suptitle(title, fontsize=10, fontweight="bold")

    rating_min = float(df.min().min())
    rating_max = float(df.max().max())

    for idx, col in enumerate(df.columns):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        s = df[col].dropna()

        ax.hist(
            s, bins=bins, range=(rating_min, rating_max), color="#4C72B0", edgecolor="white", alpha=0.75, density=True
        )
        ax.axvline(s.mean(), color="#e07b00", linewidth=1.4, linestyle="--", label=f"mean={s.mean():.3f}")
        ax.axvline(s.median(), color="#009900", linewidth=1.4, linestyle=":", label=f"median={s.median():.3f}")

        ax.set_title(col, fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel("Rating", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(rating_min, rating_max)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

    for idx in range(n_candidates, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{election_name or 'distribution'}_candidates.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    if show:
        plt.show()
    plt.close(fig)


# ── folder crawler ────────────────────────────────────────────────────────────


def process_folder(
    folder: Path,
    *,
    bins: int = 10,
    show: bool = False,
    save_dir: Path | None = None,
) -> None:
    """Process every CSV in *folder* and export one PNG per file."""
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {folder}", file=sys.stderr)
        return
    print(f"Found {len(csvs)} CSV file(s) in {folder}\n")
    for csv_path in csvs:
        _process_file(csv_path, bins=bins, show=show, save_dir=save_dir)


# ── single-file helper ────────────────────────────────────────────────────────


def _process_file(
    csv_path: Path,
    *,
    bins: int = 10,
    show: bool = True,
    save_dir: Path | None = None,
) -> None:
    df_raw = pd.read_csv(csv_path, index_col=0)
    df_clean = clean_incomplete_data(df_raw)

    n_dropped = len(df_raw) - len(df_clean)
    # divised by 10 if a value of the dataframe is stricly above 1
    if (df_clean > 1).any().any():
        df_clean = df_clean / 10

    print(f"[{csv_path.name}]  {len(df_raw)} voters → {len(df_clean)} clean  ({n_dropped} dropped)")

    if df_clean.empty:
        print("  [SKIP] no rows after cleaning.\n")
        return

    table = summary_table(df_clean)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(table.to_string(), "\n")

    plot_distributions(df_clean, bins=bins, show=show, save_dir=save_dir, election_name=csv_path.stem)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rating distribution summary for election CSV files or folders.")
    parser.add_argument("input", help="CSV file or folder of CSV files.")
    parser.add_argument("--bins", type=int, default=10, help="Histogram bins (default: 10).")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--save-dir", default=None, help="Output directory for PNGs.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    save_dir = Path(args.save_dir) if args.save_dir else None

    if input_path.is_dir():
        process_folder(input_path, bins=args.bins, show=not args.no_show, save_dir=save_dir)
    elif input_path.is_file():
        _process_file(input_path, bins=args.bins, show=not args.no_show, save_dir=save_dir)
    else:
        print(f"Error: {input_path} is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

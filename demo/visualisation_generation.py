"""Utilities to inspect generated profiles and visualize key metrics.

This module centralizes the former demo code that lived in
``vote_simulation/simulation/simulation.py`` and adds plotting helpers.

Usage
-----
All generators (defaults: n_v=1000, n_c=3)::

    python visualisation_generation.py

Single generator::

    python visualisation_generation.py IC
    python visualisation_generation.py IC --n-v 500 --n-c 5 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from vote_simulation.models.data_generation.generator_registry import (
    get_generator_builder,
    list_generator_codes,
)

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

SKIP_CODES: set[str] = {"IANC"}


# ── core plotting helper ────────────────────────────────────────────
def generate_plots(
    gen_code: str,
    n_v: int = 1000,
    n_c: int = 3,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Build a profile and save Plot3 + MDS plots. Returns saved paths."""
    builder = get_generator_builder(gen_code)
    profile = builder(n_v=n_v, n_c=n_c, seed=seed)

    # Plot3 (svvamp 3-D utility scatter)
    profile.plot3()
    plot3_path = IMG_DIR / f"{gen_code.lower()}Plot3.png"
    plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
    plt.close()

    # MDS 2-D projection
    mds = MDS(n_components=2, random_state=seed, normalized_stress="auto")
    coords_2d = mds.fit_transform(profile.preferences_ut)

    plt.figure(figsize=(5, 5))
    plt.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
    plt.title(f"{gen_code} (n_v={n_v}, n_c={n_c}) — MDS 2D projection")
    plt.tight_layout()
    mds_path = IMG_DIR / f"{gen_code.lower()}MDS.png"
    plt.savefig(mds_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot3_path, mds_path


# ── run on all generators ───────────────────────────────────────────
def generate_all(
    n_v: int = 1000,
    n_c: int = 3,
    seed: int = 42,
) -> None:
    """Generate and save plots for every registered generator."""
    codes = list_generator_codes()
    print("Registered generators:", codes)

    for code in codes:
        if code in SKIP_CODES:
            print(f"  → {code} ... skipped (slow)", flush=True)
            continue
        print(f"  → {code} ...", end=" ", flush=True)
        p3, mds_p = generate_plots(code, n_v=n_v, n_c=n_c, seed=seed)
        print(f"saved {p3.name}, {mds_p.name}")

    print(f"\nAll plots saved to {IMG_DIR}")

if __name__ == "__main__":
    #generate_plots("EUCLID", n_v=1000, n_c=3, seed=161)
    generate_all(n_v=1000, n_c=3, seed=161)

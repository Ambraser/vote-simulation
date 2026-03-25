"""Generator registry mapping short codes to svvamp GeneratorProfile factories.

This mirrors the rule registry pattern: each generative model is referenced by
a short uppercase code (e.g. ``"UNI"``, ``"IC"``) and backed by the
corresponding ``svvamp.GeneratorProfile*`` class.

Usage
-----
>>> from vote_simulation.models.data_generation.generator_registry import get_generator_builder
>>> builder = get_generator_builder("UNI")
>>> profile = builder(n_v=100, n_c=5, seed=42)
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from svvamp import (
    GeneratorProfileCubicUniform,
    GeneratorProfileEuclideanBox,
    GeneratorProfileGaussianWell,
    GeneratorProfileIanc,
    GeneratorProfileIc,
    GeneratorProfileLadder,
    GeneratorProfilePerturbedCulture,
    GeneratorProfileSpheroid,
    GeneratorProfileUnanimous,
    GeneratorProfileUniformFewRankings,
    GeneratorProfileVMFHypercircle,
    GeneratorProfileVMFHypersphere,
    Profile,
    initialize_random_seeds,
)

# Types

GeneratorBuilder = Callable[..., Profile]
"""Signature: (n_v, n_c, seed=0, **extra) -> svvamp.Profile"""

# Internal registry

_GENERATOR_BUILDERS: dict[str, GeneratorBuilder] = {}


# Public helpers


def register_generator(code: str, builder: GeneratorBuilder) -> None:
    """Register a generator builder under a short code.

    Args:
        code: Short code - case-insensitive, will be normalized.
        builder: Callable ``(n_v, n_c, seed=0, **extra) -> Profile``.
    """
    _GENERATOR_BUILDERS[code.strip().upper()] = builder


def get_generator_builder(code: str) -> GeneratorBuilder:
    """Return the generator builder for the given code.

    Raises:
        ValueError: If code is not registered.
    """
    normalized = code.strip().upper()
    try:
        return _GENERATOR_BUILDERS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_GENERATOR_BUILDERS))
        raise ValueError(f"Unknown generator code: '{code}'. Available: {available}") from exc


def list_generator_codes() -> list[str]:
    """Return sorted list of all registered generator codes."""
    return sorted(_GENERATOR_BUILDERS)


def normalize_between_0_and_1(profile: Profile) -> Profile:
    """Return a new Profile with utilities normalized to [0, 1]."""
    ut = profile.preferences_ut
    min_ut = np.min(ut)
    max_ut = np.max(ut)
    if max_ut > min_ut:
        normalized_ut = (ut - min_ut) / (max_ut - min_ut)
    else:
        normalized_ut = np.zeros_like(ut)
    return Profile(preferences_ut=normalized_ut, labels_candidates=profile.labels_candidates)


# Seed helper


def _seed(seed: int, iteration: int = 0) -> None:
    """Set global numpy + svvamp seeds for reproducibility."""
    effective = seed + iteration
    np.random.seed(effective)  # noqa: NPY002
    initialize_random_seeds(effective)


# Candidate labels helper


def _make_labels(n_c: int) -> list[str]:
    """Generate default candidate labels ``Candidate 1 .. Candidate n_c``."""
    return [f"Candidate {i + 1}" for i in range(n_c)]


def _relabel(profile: Profile, n_c: int) -> Profile:
    """Return a new Profile with human-readable candidate labels."""
    return Profile(
        preferences_ut=profile.preferences_ut,
        labels_candidates=_make_labels(n_c),
    )


# GENERATORS BUILDER


# UNI: Cubic Uniform (each utility drawn uniformly in [-1, 1]) ----------
def _build_cubic_uniform(n_v: int, n_c: int, *, seed: int = 0, iteration: int = 0, **_kw: object) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileCubicUniform(n_v=n_v, n_c=n_c)
    return _relabel(gen(), n_c)


register_generator("UNI", _build_cubic_uniform)


# IC: Impartial Culture (random ordinal rankings) -----------------------
def _build_ic(n_v: int, n_c: int, *, seed: int = 0, iteration: int = 0, **_kw: object) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileIc(n_v=n_v, n_c=n_c)
    return _relabel(gen(), n_c)


register_generator("IC", _build_ic)


# IANC: Impartial, Anonymous and Neutral Culture ------------------------
def _build_ianc(n_v: int, n_c: int, *, seed: int = 0, iteration: int = 0, **_kw: object) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileIanc(n_v=n_v, n_c=n_c)
    return _relabel(gen(), n_c)


register_generator("IANC", _build_ianc)  # TODO : fix


# EUCLID: Euclidean Box -------------------------------------------------
def _build_euclidean_box(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    box_dimensions: list[float] | None = None,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    if box_dimensions is None:
        box_dimensions = [1.0]
    gen = GeneratorProfileEuclideanBox(n_v=n_v, n_c=n_c, box_dimensions=box_dimensions)
    return _relabel(gen(), n_c)


register_generator("EUCLID", _build_euclidean_box)


# --- GAUSS: Gaussian Well --------------------------------------------------
def _build_gaussian_well(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    sigma: list[float] | None = None,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    if sigma is None:
        sigma = [1.0]
    gen = GeneratorProfileGaussianWell(n_v=n_v, n_c=n_c, sigma=sigma)
    return _relabel(gen(), n_c)


register_generator("GAUSS", _build_gaussian_well)


# --- LADDER ----------------------------------------------------------------
def _build_ladder(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    n_rungs: int = 21,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileLadder(n_v=n_v, n_c=n_c, n_rungs=n_rungs)
    return _relabel(gen(), n_c)


register_generator("LADDER", _build_ladder)


# --- SPHEROID --------------------------------------------------------------
def _build_spheroid(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    stretching: float = 1.0,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileSpheroid(n_v=n_v, n_c=n_c, stretching=stretching)
    return _relabel(gen(), n_c)


register_generator("SPHEROID", _build_spheroid)


# --- PERTURB: Perturbed Culture --------------------------------------------
def _build_perturbed_culture(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    theta: float = 0.1,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfilePerturbedCulture(n_v=n_v, theta=theta, n_c=n_c)
    return _relabel(gen(), n_c)


register_generator("PERTURB", _build_perturbed_culture)


# --- UNANIMOUS -------------------------------------------------------------
def _build_unanimous(n_v: int, n_c: int, *, seed: int = 0, iteration: int = 0, **_kw: object) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileUnanimous(n_v=n_v, n_c=n_c)
    return _relabel(gen(), n_c)


register_generator("UNANIMOUS", _build_unanimous)


# UFR: Uniform Few Rankings ---------------------------------------------
def _build_uniform_few_rankings(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    n_max_rankings: int = 4,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileUniformFewRankings(n_v=n_v, n_c=n_c, n_max_rankings=n_max_rankings)
    return _relabel(gen(), n_c)


register_generator("UFR", _build_uniform_few_rankings)


# VMF_HC: Von Mises-Fisher Hypercircle ----------------------------------
def _build_vmf_hypercircle(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    vmf_concentration: float | list[float] = 10.0,
    vmf_probability: list[float] | None = None,
    vmf_pole: list[list[float]] | None = None,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileVMFHypercircle(
        n_v=n_v,
        n_c=n_c,
        vmf_concentration=vmf_concentration,
        vmf_probability=vmf_probability,
        vmf_pole=vmf_pole,
    )
    return _relabel(gen(), n_c)


register_generator("VMF_HC", _build_vmf_hypercircle)


# --- VMF_HS: Von Mises-Fisher Hypersphere ----------------------------------
def _build_vmf_hypersphere(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    vmf_concentration: float | list[float] = 10.0,
    stretching: float = 1.0,
    vmf_probability: list[float] | None = None,
    vmf_pole: list[list[float]] | None = None,
    **_kw: object,
) -> Profile:
    _seed(seed, iteration)
    gen = GeneratorProfileVMFHypersphere(
        n_v=n_v,
        n_c=n_c,
        vmf_concentration=vmf_concentration,
        stretching=stretching,
        vmf_probability=vmf_probability,
        vmf_pole=vmf_pole,
    )
    return _relabel(gen(), n_c)


register_generator("VMF_HS", _build_vmf_hypersphere)


if __name__ == "__main__":
    from pathlib import Path

    IMG_DIR = Path(__file__).resolve().parents[4] / "docs" / "img"
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    codes = list_generator_codes()
    print("Registered generators:", codes)

    n_v, n_c = 1000, 3

    for gen_code in codes:
        if gen_code == "IANC":
            print(f"  → {gen_code} ... skipped (slow)", flush=True)
            continue
        print(f"  → {gen_code} ...", end=" ", flush=True)
        builder = get_generator_builder(gen_code)
        profile = builder(n_v=n_v, n_c=n_c, seed=42)

        # - Plot3 (svvamp 3-D utility scatter) ---
        profile.plot3()
        plot3_path = IMG_DIR / f"{gen_code.lower()}Plot3.png"
        plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
        plt.close()

        # - MDS 2-D projection ---
        mds = MDS(n_components=2, random_state=42, normalized_stress="auto")
        coords_2d = mds.fit_transform(profile.preferences_ut)

        plt.figure(figsize=(5, 5))
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.7, edgecolors="k", linewidths=0.3)
        plt.title(f"{gen_code} (n_v={n_v}, n_c={n_c}) — MDS 2D projection")
        plt.tight_layout()
        mds_path = IMG_DIR / f"{gen_code.lower()}MDS.png"
        plt.savefig(mds_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"saved {plot3_path.name}, {mds_path.name}")

    print(f"\nAll plots saved to {IMG_DIR}")

"""Generator registry mapping short codes to svvamp GeneratorProfile factories.

Usage examples:
```python
> from vote_simulation.models.data_generation.generator_registry import get_generator_builder
> builder = get_generator_builder("UNI")
> profile = builder(n_v=100, n_c=5, seed=42)
```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
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


def make_generator_builder(
    generator_factory: Callable[..., Any],
    **default_kwargs: object,
) -> GeneratorBuilder:
    """Create a public `GeneratorBuilder` from a generator factory.

    This helper is intended for external users who want to register custom
    generators while reusing the registry's seeding and relabeling logic.

    Args:
        generator_factory: Callable ``(n_v, n_c, **kw) -> svvamp generator``
            that, when called, returns an svvamp generator object.
        **default_kwargs: Default keyword arguments forwarded to the factory.

    Returns:
        A `GeneratorBuilder` that can be registered in the registry.

    Example::

        from svvamp import GeneratorProfileEuclideanBox
        builder = make_generator_builder(
            GeneratorProfileEuclideanBox,
            box_dimensions=[1.0, 1.0, 1.0],
        )
        register_generator("MY_EUCLID_3D", builder)
    """

    def _builder(
        n_v: int,
        n_c: int,
        *,
        seed: int = 0,
        iteration: int = 0,
        **kw: object,
    ) -> Profile:
        _seed(seed, iteration)
        merged = {**default_kwargs, **kw}
        gen = generator_factory(n_v=n_v, n_c=n_c, **merged)
        return _relabel(gen(), n_c)

    return _builder


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
        box_dimensions = [1.0, 1.0]
    gen = GeneratorProfileEuclideanBox(n_v=n_v, n_c=n_c, box_dimensions=box_dimensions)
    return _relabel(gen(), n_c)


register_generator("EUCLID", _build_euclidean_box)


# Pre-configured Euclidean Box by dimensionality
register_generator("EUCLID_1D", make_generator_builder(GeneratorProfileEuclideanBox, box_dimensions=[1.0]))
register_generator("EUCLID_2D", make_generator_builder(GeneratorProfileEuclideanBox, box_dimensions=[1.0, 1.0]))
register_generator("EUCLID_3D", make_generator_builder(GeneratorProfileEuclideanBox, box_dimensions=[1.0, 1.0, 1.0]))
register_generator(
    "EUCLID_5D",
    make_generator_builder(GeneratorProfileEuclideanBox, box_dimensions=[1.0, 1.0, 1.0, 1.0, 1.0]),
)


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
    # Quick test of the registry
    code = "IANC"
    n_voters = 100
    n_candidates = 9

    builder = get_generator_builder(code)
    profile = builder(n_voters, n_candidates, seed=123)
    print(f"Generated profile with code '{code}':")
    print(f"Candidate labels: {profile.labels_candidates}")
    print(f"Utility matrix shape: {profile.preferences_ut.shape}")
    profile.plot3()

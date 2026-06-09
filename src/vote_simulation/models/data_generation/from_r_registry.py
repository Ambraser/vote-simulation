"""R-based data generators using rpy2.

Registers two generators derived from ``ref.r``:

* ``DDD_BETA``      - ``eval_ddd_beta``       (Beta marginals, random a/b in [eps, 3])
* ``DDD_BETA_POLAR``- ``eval_ddd_beta_polar``  (Polarised Beta, a=b in [eps, 0.5])

The R environment and the script are loaded **once** at first use so that
repeated calls do not pay the R startup cost.

Usage:
    from vote_simulation.models.data_generation.from_r_registry import (
        register_r_generators,
    )
    register_r_generators()          # idempotent - safe to call multiple times
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vote_simulation.models.data_generation.generator_registry import (
    Profile,
    _make_labels,
    _seed,
    register_generator,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Lazy R environment – initialised at most once per process
# ---------------------------------------------------------------------------

_R_LOCK = threading.Lock()
_r_env: Any = None  # will hold the rpy2 robjects module's r env


def _get_r_env() -> Any:
    """Return the R environment with the ref.r functions loaded (lazy, thread-safe)."""
    global _r_env  # noqa: PLW0603
    if _r_env is not None:
        return _r_env

    with _R_LOCK:
        if _r_env is not None:  # double-checked locking
            return _r_env

        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
        except ImportError as exc:
            raise ImportError("rpy2 is required for R-based generators. Install it with: uv add rpy2") from exc

        # Ensure required R packages are available
        for pkg in ("MASS", "randcorr"):
            try:
                importr(pkg)
            except Exception as exc:
                raise RuntimeError(
                    f"R package '{pkg}' is not installed. Install it inside R with: install.packages('{pkg}')"
                ) from exc

        from typing import cast as _cast

        # Source the R script
        r_script = Path(__file__).with_name("ref.r")
        r_obj = _cast(Any, ro.r)
        r_obj.source(str(r_script))

        _r_env = r_obj
        return _r_env


# Shared conversion helper


def _r_matrix_to_profile(r_func_name: str, n_v: int, n_c: int, effective_seed: int) -> Profile:
    """Call an R generator function and return an svvamp Profile.

    The R functions return a list of length K; we always request K=1 and
    take the first element (a matrix of shape ``n_c × n_v``).  The matrix
    is transposed to ``(n_v, n_c)`` to match svvamp's ``preferences_ut``
    convention.
    """
    r = _get_r_env()
    r["set.seed"](effective_seed % (2**31))
    r_func: Any = r[r_func_name]
    result: Any = r_func(n_c, n_v, 1)  # K=1
    # result is an R list; extract first element → R matrix (n_c × n_v)
    phi_r: Any = result[0]
    # Convert to numpy: rpy2 flattens column-major → reshape then transpose
    phi_np: np.ndarray = np.array(phi_r).reshape((n_c, n_v), order="F")
    preferences_ut = phi_np.T  # shape (n_v, n_c)
    return Profile(
        preferences_ut=preferences_ut,
        labels_candidates=_make_labels(n_c),
    )


# Builder functions


def _build_ddd_beta(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    **_kw: object,
) -> Profile:
    """Beta marginals; a, b drawn uniformly in [eps, 3] per candidate."""
    effective = seed * 100_000 + iteration
    _seed(seed, iteration)
    return _r_matrix_to_profile("eval_ddd_beta", n_v, n_c, effective)


def _build_ddd_beta_polar(
    n_v: int,
    n_c: int,
    *,
    seed: int = 0,
    iteration: int = 0,
    **_kw: object,
) -> Profile:
    """Polarised Beta marginals; a=b drawn uniformly in [eps, 0.5] per candidate."""
    effective = seed * 100_000 + iteration
    _seed(seed, iteration)
    return _r_matrix_to_profile("eval_ddd_beta_polar", n_v, n_c, effective)


# Registration

_REGISTERED = False
_REG_LOCK = threading.Lock()


def register_r_generators() -> None:
    """Register all R-based generators into the generator registry."""
    global _REGISTERED  # noqa: PLW0603
    if _REGISTERED:
        return
    with _REG_LOCK:
        if _REGISTERED:
            return
        register_generator("DDD_BETA", _build_ddd_beta)
        register_generator("DDD_BETA_POLAR", _build_ddd_beta_polar)
        _REGISTERED = True


# Auto-register when the module is imported
register_r_generators()

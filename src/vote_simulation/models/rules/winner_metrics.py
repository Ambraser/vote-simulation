"""Metrics computed for the set of co-winners of a voting rule.

All metrics are derived from the svvamp :class:`~svvamp.Profile` and the
integer indices of the co-winners.  When several candidates are tied at the
top (multiple co-winners), per-voter series are **stacked** across winners so
that each metric reflects the distribution for the whole winning set at once.

Available metrics
-----------------
social_acceptability
    Fraction of voters who give a strictly positive utility to **at least one**
    co-winner.  Ranges in [0, 1].
utility_mean / utility_median / utility_var
    Mean, median and variance of utilities ``preferences_ut[v, c]`` across all
    (voter, co-winner) pairs.
rank_mean / rank_median / rank_var
    Mean, median and variance of the 1-based rank of each co-winner in each
    voter's preference ordering (1 = most preferred, n_c = least preferred).
    Computed over all (voter, co-winner) pairs.
freq_first
    Fraction of voters who rank **at least one** co-winner first.
freq_last
    Fraction of voters who rank **at least one** co-winner last.
has_tie
    ``True`` iff there are strictly more than one co-winner.

Design notes
~~~~~~~~~~~~
* All computations are fully **vectorised** with NumPy — no Python loops over
  voters or candidates.
* ``preferences_borda_rk[v, c]`` equals ``n_c - 1 - rank_0based`` where
  ``rank_0based`` is 0 for the best candidate.  Hence
  ``rank_1based = n_c - preferences_borda_rk[v, c]``.
* The module is stateless: :func:`compute_winner_metrics` is a pure function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from svvamp import Profile


# ---------------------------------------------------------------------------
# Field catalogue — single source of truth for serialisation / accumulation
# ---------------------------------------------------------------------------

#: Ordered tuple of all numeric metric names stored in :class:`WinnerMetrics`.
#: ``has_tie`` (bool) and ``n_cowinners`` (int) are cast to ``float`` for
#: vectorised accumulation.
METRIC_FIELDS: tuple[str, ...] = (
    "social_acceptability",
    "utility_mean",
    "utility_median",
    "utility_var",
    "rank_mean",
    "rank_median",
    "rank_var",
    "freq_first",
    "freq_last",
    "has_tie",
    "n_cowinners",
)


@dataclass(frozen=True, slots=True)
class WinnerMetrics:
    """Immutable record of winner-population metrics for one election outcome.

    Parameters
    ----------
    social_acceptability:
        Fraction of voters with utility > 0 for at least one co-winner.
    utility_mean:
        Mean of ``preferences_ut[v, c]`` over all (voter, co-winner) pairs.
    utility_median:
        Median of the same distribution.
    utility_var:
        Variance of the same distribution (population variance, ``ddof=0``).
    rank_mean:
        Mean 1-based rank of co-winners across all (voter, co-winner) pairs.
    rank_median:
        Median 1-based rank.
    rank_var:
        Variance of 1-based ranks.
    freq_first:
        Fraction of voters for whom at least one co-winner is ranked 1st.
    freq_last:
        Fraction of voters for whom at least one co-winner is ranked last.
    has_tie:
        ``True`` iff there are multiple co-winners.
    n_cowinners:
        Number of co-winners.
    """

    social_acceptability: float
    utility_mean: float
    utility_median: float
    utility_var: float
    rank_mean: float
    rank_median: float
    rank_var: float
    freq_first: float
    freq_last: float
    has_tie: bool
    n_cowinners: int

    def to_dict(self) -> dict[str, float | bool | int]:
        """Return a plain :class:`dict` suitable for serialisation / dataframes."""
        return {
            "social_acceptability": self.social_acceptability,
            "utility_mean": self.utility_mean,
            "utility_median": self.utility_median,
            "utility_var": self.utility_var,
            "rank_mean": self.rank_mean,
            "rank_median": self.rank_median,
            "rank_var": self.rank_var,
            "freq_first": self.freq_first,
            "freq_last": self.freq_last,
            "has_tie": self.has_tie,
            "n_cowinners": self.n_cowinners,
        }


def compute_winner_metrics(
    profile: Profile,
    cowinner_indices: np.ndarray,
) -> WinnerMetrics:
    """Compute :class:`WinnerMetrics` for a given profile and co-winner set.

    Parameters
    ----------
    profile:
        A fitted :class:`svvamp.Profile`.
    cowinner_indices:
        1-D integer array of co-winner candidate indices (0-based).

    Returns
    -------
    WinnerMetrics
        All metrics as an immutable dataclass instance.

    Notes
    -----
    Accessing ``profile.preferences_ut`` and ``profile.preferences_borda_rk``
    triggers svvamp's lazy evaluation — already-computed values are cached, so
    there is no redundant computation across successive calls.
    """
    idx = np.asarray(cowinner_indices, dtype=int)
    n_v: int = int(profile.n_v)
    n_c: int = int(profile.n_c)
    n_w: int = int(idx.size)
    has_tie = n_w > 1

    # Utility metrics                                                      #
    # Shape (n_v, n_w) — all utility values for co-winners across voters.
    ut: np.ndarray = np.asarray(profile.preferences_ut, dtype=float)[:, idx]  # (n_v, n_w)

    # Social acceptability: fraction of voters with ut > 0 for ≥1 co-winner.
    social_acceptability = float(np.any(ut > 0.0, axis=1).mean())

    # Flatten to a single series over all (voter, co-winner) pairs.
    ut_flat = ut.ravel()
    utility_mean = float(ut_flat.mean())
    utility_median = float(np.median(ut_flat))
    utility_var = float(ut_flat.var())

    # Rank metrics                                                         #
    # preferences_borda_rk[v, c] = n_c - 1 - (0-based rank)
    # ⟹  1-based rank = n_c - preferences_borda_rk[v, c]
    borda_rk: np.ndarray = np.asarray(profile.preferences_borda_rk, dtype=int)[:, idx]  # (n_v, n_w)
    ranks_1based: np.ndarray = n_c - borda_rk  # (n_v, n_w)

    ranks_flat = ranks_1based.ravel().astype(float)
    rank_mean = float(ranks_flat.mean())
    rank_median = float(np.median(ranks_flat))
    rank_var = float(ranks_flat.var())

    # First / last frequency                                              #
    # preferences_borda_rk[v, c] == n_c - 1  ---  candidate c is ranked 1st by voter v
    # preferences_borda_rk[v, c] == 0         --- candidate c is ranked last by voter v
    freq_first = float(np.any(borda_rk == n_c - 1, axis=1).mean())
    freq_last = float(np.any(borda_rk == 0, axis=1).mean())

    return WinnerMetrics(
        social_acceptability=social_acceptability,
        utility_mean=utility_mean,
        utility_median=utility_median,
        utility_var=utility_var,
        rank_mean=rank_mean,
        rank_median=rank_median,
        rank_var=rank_var,
        freq_first=freq_first,
        freq_last=freq_last,
        has_tie=has_tie,
        n_cowinners=n_w,
    )


def metrics_to_array(m: WinnerMetrics) -> np.ndarray:
    """Serialise a :class:`WinnerMetrics` instance to a ``float64`` array.

    The order follows :data:`METRIC_FIELDS`.  Boolean and integer fields are
    cast to ``float64`` so that the whole record can be accumulated with a
    single NumPy addition, enabling O(1) mean/variance computation in the
    series accumulator.

    Parameters
    ----------
    m:
        A :class:`WinnerMetrics` instance.

    Returns
    -------
    np.ndarray
        Shape ``(len(METRIC_FIELDS),)`` of dtype ``float64``.
    """
    return np.array(
        [
            m.social_acceptability,
            m.utility_mean,
            m.utility_median,
            m.utility_var,
            m.rank_mean,
            m.rank_median,
            m.rank_var,
            m.freq_first,
            m.freq_last,
            float(m.has_tie),
            float(m.n_cowinners),
        ],
        dtype=np.float64,
    )

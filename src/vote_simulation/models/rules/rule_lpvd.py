"""Lp-mean Voting.

Each candidate's score is the **Lp Fréchet mean** of the utilities they
receive across all voters — the value :math:`y^*` that minimises the total
Lp distance to all voter utilities:

.. math::

    \\text{score}(c) = \\arg\\min_{y \\in [u_{\\min}, u_{\\max}]}
        \\sum_{i=1}^{n} (u_{ic} - y)^p

This mirrors the R ``Lpmoy`` function which calls ``optimize(FoncLp, …)``
where ``FoncLp(y) = sum((x - y)^p)``.

For even :math:`p` the objective is convex and the optimum is unique.
The default exponent is ``p=4`` (the original L4DV rule).

Co-winners are **all candidates sharing the maximum Lp Fréchet mean score**.

``scores_`` is a 1-D float array where ``scores_[c]`` is the Lp Fréchet mean
received by candidate ``c``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from svvamp import Profile

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


def _lp_frechet_mean(scores: np.ndarray, p: float) -> float:
    """Return the Lp Fréchet mean of a 1-D score vector.

    Finds ``y`` in ``[min(scores), max(scores)]`` that minimises
    ``sum((scores - y) ** p)``, mirroring the R ``Lpmoy`` function.

    Parameters
    ----------
    scores:
        1-D array of voter scores for a single candidate.
    p:
        The exponent. Must be a positive even integer for the objective to be
        convex; p=4 is the default (L4DV).
    """
    lo, hi = float(scores.min()), float(scores.max())
    if lo == hi:
        return lo
    result = minimize_scalar(
        lambda y: float(np.sum(np.abs(scores - y) ** p)),
        bounds=(lo, hi),
        method="bounded",
    )
    return float(result.x)


def _lp_mean(utilities: np.ndarray, p: float) -> np.ndarray:
    """Compute the Lp Fréchet mean over voters for each candidate.

    Parameters
    ----------
    utilities:
        2-D array of shape ``(n_voters, n_candidates)``.
    p:
        The exponent (must be a positive integer).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_candidates,)`` with the Lp mean per candidate.
    """
    ut = np.asarray(utilities, dtype=np.float64)
    # utilities shape: (n_voters, n_candidates) — iterate over candidates (columns)
    return np.array([_lp_frechet_mean(ut[:, c], p) for c in range(ut.shape[1])])


class LpvdResult(SvvampRuleWrapper):
    """Lp-mean voting result with co-winner semantics.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Lp-mean voting.
    p:
        The exponent of the power mean (default ``4``).

    Attributes
    ----------
    scores_:
        1-D float array — Lp mean per candidate.
    cowinners_:
        Labels of all candidates tied at the maximum Lp mean.
    profile_:
        The svvamp profile used for the election.
    p:
        The exponent used.
    """

    def __init__(self, profile: Profile, *, p: float = 4) -> None:
        self.profile_ = profile
        self.p = p
        # preferences_ut is (n_voters, n_candidates)
        self.scores_ = _lp_mean(profile.preferences_ut, p)
        self.cowinners_ = self._max_score_cowinners(self.scores_)


def _build_lpvd(*, p: float = 4):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Lp-mean voting.

    Parameters
    ----------
    p:
        The power mean exponent (default ``4``).
    """

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return LpvdResult(profile, p=p)

    return builder


register_rule("L1DV", _build_lpvd(p=1))  # L1 mean = arithmetic mean = utilitarian voting
register_rule("L2DV", _build_lpvd(p=2))  # L2 mean = quadratic mean
register_rule("L3DV", _build_lpvd(p=3))  # L3 mean
register_rule("L4DV", _build_lpvd(p=4))  # L4 mean
register_rule("L5DV", _build_lpvd(p=5))  # L5 mean
register_rule("L6DV", _build_lpvd(p=6))  # L6 mean

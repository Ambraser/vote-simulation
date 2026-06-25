"""Random-k Approval voting wrapper based on a Poisson law.

For each voter ``v``, the rule samples an independent
``k_v ~ Poisson(lambda=c/3)`` where ``c`` is the number of candidates.
Voter ``v`` then approves her top-``k_v`` candidates.

To keep the rule well-defined, each sampled ``k_v`` is clipped to ``[1, c-1]``.

Co-winners are all candidates sharing the maximum approval score.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.base import SvvampRuleWrapper


def _sample_k_poisson_per_voter(profile: Profile) -> np.ndarray:
    """Sample and return a valid k per voter from Poisson(c/3).

    Parameters
    ----------
    profile:
        Profile providing number of voters and candidates.

    Returns
    -------
    np.ndarray
        1-D integer array ``k_per_voter`` of shape ``(n_voters,)`` where each
        entry is in ``[1, c-1]`` when ``c > 0``.
    """
    n_candidates = int(profile.n_c)
    n_voters = int(profile.n_v)

    if n_candidates <= 0:
        return np.zeros(n_voters, dtype=int)

    lam = n_candidates / 3.0
    sampled = np.asarray(np.random.poisson(lam=lam, size=n_voters), dtype=int)
    return np.clip(sampled, 1, n_candidates - 1)


def _k_poisson_approval_scores(profile: Profile, k_per_voter: np.ndarray) -> np.ndarray:
    """Compute per-candidate approval scores with voter-specific ``k``.

    Voter ``v`` approves candidates whose rank is in top-``k_per_voter[v]``.
    """
    n_candidates = int(profile.n_c)
    if n_candidates <= 0:
        return np.zeros(0, dtype=int)

    borda_rk = np.asarray(profile.preferences_borda_rk, dtype=int)
    ranks_1based = n_candidates - borda_rk
    approves = ranks_1based <= np.asarray(k_per_voter, dtype=int)[:, None]
    return np.asarray(approves.sum(axis=0), dtype=int)


class KRandomPoissonApprovalResult(SvvampRuleWrapper):
    """Random-k Approval result with voter-specific stochastic ``k``.

    The rule draws ``k_v ~ Poisson(c/3)`` independently for each voter, clips
    every draw to ``[1, c-1]``, then computes approval scores from top-``k_v``
    approvals.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Random-k Approval.

    Attributes
    ----------
    sampled_k_per_voter:
        Effective sampled ``k`` values, one per voter.
    lambda_poisson:
        The Poisson parameter ``c/3`` used for sampling.
    cowinners_:
        Candidate labels tied at maximum approval score.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self.lambda_poisson = float(profile.n_c) / 3.0
        self.sampled_k_per_voter = _sample_k_poisson_per_voter(profile)
        self.scores_ = _k_poisson_approval_scores(profile, self.sampled_k_per_voter)
        self.cowinners_ = self._max_score_cowinners(self.scores_)


def _build_k_random_poisson_approval():
    """Return a rule builder for Random-k Approval (Poisson(c/3))."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return KRandomPoissonApprovalResult(profile)

    return builder


register_rule("AP_KRP", _build_k_random_poisson_approval())
register_rule("AP_K_POISSON", _build_k_random_poisson_approval())

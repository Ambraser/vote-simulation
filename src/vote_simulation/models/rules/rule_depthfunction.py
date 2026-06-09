"""Depth Function voting rule.

Implements the metric-based ranking depth from:

    Goibert, Clémençon, Irurozki, Mozharovskyi (2022).
    "Statistical Depth Functions for Ranking Distributions: Definitions,
     Statistical Learning and Applications."

Definitions
-----------
Let P be an empirical distribution over strict rankings of n candidates (from
the voters' ballots).  The empirical pairwise probability is:

    p[i, j] = #{voters ranking i before j} / n_v

The Kemeny risk of a ranking σ is the expected Kendall-tau distance:

    L_P(σ) = Σ_{i<j} [ p[i,j] · 1[σ(i)>σ(j)] + p[j,i] · 1[σ(i)<σ(j)] ]

where σ(i) is the rank position of candidate i (0 = most preferred).

By **Proposition 5** of the paper (Kendall-tau case), the ranking depth is:

    D_P(σ) = C(n,2) - L_P(σ)

where C(n,2) = n(n-1)/2 is the maximum possible Kendall-tau distance.

Winner
------
The winner is the candidate at position 0 in the depth-maximising ranking.
Co-winners are all candidates that appear at position 0 in *some* optimal
(i.e. maximum-depth) ranking.

Per-candidate depth scores
--------------------------
``scores_[c]`` is the maximum depth over all rankings placing candidate c at
position 0.  For ``n_c ≤ EXACT_THRESHOLD`` this is computed exactly by
exhaustive enumeration.  For larger profiles a greedy approximation is used:
candidates are placed in order of their majority preference score
(Σ_{j≠c} p[c, j] for the remaining pool).

Complexity
----------
* Exact (n_c ≤ EXACT_THRESHOLD): O(n_c! · n_c²) time.
* Greedy (n_c > EXACT_THRESHOLD): O(n_c³) time.

Note
----
Under the SST (Strictly Stochastically Transitive) condition the deepest
ranking is unique and its top candidate is the Condorcet winner (Proposition 6
of the paper).  In the general case, depth maximisation is equivalent to
solving the Kemeny problem (NP-hard), hence the exact mode is restricted to
small profiles.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule

# Candidates threshold below which exhaustive enumeration is used.
EXACT_THRESHOLD: int = 8


class DepthFunctionResult(SvvampRuleWrapper):
    """Depth-function voting rule (Goibert et al., AISTATS 2022).

    Computes the empirical ranking depth under the Kendall-tau metric and
    returns the candidate(s) at position 0 in the depth-maximising ranking(s).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run the depth function rule.

    Attributes
    ----------
    scores_ : numpy.ndarray, shape (n_c,)
        Per-candidate depth score.  ``scores_[c]`` is the maximum depth of
        any ranking that places candidate c at position 0.  Higher is better.
    cowinners_ : list[str]
        Labels of all candidates tied at the maximum depth score.
    profile_ : svvamp.Profile
        The profile used for this election.
    """

    def __init__(self, profile: Profile, exact_threshold: int = EXACT_THRESHOLD) -> None:
        self.profile_ = profile
        n_c: int = profile.n_c
        n_v: int = profile.n_v

        # Pairwise probability matrix.
        # p[i, j] = fraction of voters who rank candidate i strictly before j.
        p: np.ndarray = np.array(profile.matrix_duels_rk, dtype=float) / max(n_v, 1)

        if n_c <= exact_threshold:
            scores, winner_indices = self._exact(p, n_c)
        else:
            scores, winner_indices = self._greedy(p, n_c)

        self.scores_: np.ndarray = scores
        self.cowinners_: list[str] = self._resolve_cowinners(winner_indices)

    # ------------------------------------------------------------------
    # Exact enumeration (small n_c)
    # ------------------------------------------------------------------

    def _exact(self, p: np.ndarray, n_c: int) -> tuple[np.ndarray, np.ndarray]:
        """Enumerate all n_c! rankings and return per-candidate best depths.

        Returns
        -------
        scores : ndarray, shape (n_c,)
            ``scores[c]`` = max depth over all rankings that place c first.
        winner_indices : ndarray of int
            Candidates at position 0 in some global maximum-depth ranking.
        """
        best_global: float = -np.inf
        # best_depth_for[c] = best depth of any ranking with c at position 0
        best_depth_for: np.ndarray = np.full(n_c, -np.inf, dtype=float)
        winner_set: set[int] = set()

        rank_pos: np.ndarray = np.empty(n_c, dtype=int)

        for perm in permutations(range(n_c)):
            # Fill rank_pos array
            for pos, cand in enumerate(perm):
                rank_pos[cand] = pos

            depth: float = self._depth_from_rank_pos(rank_pos, p, n_c)
            top: int = perm[0]

            # Update per-candidate best
            if depth > best_depth_for[top]:
                best_depth_for[top] = depth

            # Update global best and winner set
            if depth > best_global + 1e-12:
                best_global = depth
                winner_set = {top}
            elif abs(depth - best_global) <= 1e-12:
                winner_set.add(top)

        winner_indices = np.array(sorted(winner_set), dtype=int)
        return best_depth_for, winner_indices

    # ------------------------------------------------------------------
    # Greedy approximation (large n_c)
    # ------------------------------------------------------------------

    def _greedy(self, p: np.ndarray, n_c: int) -> tuple[np.ndarray, np.ndarray]:
        """Greedy depth approximation for larger profiles.

        For each candidate c, builds a ranking by fixing c at position 0
        then iteratively adding the remaining candidate with the highest
        majority-preference score over the remaining pool.  Depth is computed
        for each such ranking; the candidate(s) with the highest depth are
        returned as co-winners.

        Returns
        -------
        scores : ndarray, shape (n_c,)
            Approximate depth of the best greedy ranking starting with c.
        winner_indices : ndarray of int
            Candidates tied at the maximum approximate depth score.
        """
        scores: np.ndarray = np.zeros(n_c, dtype=float)
        for c in range(n_c):
            ranking = self._greedy_ranking_from(c, p, n_c)
            scores[c] = self._depth_of_ranking(ranking, p, n_c)

        best: float = scores.max()
        winner_indices: np.ndarray = np.flatnonzero(np.isclose(scores, best))
        return scores, winner_indices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _depth_from_rank_pos(rank_pos: np.ndarray, p: np.ndarray, n_c: int) -> float:
        """Compute D_P(σ) = C(n,2) − L_P(σ) from a rank-position array.

        Parameters
        ----------
        rank_pos : ndarray, shape (n_c,)
            ``rank_pos[c]`` = position of candidate c in ranking σ (0 = best).
        p : ndarray, shape (n_c, n_c)
            Pairwise probability matrix.
        n_c : int
            Number of candidates.
        """
        risk: float = 0.0
        for i in range(n_c):
            for j in range(i + 1, n_c):
                # If σ places i before j (i preferred in σ): disagreement = p[j,i]
                # If σ places j before i (j preferred in σ): disagreement = p[i,j]
                if rank_pos[i] < rank_pos[j]:
                    risk += p[j, i]
                else:
                    risk += p[i, j]
        return n_c * (n_c - 1) / 2.0 - risk

    @staticmethod
    def _greedy_ranking_from(start: int, p: np.ndarray, n_c: int) -> list[int]:
        """Build a greedy ranking starting with *start*.

        At each step, among the remaining candidates, pick the one with the
        highest total pairwise majority preference over the other remaining
        candidates (i.e. the Borda-style winner of the remaining pool).
        """
        remaining: list[int] = list(range(n_c))
        remaining.remove(start)
        ranking: list[int] = [start]
        while remaining:
            best_c: int = max(
                remaining,
                key=lambda c: sum(p[c, j] for j in remaining if j != c),
            )
            ranking.append(best_c)
            remaining.remove(best_c)
        return ranking

    @staticmethod
    def _depth_of_ranking(ranking: list[int], p: np.ndarray, n_c: int) -> float:
        """Compute D_P(σ) for a ranking given as an ordered list of candidates."""
        rank_pos: np.ndarray = np.empty(n_c, dtype=int)
        for pos, cand in enumerate(ranking):
            rank_pos[cand] = pos
        return DepthFunctionResult._depth_from_rank_pos(rank_pos, p, n_c)
    
    


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_depthfunction(exact_threshold: int = EXACT_THRESHOLD):

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return DepthFunctionResult(profile, exact_threshold=exact_threshold)

    return builder


# ---------------------------------------------------------------------------
# Rule registration
# ---------------------------------------------------------------------------

register_rule("DEPF", _build_depthfunction())
register_rule("DEPF_GREEDY", _build_depthfunction(exact_threshold=0))
register_rule("DEPF_EXACT", _build_depthfunction(exact_threshold=float("inf")))

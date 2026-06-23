"""Random Winner Voting.

One candidate is chosen uniformly at random as the winner,
regardless of voter preferences.

``scores_`` is a 1-D float array with a single 1.0 at the elected
candidate's index and 0.0 everywhere else.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


def _random_winner_scores(profile: Profile) -> np.ndarray:
    """Return the scores for the random winner rule on a given profile.

    A single candidate is chosen uniformly at random; she receives score 1.0
    and all others receive 0.0.
    """
    n_candidates: int = profile.n_c
    scores = np.zeros(n_candidates, dtype=float)
    if n_candidates == 0:
        return scores
    elected: int = np.random.randint(n_candidates)
    scores[elected] = 1.0
    return scores


class RandomWinnerResult(SvvampRuleWrapper):
    """Random Winner rule result.

    One candidate is chosen uniformly at random as the winner,
    regardless of voter preferences.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Random Winner.

    Attributes
    ----------
    scores_:
        1-D float array with 1.0 at the elected candidate's index, 0.0 elsewhere.
    cowinners_:
        Singleton list containing the elected candidate's label.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self.scores_ = _random_winner_scores(profile)
        self.cowinners_ = self._max_score_cowinners(self.scores_)


def _build_random_winner(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
    """Build a :class:`RandomWinnerResult` from ballots or a svvamp profile."""
    profile = _ensure_profile(profile_or_ballots, candidates)
    return RandomWinnerResult(profile)


register_rule("RANDOM_WINNER", _build_random_winner)

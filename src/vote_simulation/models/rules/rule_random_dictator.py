"""Random Dictator Voting.

One voter is selected uniformly at random and her best graded candidate is elected.
Co-winners are all candidates sharing the maximum score from the selected voter.

``scores_`` is a 1-D float array where ``scores_[c]`` is the grade given to
candidate ``c`` by the randomly selected voter.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


def _random_dictator_scores(profile: Profile) -> np.ndarray:
    """Return the scores for the random dictator rule on a given profile."""
    n_voters: int = profile.n_v
    if n_voters == 0:
        return np.zeros(profile.n_c, dtype=float)
    random_voter_index: int = np.random.randint(n_voters)
    return np.array(profile.preferences_ut[random_voter_index], dtype=float)


class RandomDictatorResult(SvvampRuleWrapper):
    """Random Dictator rule result.

    One voter is selected uniformly at random and her best graded candidate is
    elected. Co-winners are all candidates sharing the maximum score from the
    selected voter.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Random Dictator.

    Attributes
    ----------
    scores_:
        1-D float array of grades given by the randomly selected voter.
    cowinners_:
        List of candidate labels tied at the maximum grade.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self.scores_ = _random_dictator_scores(profile)
        self.cowinners_ = self._max_score_cowinners(self.scores_)


def _build_random_dictator(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
    """Build a :class:`RandomDictatorResult` from ballots or a svvamp profile."""
    profile = _ensure_profile(profile_or_ballots, candidates)
    return RandomDictatorResult(profile)


register_rule("RANDOM_DICTATOR", _build_random_dictator)

"""Schulze method wrapper

``scores_[c, d]`` is the width of the widest path from candidate ``c`` to
candidate ``d`` in the capacitated graph of pairwise duels.

Candidate ``c`` is a **potential winner** (co-winner) iff no candidate ``d``
is strictly better than ``c``, i.e. there is no ``d`` with
``scores_[d, c] > scores_[c, d]``.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleSchulze

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class SchulzeResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleSchulze` with proper co-winner semantics.

    Co-winners are all candidates ``c`` for which no other candidate ``d``
    satisfies ``scores_[d, c] > scores_[c, d]`` (i.e. no candidate beats ``c``
    in the widest-path sense).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Schulze.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleSchulze()(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = self._inner.scores_  # 2-D: scores_[c, d] = widest path c→d
        n_c = scores.shape[0]
        # c is a potential winner iff no d strictly beats it
        potential = np.array([not any(scores[d, c] > scores[c, d] for d in range(n_c) if d != c) for c in range(n_c)])
        indices = np.where(potential)[0]
        return self._resolve_cowinners(indices)


def _build_schulze():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Schulze."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return SchulzeResult(profile)

    return builder


register_rule("SCHU", _build_schulze())

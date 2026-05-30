"""Young rule wrapper.

Young's method elects the Condorcet winner (rk) when one exists.  When no
Condorcet winner exists, the winner is undefined (``w_ = nan``) and
``cowinners_`` is the empty list.

``svvamp.RuleYoung`` is **not** a subclass of ``Rule`` and exposes only ``w_``
(no ``scores_``).  Since a Condorcet winner is always unique when it exists,
ties are impossible and ``cowinners_`` always has 0 or 1 element.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile
from svvamp.rules.rule_young import RuleYoung

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class YoungResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleYoung` with proper co-winner semantics.

    Co-winners:

    * **Exactly one** — the Condorcet winner — when one exists.
    * **Empty list** — when no Condorcet winner exists (``w_`` = nan).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run the Young method.

    Attributes
    ----------
    cowinners_:
        List of 0 or 1 candidate labels.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleYoung()(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        w = self._inner.w_
        # w_ is nan when no Condorcet winner exists
        try:
            w_float = float(w)
        except (TypeError, ValueError):
            w_float = float("nan")

        if np.isfinite(w_float):
            # A unique Condorcet winner exists
            return self._resolve_cowinners(np.array([int(w_float)]))

        # No Condorcet winner: fall back to weak Condorcet winners (e.g. exact
        # two-candidate tie where neither strictly dominates the other).
        weak_winners = getattr(self.profile_, "weak_condorcet_winners", None)
        if weak_winners is not None:
            indices = np.flatnonzero(np.asarray(weak_winners, dtype=bool))
            if indices.size >= 1:
                return self._resolve_cowinners(indices)

        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_young():
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return YoungResult(profile)

    return builder


register_rule("YOUN", _build_young())

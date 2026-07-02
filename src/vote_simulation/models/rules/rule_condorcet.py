"""Condorcet rule wrapper.

Everyone is tied if there is no Condorcet winner. Otherwise the unique
Condorcet winner is the sole winner of the election.

``svvamp`` does not expose a dedicated ``RuleCondorcet`` class here, so this
wrapper reads the Condorcet metadata from :class:`svvamp.Profile` directly and
falls back to a full tie when no strict winner exists.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class CondorcetResult(SvvampRuleWrapper):
	"""Wrapper around the profile Condorcet check with proper co-winner semantics.

	Co-winners are:

	* The unique Condorcet winner when one exists.
	* All candidates when no strict Condorcet winner exists.

	Parameters
	----------
	profile:
		A :class:`svvamp.Profile` on which to evaluate the Condorcet rule.

	Attributes
	----------
	cowinners_:
		List of candidate labels that win under the Condorcet rule.
	profile_:
		The svvamp profile used for the election.
	"""

	def __init__(self, profile: Profile) -> None:
		self.profile_ = profile
		self._inner = profile
		self.cowinners_ = self._compute_cowinners()

	def _compute_cowinners(self) -> list[str]:
		exists = bool(getattr(self.profile_, "exists_condorcet_winner_ut_abs", False))
		winner = getattr(self.profile_, "condorcet_winner_ut_abs", None)

		try:
			winner_float = float(winner)
		except (TypeError, ValueError):
			winner_float = float("nan")

		if exists and np.isfinite(winner_float):
			return self._resolve_cowinners(np.array([int(winner_float)], dtype=int))

		n_candidates = int(getattr(self.profile_, "n_c", 0))
		return self._resolve_cowinners(np.arange(n_candidates, dtype=int))


def _build_condorcet():
	"""Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Condorcet."""

	def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
		profile = _ensure_profile(profile_or_ballots, candidates)
		return CondorcetResult(profile)

	return builder


register_rule("COND", _build_condorcet())







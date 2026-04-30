"""Generic wrapper for svvamp rules whose co-winners live in ``scores_``.

Any rule where the winner is simply the candidate(s) with the highest
1-D ``scores_`` array can subclass :class:`ScoreBasedRuleWrapper` and get
co-winner detection for free.

Known rules that fit this pattern
----------------------------------
* **Approval** — ``scores_[c]`` = number of approving voters (integer).
* **Copeland** — ``scores_[c]`` = number of pairwise victories (float).
* **Borda** — ``scores_[c]`` = Borda sum (integer or float).
* **Plurality** — ``scores_[c]`` = number of first-place votes (integer).
* **Maximin** — ``scores_[c]`` = min pairwise score (float).
* **Veto** — ``scores_[c]`` = votes *not* in last place (integer).
* … (any rule where ``scores_`` is 1-D and higher = better)

Usage
------
Subclass :class:`ScoreBasedRuleWrapper`, implement :meth:`_build_inner` to
return the configured svvamp rule instance, and call :meth:`_init_score_based`
from your ``__init__``::

    class BordaResult(ScoreBasedRuleWrapper):
        def __init__(self, profile: Profile) -> None:
            self.profile_ = profile
            self._inner = RuleBorda()(profile)
            self.cowinners_ = self._init_score_based()

That's all — ``cowinners_``, ``cowinner_indices_``, and ``compute_metrics()``
are all ready.
"""

from __future__ import annotations

from svvamp import Profile

from vote_simulation.models.rules.base import SvvampRuleWrapper


class ScoreBasedRuleWrapper(SvvampRuleWrapper):
    """Mixin for svvamp rules that expose a 1-D ``scores_`` array.

    The subclass contract is minimal:

    1. Set ``self.profile_`` (:class:`svvamp.Profile`).
    2. Set ``self._inner`` (the executed svvamp rule instance, i.e. the result
       of ``RuleXxx(...)(profile)``).
    3. Call ``self._init_score_based()`` and assign the result to
       ``self.cowinners_``.

    Everything else — ``cowinner_indices_``, ``compute_metrics()``, label
    resolution — is inherited from :class:`~vote_simulation.models.rules.base.SvvampRuleWrapper`.

    Parameters
    ----------
    (set on the instance, not passed to ``__init__``)

    profile_ : svvamp.Profile
    _inner   : executed svvamp rule (has a ``scores_`` 1-D attribute)
    """

    _inner: object  # executed svvamp rule instance

    def _init_score_based(self) -> list[str]:
        """Compute and return co-winners from ``_inner.scores_``.

        Delegates to :meth:`~vote_simulation.models.rules.base.SvvampRuleWrapper._max_score_cowinners`,
        which also sets ``cowinner_indices_`` as a side-effect.

        Returns
        -------
        list[str]
            Candidate labels tied at the maximum score.

        Raises
        ------
        AttributeError
            If ``_inner`` or ``_inner.scores_`` is not set.
        """
        scores = getattr(self._inner, "scores_", None)
        if scores is None:
            raise AttributeError(
                f"{type(self._inner).__name__} does not expose a 'scores_' attribute. "
                "ScoreBasedRuleWrapper requires a 1-D scores_ array."
            )
        return self._max_score_cowinners(scores)

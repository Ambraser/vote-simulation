"""Generic wrapper for svvamp elimination-round rules.

Rules like **Baldwin** and **Nanson** work by iteratively eliminating the
weakest candidate(s) until one survives.  Their ``scores_`` is a **2-D**
array ``[round, candidate]`` where ``np.inf`` marks an already-eliminated
candidate.

Co-winner semantics
-------------------
The co-winners are **all candidates that survive to the final round**, i.e.
all ``c`` such that ``scores_[-1, c] != np.inf`` (finite score in the last
elimination round).

In practice this means:

* **Single winner** — the normal case: one candidate has a finite score in
  the last row.
* **Tied final round** — multiple candidates share the same minimum Borda
  score in the penultimate round so none gets eliminated; all of them are
  co-winners.

Known rules that fit this pattern
-----------------------------------
* **Baldwin** — iterative Borda elimination (lowest Borda score eliminated
  each round, ties broken by highest index).
* **Nanson** — eliminates all candidates below the average Borda score each
  round.

Usage
------
Subclass :class:`EliminationBasedRuleWrapper`, set ``profile_`` and
``_inner``, then call :meth:`_init_elimination_based`::

    class BaldwinResult(EliminationBasedRuleWrapper):
        def __init__(self, profile: Profile) -> None:
            self.profile_ = profile
            self._inner = RuleBaldwin()(profile)
            self.cowinners_ = self._init_elimination_based()
"""

from __future__ import annotations

import numpy as np

from vote_simulation.models.rules.base import SvvampRuleWrapper


class EliminationBasedRuleWrapper(SvvampRuleWrapper):
    """Mixin for svvamp rules that expose a 2-D ``scores_[round, candidate]``.

    The subclass contract is identical to :class:`ScoreBasedRuleWrapper`:

    1. Set ``self.profile_``.
    2. Set ``self._inner`` (executed svvamp rule instance).
    3. Call ``self._init_elimination_based()`` and assign the result to
       ``self.cowinners_``.

    The method reads ``_inner.scores_`` (shape ``[n_rounds, n_candidates]``),
    looks at the **last round** (``scores_[-1, :]``), and returns all
    candidates with a finite score — those that survived all elimination
    rounds.
    """

    _inner: object

    def _init_elimination_based(self) -> list[str]:
        """Compute co-winners from the last elimination round of ``_inner.scores_``.

        Returns
        -------
        list[str]
            Labels of all candidates surviving to the final round.

        Raises
        ------
        AttributeError
            If ``_inner.scores_`` is missing or not 2-D.
        ValueError
            If ``scores_`` has an unexpected shape.
        """
        scores = getattr(self._inner, "scores_", None)
        if scores is None:
            raise AttributeError(
                f"{type(self._inner).__name__} does not expose a 'scores_' attribute."
            )

        arr = np.asarray(scores, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"EliminationBasedRuleWrapper expects a 2-D scores_ array "
                f"(shape [n_rounds, n_candidates]), got shape {arr.shape}. "
                "Use ScoreBasedRuleWrapper for 1-D score arrays."
            )

        # Last row: surviving candidates have a finite (non-inf) score.
        # Co-winners = those tied at the MAXIMUM finite score.
        # (Replace inf with -inf so eliminated candidates are never the max.)
        last_round = arr[-1, :]
        last_round_clipped = np.where(np.isfinite(last_round), last_round, -np.inf)
        best = last_round_clipped.max()
        survivors = np.flatnonzero(np.isclose(last_round_clipped, float(best)))
        return self._resolve_cowinners(survivors)

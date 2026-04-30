"""Shared base class for svvamp rule wrappers.

Every per-rule module (``rule_copeland.py``, ``rule_approval.py``, …) inherits
from :class:`SvvampRuleWrapper` to get uniform label-resolution helpers and a
consistent ``cowinners_`` interface.

The design contract is:

* The subclass **must** set ``self.profile_`` before calling helpers.
* The subclass **must** populate ``self.cowinners_`` (typically in ``__init__``).
* ``_labels_for(indices)`` converts a numpy index array to candidate-label strings.
* ``_resolve_cowinners(indices)`` stores ``cowinner_indices_`` *and* returns labels —
  the canonical hook every subclass should use instead of ``_labels_for`` directly.
* ``_max_score_cowinners(scores)`` returns labels for all candidates tied at the
  maximum of a 1-D score array — covers the vast majority of score-based rules.
* ``compute_metrics()`` returns a :class:`~vote_simulation.models.rules.winner_metrics.WinnerMetrics`
  dataclass for post-hoc analysis of the winning set.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.winner_metrics import WinnerMetrics, compute_winner_metrics


class SvvampRuleWrapper:
    """Base wrapper that exposes a uniform ``cowinners_`` attribute.

    Subclasses must:

    1. Assign ``self.profile_`` (a :class:`svvamp.Profile`) before using helpers.
    2. Call ``self._resolve_cowinners(indices)`` and assign the result to
       ``self.cowinners_`` in their ``__init__``.

    Helpers
    -------
    _labels_for(indices)
        Resolve an integer index array to candidate-label strings.
    _resolve_cowinners(indices)
        Store ``cowinner_indices_`` and return the corresponding labels.
        **Preferred** over calling ``_labels_for`` directly.
    _max_score_cowinners(scores)
        Shorthand for score-based rules: resolves all labels tied at ``max(scores)``
        and stores ``cowinner_indices_`` as a side-effect.
    compute_metrics()
        Return a :class:`~vote_simulation.models.rules.winner_metrics.WinnerMetrics`
        instance computed from ``cowinner_indices_`` and ``profile_``.
    """

    profile_: Profile
    cowinners_: list[str]
    cowinner_indices_: np.ndarray  # 1-D int array, set by _resolve_cowinners

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _labels_for(self, indices: np.ndarray) -> list[str]:
        """Convert a numpy array of candidate indices to their string labels."""
        labels = getattr(self.profile_, "labels_candidates", None)
        if labels is not None:
            return [str(labels[int(i)]) for i in indices]
        return [str(int(i)) for i in indices]

    def _resolve_cowinners(self, indices: np.ndarray) -> list[str]:
        """Store *cowinner_indices_* and return the corresponding label strings.

        This is the **canonical** hook subclasses should call instead of
        ``_labels_for`` directly so that ``cowinner_indices_`` is always kept in
        sync with ``cowinners_``.

        Parameters
        ----------
        indices:
            1-D integer array of co-winner candidate indices (0-based).
        """
        self.cowinner_indices_ = np.asarray(indices, dtype=int)
        return self._labels_for(self.cowinner_indices_)

    def _max_score_cowinners(self, scores: np.ndarray) -> list[str]:
        """Return labels of all candidates sharing the maximum score.

        This covers most score-based rules (Borda, Copeland, Plurality, Approval…).
        Uses ``numpy.isclose`` for float scores, exact equality for integers.
        Stores ``cowinner_indices_`` as a side-effect via :meth:`_resolve_cowinners`.

        Parameters
        ----------
        scores:
            1-D array of per-candidate scores.
        """
        arr = np.asarray(scores)
        # print(arr)
        # print(self._inner.scores_)
        # self._inner.demo_results_()
        best = arr.max()
        if np.issubdtype(arr.dtype, np.integer):
            tied = np.flatnonzero(arr == best)
        else:
            tied = np.flatnonzero(np.isclose(arr.astype(float), float(best)))
        return self._resolve_cowinners(tied)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> WinnerMetrics:
        """Compute winner metrics for the current election outcome.

        Returns
        -------
        WinnerMetrics
            Immutable dataclass containing social acceptability, utility
            statistics, rank statistics, first/last frequencies and tie info.

        Raises
        ------
        AttributeError
            If ``cowinner_indices_`` has not been set yet (i.e. the subclass did
            not call :meth:`_resolve_cowinners` or :meth:`_max_score_cowinners`).
        """
        if not hasattr(self, "cowinner_indices_"):
            raise AttributeError(
                "cowinner_indices_ is not set. "
                "Make sure the subclass calls _resolve_cowinners() or "
                "_max_score_cowinners() during __init__."
            )
        return compute_winner_metrics(self.profile_, self.cowinner_indices_)

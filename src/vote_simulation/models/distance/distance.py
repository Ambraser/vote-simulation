"""Distance metrics for comparing sets of winners.

The base class`Distance` class is a **stateless strategy**: instantiate once,
then call meth`compute` as many times as needed.  This avoids creating N²
objects when building a pairwise distance matrix.

Subclass it and override `compute` to define new metrics.
"""

from __future__ import annotations


class Distance:
    """Abstract base for pairwise winner-set distance metrics.

    Subclasses must override method `compute`.
    """

    def compute(self, winners1: frozenset[str], winners2: frozenset[str]) -> float:
        """Return the distance between two winner sets.

        Args:
            winners1: Winner set of rule 1.
            winners2: Winner set of rule 2.

        Returns:
            A float in [0, 1] where 0 means identical and 1 means maximally different.
        """
        raise NotImplementedError("Subclasses must implement compute().")


class BinaryDistance(Distance):
    """Simple 0/1 distance: 0 when the sets are equal, 1 otherwise."""

    def compute(self, winners1: frozenset[str], winners2: frozenset[str]) -> float:
        return 0.0 if winners1 == winners2 else 1.0


class JaccardDistance(Distance):
    """Jaccard distance: ``1 - |intersection| / |union|``.

    Ranges from 0 (identical sets) to 1 (disjoint sets).
    Returns 0 when both sets are empty.
    """

    def compute(self, winners1: frozenset[str], winners2: frozenset[str]) -> float:
        intersection = len(winners1 & winners2)
        union = len(winners1 | winners2)
        if union == 0:
            return 0.0
        return 1.0 - intersection / union

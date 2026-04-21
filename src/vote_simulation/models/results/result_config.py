"""
Defines the :class:`ResultConfig` dataclass for describing simulation contexts.

This class encapsulates the parameters that define a simulation run as such :

- the generation models used,
- the number of voters and candidates,
- the rules applied,
- and the number of iterations.

The class provides for adding rules to existing configs, merging configs,
and generating labels for results based on their parameters.
"""

from __future__ import annotations

from builtins import max as builtins_max
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ResultConfig:
    """Describes the simulation context attached to a result.

    Supports single-valued **and** multi-valued configurations to
    express metadata.

    All collection fields use :class:`frozenset` for immutability and
    light membership checks.
    """

    gen_models: frozenset[str] = field(default_factory=frozenset)
    n_voters: frozenset[int] = field(default_factory=frozenset)
    n_candidates: frozenset[int] = field(default_factory=frozenset)
    rules_codes: frozenset[str] = field(default_factory=frozenset)
    n_iterations: int = 0

    # Factories

    @staticmethod
    def single(
        gen_model: str = "",
        n_voters: int = 0,
        n_candidates: int = 0,
        n_iterations: int = 0,
        rules_codes: list[str] | None = None,
    ) -> ResultConfig:
        """Create a config for a single (model, voters, candidates) combo."""
        return ResultConfig(
            gen_models=frozenset({gen_model}) if gen_model else frozenset(),
            n_voters=frozenset({n_voters}) if n_voters else frozenset(),
            n_candidates=frozenset({n_candidates}) if n_candidates else frozenset(),
            rules_codes=frozenset(rules_codes) if rules_codes else frozenset(),
            n_iterations=n_iterations,
        )

    # Merge / combine

    def merge(self, other: ResultConfig) -> ResultConfig:
        """Return the union of two configs (idempotent & commutative)."""
        return ResultConfig(
            gen_models=self.gen_models | other.gen_models,
            n_voters=self.n_voters | other.n_voters,
            n_candidates=self.n_candidates | other.n_candidates,
            rules_codes=self.rules_codes | other.rules_codes,
            n_iterations=builtins_max(self.n_iterations, other.n_iterations),
        )

    def base_config(self) -> ResultConfig:
        """Return a copy with rules_codes cleared (for cache keys based on data params only)."""
        return ResultConfig(
            gen_models=self.gen_models,
            n_voters=self.n_voters,
            n_candidates=self.n_candidates,
            rules_codes=frozenset(),
            n_iterations=self.n_iterations,
        )

    def matches_base(self, other: ResultConfig) -> bool:
        """Check if two configs have identical base parameters (ignoring rules_codes)."""
        return (
            self.gen_models == other.gen_models
            and self.n_voters == other.n_voters
            and self.n_candidates == other.n_candidates
            and self.n_iterations == other.n_iterations
        )

    # Labels

    @property
    def label(self) -> str:
        """Base label suitable for directory / file names (excludes rules).

        Used for cache keys and data organization directories.
        Format depends on how many values are set::

            "UNI_v101_c3"          (single model, voters, candidates)
            "IC_UNI_v11_101_c3_14" (multiple values)

        When n_iterations is set, appends ``_i{n_iterations}``.
        """
        models = "_".join(sorted(self.gen_models)) or "UNKNOWN"
        voters = "_".join(str(v) for v in sorted(self.n_voters)) or "0"
        candidates = "_".join(str(c) for c in sorted(self.n_candidates)) or "0"
        base = f"{models}_v{voters}_c{candidates}"
        if self.n_iterations:
            base += f"_i{self.n_iterations}"
        return base

    @property
    def label_with_rules(self) -> str:
        """Full label including rules codes (for complete identification).

        Format: ``{base_label}_r{rules_joined}``
        """
        base = self.label
        if self.rules_codes:
            rules = "_".join(sorted(self.rules_codes))
            return f"{base}_r{rules}"
        return base

    @property
    def description(self) -> str:
        """Human-readable description for plot titles.

        Automatically switches between singular and plural phrasing depending
        on how many distinct values are present.
        """
        parts: list[str] = []
        if self.gen_models:
            if len(self.gen_models) == 1:
                parts.append(next(iter(self.gen_models)))
            else:
                parts.append(f"Models: {', '.join(sorted(self.gen_models))}")
        if self.n_voters:
            if len(self.n_voters) == 1:
                parts.append(f"{next(iter(self.n_voters))} voters")
            else:
                parts.append(f"Voters: {', '.join(str(v) for v in sorted(self.n_voters))}")
        if self.n_candidates:
            if len(self.n_candidates) == 1:
                parts.append(f"{next(iter(self.n_candidates))} cand.")
            else:
                parts.append(f"Candidates: {', '.join(str(c) for c in sorted(self.n_candidates))}")
        return " · ".join(parts) if parts else ""

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, str]:
        """Serialize to a ``{key: csv_string}`` mapping."""
        return {
            "gen_models": ",".join(sorted(self.gen_models)),
            "n_voters": ",".join(str(v) for v in sorted(self.n_voters)),
            "n_candidates": ",".join(str(c) for c in sorted(self.n_candidates)),
            "n_iterations": str(self.n_iterations),
            "rules_codes": ",".join(sorted(self.rules_codes)),
        }

    @staticmethod
    def from_dict(data: dict[str, str]) -> ResultConfig:
        """Deserialize from a ``{key: csv_string}`` mapping."""
        gen_models = frozenset(m for m in data.get("gen_models", "").split(",") if m)
        n_voters = frozenset(int(v) for v in data.get("n_voters", "").split(",") if v)
        n_candidates = frozenset(int(c) for c in data.get("n_candidates", "").split(",") if c)
        n_iterations = int(data["n_iterations"]) if data.get("n_iterations") else 0
        rules_codes = frozenset(c for c in data.get("rules_codes", "").split(",") if c)
        return ResultConfig(
            gen_models=gen_models,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_iterations=n_iterations,
            rules_codes=rules_codes,
        )

    def __bool__(self) -> bool:
        return bool(self.gen_models or self.n_voters or self.n_candidates or self.n_iterations or self.rules_codes)

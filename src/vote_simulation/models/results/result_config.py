from __future__ import annotations

from builtins import max as builtins_max
from builtins import min as builtins_min

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ResultConfig:
    """Describes the simulation context attached to a result.

    Supports single-valued **and** multi-valued configurations to
    express metadata.

    All collection fields use :class:`frozenset` for immutability and
    O(1) membership checks.
    """

    gen_models: frozenset[str] = field(default_factory=frozenset)
    n_voters: frozenset[int] = field(default_factory=frozenset)
    n_candidates: frozenset[int] = field(default_factory=frozenset)
    n_iterations: int = 0

    # -- Factories --------------------------------------------------------

    @staticmethod
    def single(
        gen_model: str = "",
        n_voters: int = 0,
        n_candidates: int = 0,
        n_iterations: int = 0,
    ) -> ResultConfig:
        """Create a config for a single (model, voters, candidates) combo."""
        return ResultConfig(
            gen_models=frozenset({gen_model}) if gen_model else frozenset(),
            n_voters=frozenset({n_voters}) if n_voters else frozenset(),
            n_candidates=frozenset({n_candidates}) if n_candidates else frozenset(),
            n_iterations=n_iterations,
        )

    # -- Merge / combine --------------------------------------------------

    def merge(self, other: ResultConfig) -> ResultConfig:
        """Return the union of two configs (idempotent & commutative)."""
        return ResultConfig(
            gen_models=self.gen_models | other.gen_models,
            n_voters=self.n_voters | other.n_voters,
            n_candidates=self.n_candidates | other.n_candidates,
            n_iterations=builtins_max(self.n_iterations, other.n_iterations),
        )

    # -- Labels -----------------------------------------------------------

    @property
    def label(self) -> str:
        """Short slug suitable for directory / file names.

        Example: ``"VMF_HC_v101_c3"`` or ``"IC_UNI_v11_101_c3_14"``.
        """
        models = "_".join(sorted(self.gen_models)) or "UNKNOWN"
        voters = "_".join(str(v) for v in sorted(self.n_voters)) or "0"
        candidates = "_".join(str(c) for c in sorted(self.n_candidates)) or "0"
        base = f"{models}_v{voters}_c{candidates}"
        if self.n_iterations:
            base += f"_i{self.n_iterations}"
        return base

    @property
    def description(self) -> str:
        """Human‑readable description for plot titles.

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
        }

    @staticmethod
    def from_dict(data: dict[str, str]) -> ResultConfig:
        """Deserialize from a ``{key: csv_string}`` mapping."""
        gen_models = frozenset(m for m in data.get("gen_models", "").split(",") if m)
        n_voters = frozenset(int(v) for v in data.get("n_voters", "").split(",") if v)
        n_candidates = frozenset(int(c) for c in data.get("n_candidates", "").split(",") if c)
        n_iterations = int(data["n_iterations"]) if data.get("n_iterations") else 0
        return ResultConfig(
            gen_models=gen_models,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_iterations=n_iterations,
        )

    def __bool__(self) -> bool:
        return bool(self.gen_models or self.n_voters or self.n_candidates or self.n_iterations)


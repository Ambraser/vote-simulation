"""Rule registry and rule-related helpers."""

# Import rule modules so their register_rule() calls are executed on import.
from vote_simulation.models.rules import rule_approval as _rule_approval  # noqa: F401
from vote_simulation.models.rules import rule_copeland as _rule_copeland  # noqa: F401

from vote_simulation.models.rules.registry import (
    RuleResult,
    get_all_rules_codes,
    get_rule_builder,
    make_rule_builder,
    register_rule,
)
from vote_simulation.models.rules.winner_metrics import WinnerMetrics, compute_winner_metrics

__all__ = [
    "RuleResult",
    "WinnerMetrics",
    "compute_winner_metrics",
    "get_rule_builder",
    "make_rule_builder",
    "register_rule",
    "get_all_rules_codes",
]

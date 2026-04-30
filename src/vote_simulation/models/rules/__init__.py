"""Rule registry and rule-related helpers."""

# Import rule modules so their register_rule() calls are executed on import.
from vote_simulation.models.rules import rule_approval as _rule_approval  # noqa: F401
from vote_simulation.models.rules import rule_baldwin as _rule_baldwin  # noqa: F401
from vote_simulation.models.rules import rule_black as _rule_black  # noqa: F401
from vote_simulation.models.rules import rule_borda as _rule_borda  # noqa: F401
from vote_simulation.models.rules import rule_bucklin as _rule_bucklin  # noqa: F401
from vote_simulation.models.rules import rule_coombs as _rule_coombs  # noqa: F401
from vote_simulation.models.rules import rule_copeland as _rule_copeland  # noqa: F401
from vote_simulation.models.rules import rule_icrv as _rule_icrv  # noqa: F401
from vote_simulation.models.rules import rule_irv as _rule_irv  # noqa: F401
from vote_simulation.models.rules import rule_iterated_bucklin as _rule_iterated_bucklin  # noqa: F401
from vote_simulation.models.rules import rule_k_approval as _rule_k_approval  # noqa: F401
from vote_simulation.models.rules import rule_kemeny as _rule_kemeny  # noqa: F401
from vote_simulation.models.rules import rule_kim_roush as _rule_kim_roush  # noqa: F401
from vote_simulation.models.rules import rule_majority_judgment as _rule_majority_judgment  # noqa: F401
from vote_simulation.models.rules import rule_maximin as _rule_maximin  # noqa: F401
from vote_simulation.models.rules import rule_nanson as _rule_nanson  # noqa: F401
from vote_simulation.models.rules import rule_plurality as _rule_plurality  # noqa: F401
from vote_simulation.models.rules import rule_range_voting as _rule_range_voting  # noqa: F401
from vote_simulation.models.rules import rule_schulze as _rule_schulze  # noqa: F401
from vote_simulation.models.rules import rule_slater as _rule_slater  # noqa: F401
from vote_simulation.models.rules import rule_split_cycle as _rule_split_cycle  # noqa: F401
from vote_simulation.models.rules import rule_star as _rule_star  # noqa: F401
from vote_simulation.models.rules import rule_tideman as _rule_tideman  # noqa: F401
from vote_simulation.models.rules import rule_two_round as _rule_two_round  # noqa: F401
from vote_simulation.models.rules import rule_veto as _rule_veto  # noqa: F401
from vote_simulation.models.rules import rule_woodall as _rule_woodall  # noqa: F401
from vote_simulation.models.rules import rule_young as _rule_young  # noqa: F401
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

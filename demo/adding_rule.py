"""Small demo showing how to add a custom voting rule in the registry.

This example registers an Approval rule with threshold 0.8

Note that threshold rule in general is implemented as strictly greater, equals values do not count.
"""

from __future__ import annotations

from typing import Any, cast

from svvamp import RuleApproval

from vote_simulation.models.rules.registry import (
    get_rule_builder,
    make_rule_builder,
    register_rule,
)


def main() -> None:
    register_rule("AP_T8", make_rule_builder(lambda profile: RuleApproval(approval_threshold=0.8)(profile)))

    ballots = [
        {"Alice": 1.0, "Bob": 0.9, "Chloé": 0.2},
        {"Alice": 0.85, "Bob": 0.7, "Chloé": 0.9},
        {"Alice": 0.3, "Bob": 0.95, "Chloé": 0.81},
        {"Alice": 0.82, "Bob": 0.6, "Chloé": 0.9},
    ]

    rule_result = get_rule_builder("AP_T8")(ballots, None)
    approval_rule = cast(Any, rule_result)
    winner_label = rule_result.cowinners_[0]

    print("Rule code:", "AP_T8")
    print("Approval threshold:", 0.8)
    print("Winner:", winner_label)
    print("Score vector:", approval_rule.scores_)
    print("Co-winners:", rule_result.cowinners_)


if __name__ == "__main__":
    main()

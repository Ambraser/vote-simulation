# Rules

All rules expose a `cowinners_` attribute. They are integrated via the `rule_codes` key in `config/simulation.toml`.

## Rule codes

### Main rules

| Code | Rule name | Brief description |
|---|---|---|
| PLU1 | [Plurality](https://francois-durand.github.io/svvamp/reference/rules/rule_plurality.html) | Candidate with the most first-place votes wins |
| PLU2 | [Two-Round Plurality (Runoff)](https://francois-durand.github.io/svvamp/reference/rules/rule_two_round.html) | Top two candidates compete in a second-round runoff |
| HARE | [Hare / Instant-Runoff Voting](https://francois-durand.github.io/svvamp/reference/rules/rule_irv.html) | Eliminate lowest candidates iteratively using ranked ballots |
| BUCK_R | [Bucklin Voting](https://francois-durand.github.io/svvamp/reference/rules/rule_bucklin.html) | Selects candidate reaching majority at lowest rank level |
| BUCK_I | [Iterated Bucklin](https://francois-durand.github.io/svvamp/reference/rules/rule_iterated_bucklin.html) | Repeated Bucklin elimination process |
| COPE | [Copeland's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_copeland.html) | Score based on pairwise wins minus pairwise losses |
| SCHU | [Schulze Method](https://francois-durand.github.io/svvamp/reference/rules/rule_schulze.html) | Strongest path method on pairwise comparisons |
| MMIN | [Maximin (Minimax)](https://francois-durand.github.io/svvamp/reference/rules/rule_maximin.html) | Candidate with the strongest worst pairwise defeat wins |
| BLAC | [Black's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_black.html) | Condorcet winner if one exists, otherwise Borda winner |
| KIMR | [Kim-Roush](https://francois-durand.github.io/svvamp/reference/rules/rule_kim_roush.html) | Each round, candidates with veto score strictly below average are simultaneously eliminated |
| WOOD | [Woodall's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_woodall.html) | Perform IRV on the Smith set |
| YOUN | [Young's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_young.html) | Elect the candidate who can become a Condorcet winner by removing the fewest voters |
| TIDE | [Tideman's Ranked Pairs](https://francois-durand.github.io/svvamp/reference/rules/rule_tideman.html) | Lock pairwise victories from strongest to weakest, avoiding cycles |
| BORD | [Borda Count](https://francois-durand.github.io/svvamp/reference/rules/rule_borda.html) | Score candidates by ranked positions |
| COOM | [Coombs' Method](https://francois-durand.github.io/svvamp/reference/rules/rule_coombs.html) | IRV variant eliminating the most disliked candidate each round |
| NANS | [Nanson's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_nanson.html) | Iterative elimination of candidates below the Borda average |
| BALD | [Baldwin's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_baldwin.html) | Iterative elimination of the lowest Borda score |
| MJ | [Majority Judgment](https://francois-durand.github.io/svvamp/reference/rules/rule_majority_judgment.html) | Elect the candidate with the highest median grade |
| RV | [Range Voting](https://francois-durand.github.io/svvamp/reference/rules/rule_range_voting.html) | Candidate with the highest average score wins |
| STAR | [STAR Voting](https://francois-durand.github.io/svvamp/reference/rules/rule_star.html) | Score then automatic runoff between the top two scored candidates |
| VETO | [Veto (Anti-Plurality)](https://francois-durand.github.io/svvamp/reference/rules/rule_veto.html) | Candidate with the fewest last-place votes wins |
| AP_T0GE | [Approval Voting (utility ≥ 0)](https://francois-durand.github.io/svvamp/reference/rules/rule_approval.html) | Approve all candidates with non-negative utility |
| AP_T05 … AP_T09 | [Approval Voting (threshold)](https://francois-durand.github.io/svvamp/reference/rules/rule_approval.html) | Approve candidates above the given utility threshold (0.05, 0.06 … 0.9) |
| AP_K2 … AP_K12 | [K-Approval](https://francois-durand.github.io/svvamp/reference/rules/rule_k_approval.html) | Approve a fixed number of top candidates (K = 2 … 12) |
| AP_KRP | K-Approval (K = random Poisson) | K drawn randomly from a Poisson distribution with λ = c/3 |
| L1DV … L6DV | Lp Distance Voting (p = 1 … 6) | Weighted Lp-norm distance-based voting rules |
| DEPF | Depth Function (default) | Winner selection based on statistical data depth |
| DEPF_EXACT | Depth Function (exact) | Exact computation of the depth function winner |
| DEPF_GREEDY | Depth Function (greedy) | Greedy approximation of the depth function winner |
| DODG_C | [Dodgson's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_dodgson.html) | Minimum swaps needed for a candidate to become a Condorcet winner |
| DODG_S | [Dodgson's Method (score variant)](https://francois-durand.github.io/svvamp/reference/rules/rule_dodgson.html) | Same, using the svvamp score-based implementation |
| COND | Condorcet | Weak Condorcet winner(s) elected; all tied otherwise |
| RANDOM_DICTATOR | Random Dictator | A random voter is picked and their top candidate is elected |
| RANDOM_WINNER | Random Winner | A random candidate is picked uniformly |

### Additional registered rules

These rules are also available in the registry but less commonly used.

| Code | Rule name |
|---|---|
| KEME | [Kemeny-Young](https://francois-durand.github.io/svvamp/reference/rules/rule_kemeny.html) — ⚠️ high computational cost |
| KEME_LAZY | Kemeny-Young (lazy approximation) |
| SLAT | [Slater's Method](https://francois-durand.github.io/svvamp/reference/rules/rule_slater.html) — ⚠️ high computational cost |
| SPCY | [Split Cycle](https://francois-durand.github.io/svvamp/reference/rules/rule_split_cycle.html) |
| RPAR | [Ranked Pairs](https://francois-durand.github.io/svvamp/reference/rules/rule_ranked_pairs.html) |
| SIRV | [Smith-IRV](https://francois-durand.github.io/svvamp/reference/rules/rule_smith_irv.html) |
| EXHB | [Exhaustive Ballot](https://francois-durand.github.io/svvamp/reference/rules/rule_exhaustive_ballot.html) |
| ICRV | [ICRV](https://francois-durand.github.io/svvamp/reference/rules/rule_icrv.html) |
| IRVA | [IRV Average](https://francois-durand.github.io/svvamp/reference/rules/rule_irv_average.html) |
| IRVD | [IRV Duels](https://francois-durand.github.io/svvamp/reference/rules/rule_irv_duels.html) |
| CAIR | [Condorcet-Abs-IRV](https://francois-durand.github.io/svvamp/reference/rules/rule_condorcet_abs_irv.html) |
| CVIR | [Condorcet-Vtb-IRV](https://francois-durand.github.io/svvamp/reference/rules/rule_condorcet_vtb_irv.html) |
| CSUM | [Condorcet Sum Defeats](https://francois-durand.github.io/svvamp/reference/rules/rule_condorcet_sum_defeats.html) |
| APLU | Approval (all candidates approved) |
| AP_T | Approval Voting (threshold = 0.7, alias) |
| AP_K | K-Approval (K = 2, alias) |
| AP_T05GE | Approval Voting (utility ≥ 0.5) |
| AP_K_POISSON | K-Approval (K = random Poisson, alias for AP_KRP) |

## How to add a rule

Use the public registry helpers to register custom rules:

```python
from svvamp import RuleApproval

from vote_simulation.models.rules.registry import get_rule_builder, make_rule_builder, register_rule


register_rule(
    "AP_T8",
    make_rule_builder(lambda profile: RuleApproval(approval_threshold=0.8)(profile)),
)

ballots = [
    {"Alice": 1.0, "Bob": 0.9, "Chloe": 0.2},
    {"Alice": 0.85, "Bob": 0.7, "Chloe": 0.9},
]

result = get_rule_builder("AP_T8")(ballots)
print(result.cowinners_)
```

::: vote_simulation.models.rules.registry

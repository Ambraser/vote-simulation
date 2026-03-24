# Table des codes de règles

All rules expose a `cowinners_` attribute. They are integrated via the `rule_codes` key in `config/simulation.toml`.

| code | rule | doc SVVAMP | status |
|---|---|---|---|
| AP_T | Approval (threshold 0.7) | [rule_approval](https://francois-durand.github.io/svvamp/reference/rules/rule_approval.html) | Tested & validated |
| AP_K | K-Approval (`k=2`) | [rule_k_approval](https://francois-durand.github.io/svvamp/reference/rules/rule_k_approval.html) | Tested & validated |
| BALD | Baldwin | [rule_baldwin](https://francois-durand.github.io/svvamp/reference/rules/rule_baldwin.html) | Tested & validated |
| BLAC | Black | [rule_black](https://francois-durand.github.io/svvamp/reference/rules/rule_black.html) | Tested & validated |
| BORD | Borda | [rule_borda](https://francois-durand.github.io/svvamp/reference/rules/rule_borda.html) | Tested & validated |
| BUCK_I | Iterated Bucklin | [rule_iterated_bucklin](https://francois-durand.github.io/svvamp/reference/rules/rule_iterated_bucklin.html) | Tested & validated |
| BUCK_R | Bucklin | [rule_bucklin](https://francois-durand.github.io/svvamp/reference/rules/rule_bucklin.html) | Tested & validated |
| CAIR | Condorcet Abs IRV | [rule_condorcet_abs_irv](https://francois-durand.github.io/svvamp/reference/rules/rule_condorcet_abs_irv.html) | Tested & validated |
| COOM | Coombs | [rule_coombs](https://francois-durand.github.io/svvamp/reference/rules/rule_coombs.html) | Tested & validated |
| COPE | Copeland | [rule_copeland](https://francois-durand.github.io/svvamp/reference/rules/rule_copeland.html) | Tested & validated |
| CSUM | Condorcet Sum Defeats | [rule_condorcet_sum_defeats](https://francois-durand.github.io/svvamp/reference/rules/rule_condorcet_sum_defeats.html) | Tested & validated |
| CVIR | Condorcet Vtb IRV | [rule_condorcet_vtb_irv](https://francois-durand.github.io/svvamp/reference/rules/rule_condorcet_vtb_irv.html) | Tested & validated |
| DODG_C | Dodgson (C) | |❌ Not implemented |
| DODG_S | Dodgson (S) | [rule_dodgson](https://francois-durand.github.io/svvamp/reference/rules/rule_dodgson.html) | may be `nan` |
| EXHB | Exhaustive Ballot | [rule_exhaustive_ballot](https://francois-durand.github.io/svvamp/reference/rules/rule_exhaustive_ballot.html) | Tested & validated |
| HARE | IRV / Hare | [rule_irv](https://francois-durand.github.io/svvamp/reference/rules/rule_irv.html) | Tested & validated |
| ICRV | ICRV | [rule_icrv](https://francois-durand.github.io/svvamp/reference/rules/rule_icrv.html) | ⚠️ No dedicated test |
| IRV | IRV (alias de `HARE`) |  see HARE |
| IRVA | IRV Average | [rule_irv_average](https://francois-durand.github.io/svvamp/reference/rules/rule_irv_average.html) | ⚠️ No dedicated test |
| IRVD | IRV Duels | [rule_irv_duels](https://francois-durand.github.io/svvamp/reference/rules/rule_irv_duels.html) | ⚠️ No dedicated test |
| KEME | Kemeny | [rule_kemeny](https://francois-durand.github.io/svvamp/reference/rules/rule_kemeny.html) | ⚠️ No dedicated test — high computational time |
| KIMR | Kim-Roush | [rule_kim_roush](https://francois-durand.github.io/svvamp/reference/rules/rule_kim_roush.html) | Tested & validated |
| L4VD | L4VD | — | ❌ Not implemented |
| MJ | Majority Judgment | [rule_majority_judgment](https://francois-durand.github.io/svvamp/reference/rules/rule_majority_judgment.html) | Tested & validated |
| MMAX | Maximin | [rule_maximin](https://francois-durand.github.io/svvamp/reference/rules/rule_maximin.html) | Tested & validated |
| NANS | Nanson | [rule_nanson](https://francois-durand.github.io/svvamp/reference/rules/rule_nanson.html) | Tested & validated |
| PLU1 | Plurality | [rule_plurality](https://francois-durand.github.io/svvamp/reference/rules/rule_plurality.html) | Tested & validated |
| PLU2 | Two-Round | [rule_two_round](https://francois-durand.github.io/svvamp/reference/rules/rule_two_round.html) | Tested & validated |
| RPAR | Ranked Pairs | [rule_ranked_pairs](https://francois-durand.github.io/svvamp/reference/rules/rule_ranked_pairs.html) | Tested & validated |
| RV | Range Voting | [rule_range_voting](https://francois-durand.github.io/svvamp/reference/rules/rule_range_voting.html) | Tested & validated |
| SCHU | Schulze | [rule_schulze](https://francois-durand.github.io/svvamp/reference/rules/rule_schulze.html) | Tested & validated  |
| SIRV | Smith IRV | [rule_smith_irv](https://francois-durand.github.io/svvamp/reference/rules/rule_smith_irv.html) | Tested & validated |
| SLAT | Slater | [rule_slater](https://francois-durand.github.io/svvamp/reference/rules/rule_slater.html) | ⚠️ No dedicated test -- computational time too long |
| SPCY | Split Cycle | [rule_split_cycle](https://francois-durand.github.io/svvamp/reference/rules/rule_split_cycle.html) | Tested & validated |
| STAR | STAR | [rule_star](https://francois-durand.github.io/svvamp/reference/rules/rule_star.html) | Tested & validated |
| TIDE | Tideman | [rule_tideman](https://francois-durand.github.io/svvamp/reference/rules/rule_tideman.html) | Tested & validated |
| VETO | Veto (Anti-plurality) | [rule_veto](https://francois-durand.github.io/svvamp/reference/rules/rule_veto.html) | Tested & validated |
| WOOD | Woodall | [rule_woodall](https://francois-durand.github.io/svvamp/reference/rules/rule_woodall.html) | Tested & validated |
| YOUN | Young | [rule_young](https://francois-durand.github.io/svvamp/reference/rules/rule_young.html) | Tested & validated |


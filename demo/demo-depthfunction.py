"""Démonstration du Depth Function voting rule (Goibert et al., AISTATS 2022).

Ce script illustre le fonctionnement de la règle DEPF sur plusieurs profils :

  1. Profil avec un vainqueur de Condorcet clair.
  2. Profil avec un cycle de Condorcet (paradoxe de Condorcet).
  3. Grand profil aléatoire (mode greedy, n_c > EXACT_THRESHOLD).
  4. Comparaison DEPF vs Kemeny et Borda sur un même profil.
  5. Profil généré avec un modèle d'urne Impartiale Culture (via vote_simulation).
"""

from __future__ import annotations

import numpy as np
import svvamp

from vote_simulation.models.rules import get_rule_builder
from vote_simulation.models.rules.rule_depthfunction import (
    EXACT_THRESHOLD,
    DepthFunctionResult,
)

SEP = "=" * 60


def show_result(result: DepthFunctionResult, label: str = "") -> None:
    """Affiche les informations clés d'un résultat DEPF."""
    if label:
        print(f"\n{label}")
        print("-" * len(label))
    print(f"  Co-gagnant(s) : {result.cowinners_}")
    print(f"  Scores de profondeur :")
    for c, s in enumerate(result.scores_):
        marker = " <-- MAX" if c in result.cowinner_indices_ else ""
        print(f"    Candidat {c} : {s:.4f}{marker}")
    metrics = result.compute_metrics()
    print(f"  Nb co-gagnants : {metrics.n_cowinners}")
    print(f"  Acceptabilité sociale : {metrics.social_acceptability:.2%}")


# ---------------------------------------------------------------------------
# 1. Vainqueur de Condorcet clair
# ---------------------------------------------------------------------------
print(SEP)
print("1. PROFIL AVEC VAINQUEUR DE CONDORCET")
print(SEP)
# Candidat 0 est préféré par tous à 1, et tous préfèrent 1 à 2.
prefs_condorcet = np.array([
    [0, 1, 2],
    [0, 1, 2],
    [0, 2, 1],
    [0, 1, 2],
])
profile_c = svvamp.Profile(preferences_rk=prefs_condorcet)
result_c = get_rule_builder("DEPF")(profile_c)
show_result(result_c, "Votes : 0>1>2, 0>1>2, 0>2>1, 0>1>2")
print(f"  -> Vainqueur de Condorcet attendu : candidat 0")

# ---------------------------------------------------------------------------
# 2. Paradoxe de Condorcet (cycle 0>1>2>0)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("2. PARADOXE DE CONDORCET (cycle)")
print(SEP)
# 0>1>2, 1>2>0, 2>0>1  → pas de vainqueur de Condorcet
prefs_cycle = np.array([
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
])
profile_cy = svvamp.Profile(preferences_rk=prefs_cycle)
result_cy = get_rule_builder("DEPF")(profile_cy)
show_result(result_cy, "Votes : 0>1>2 | 1>2>0 | 2>0>1")
print(f"  -> Distribution uniforme : scores tous égaux attendus")

# ---------------------------------------------------------------------------
# 3. Co-gagnants (égalité parfaite)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("3. PROFIL AVEC CO-GAGNANTS")
print(SEP)
# Symétrie exacte entre 0 et 1
prefs_tie = np.array([
    [0, 1, 2],
    [1, 0, 2],
    [0, 1, 2],
    [1, 0, 2],
])
profile_t = svvamp.Profile(preferences_rk=prefs_tie)
result_t = get_rule_builder("DEPF")(profile_t)
show_result(result_t, "Votes : 0>1>2 et 1>0>2 (équilibre parfait)")

# ---------------------------------------------------------------------------
# 4. Grand profil aléatoire (mode greedy, n_c > EXACT_THRESHOLD)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"4. GRAND PROFIL ALÉATOIRE  (n_c = {EXACT_THRESHOLD + 2}, mode greedy)")
print(SEP)
np.random.seed(0)
n_c_large = EXACT_THRESHOLD + 2
n_v_large = 200
prefs_large = np.argsort(np.random.rand(n_v_large, n_c_large), axis=1)
profile_l = svvamp.Profile(preferences_rk=prefs_large)
result_l = get_rule_builder("DEPF")(profile_l)
show_result(result_l, f"{n_v_large} votants, {n_c_large} candidats (approximation greedy)")

# ---------------------------------------------------------------------------
# 5. Comparaison DEPF / Kemeny / Borda sur le même profil
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("5. COMPARAISON DEPF vs KEMENY vs BORDA")
print(SEP)
np.random.seed(42)
prefs_cmp = np.argsort(np.random.rand(30, 6), axis=1)
profile_cmp = svvamp.Profile(preferences_rk=prefs_cmp)

result_depf  = get_rule_builder("DEPF")(profile_cmp)
result_kemeny = get_rule_builder("KEME")(profile_cmp)
result_borda  = get_rule_builder("BORD")(profile_cmp)

print(f"  Profil : 30 votants, 6 candidats (aléatoire seed=42)")
print(f"  DEPF   co-gagnant(s) : {result_depf.cowinners_}")
print(f"  Kemeny co-gagnant(s) : {result_kemeny.cowinners_}")
print(f"  Borda  co-gagnant(s) : {result_borda.cowinners_}")
print()
print("  Scores DEPF (profondeur du meilleur classement avec ce candidat en tête) :")
for c, s in enumerate(result_depf.scores_):
    print(f"    Candidat {c} : {s:.4f}")

# ---------------------------------------------------------------------------
# 6. Profil IC généré via vote_simulation
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("6. PROFIL IMPARTIAL CULTURE (via vote_simulation DataGenerator)")
print(SEP)
try:
    from vote_simulation.models.data_generation.data_instance import DataInstance

    di = DataInstance.from_generator("IC", 101, 5, seed=7)
    result_ic = get_rule_builder("DEPF")(di.profile)
    show_result(result_ic, "IC — 101 votants, 5 candidats, seed=7")
except Exception as exc:
    print(f"  (Erreur : {exc})")

print(f"\n{SEP}")
print("Démonstration terminée.")
print(SEP)

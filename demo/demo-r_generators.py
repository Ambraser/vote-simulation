"""Quick visual inspection of the R-based DDD_BETA and DDD_BETA_POLAR generators."""

import numpy as np

from vote_simulation.models.data_generation.from_r_registry import register_r_generators
from vote_simulation.models.data_generation.generator_registry import get_generator_builder

register_r_generators()

N_V = 10
N_C = 4
SEED = 42


def show_profile(name: str, n_v: int, n_c: int, seed: int) -> None:
    builder = get_generator_builder(name)
    profile = builder(n_v=n_v, n_c=n_c, seed=seed)

    ut: np.ndarray = np.asarray(profile.preferences_ut)  # (n_v, n_c)

    print(f"{'=' * 60}")
    print(f"  {name}  |  {n_v} voters × {n_c} candidates  |  seed={seed}")
    print(f"{'=' * 60}")

    # Header
    header = f"{'Voter':>6}  " + "  ".join(f"{c:>12}" for c in profile.labels_candidates)
    print(header)
    print("-" * len(header))

    for v in range(n_v):
        row = f"{v + 1:>6}  " + "  ".join(f"{ut[v, c]:>12.4f}" for c in range(n_c))
        print(row)

    print()
    print("  mean  :  " + "  ".join(f"{ut[:, c].mean():>12.4f}" for c in range(n_c)))
    print("  std   :  " + "  ".join(f"{ut[:, c].std():>12.4f}" for c in range(n_c)))
    print("  min   :  " + "  ".join(f"{ut[:, c].min():>12.4f}" for c in range(n_c)))
    print("  max   :  " + "  ".join(f"{ut[:, c].max():>12.4f}" for c in range(n_c)))


if __name__ == "__main__":
    show_profile("DDD_BETA", N_V, N_C, SEED)
    show_profile("DDD_BETA_POLAR", N_V, N_C, SEED)

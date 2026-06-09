"""Data generation module: profile generators and data instances."""

from vote_simulation.models.data_generation import (
    from_r_registry as _from_r_registry,  # noqa: F401 – registers DDD_BETA / DDD_BETA_POLAR
)
from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.data_generation.generator_registry import (
    GeneratorBuilder,
    get_generator_builder,
    list_generator_codes,
    make_generator_builder,
    register_generator,
)

__all__ = [
    "DataInstance",
    "GeneratorBuilder",
    "get_generator_builder",
    "list_generator_codes",
    "make_generator_builder",
    "register_generator",
]

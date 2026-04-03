"""Tests for the generator registry and DataInstance generation."""

import numpy as np
import pytest

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.data_generation.generator_registry import (
    _GENERATOR_BUILDERS,
    get_generator_builder,
    list_generator_codes,
    make_generator_builder,
    register_generator,
)

#  Registry


class TestGeneratorRegistry:
    """Tests for generator registration and lookup."""

    def test_list_codes_not_empty(self):
        codes = list_generator_codes()
        assert len(codes) >= 15  # all built-in models (incl. EUCLID_*D)
        assert "UNI" in codes
        assert "IC" in codes
        for dim_code in ("EUCLID_1D", "EUCLID_2D", "EUCLID_3D", "EUCLID_5D"):
            assert dim_code in codes

    def test_get_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown generator code"):
            get_generator_builder("NONEXISTENT_MODEL_XYZ")

    def test_register_custom_generator(self):
        from svvamp import GeneratorProfileCubicUniform, Profile

        def my_builder(n_v: int, n_c: int, **_kw: object) -> Profile:
            return GeneratorProfileCubicUniform(n_v=n_v, n_c=n_c)()

        register_generator("CUSTOM_TEST", my_builder)
        assert "CUSTOM_TEST" in list_generator_codes()
        builder = get_generator_builder("CUSTOM_TEST")
        profile = builder(n_v=5, n_c=3)
        assert profile.preferences_ut.shape == (5, 3)

        # Cleanup
        del _GENERATOR_BUILDERS["CUSTOM_TEST"]

    @pytest.mark.parametrize("code", ["uni", " Uni ", "UNI"])
    def test_case_insensitive_lookup(self, code: str):
        builder = get_generator_builder(code)
        assert callable(builder)


#  Generator builders


class TestBuiltinGenerators:
    """Each built-in generator produces a valid profile with correct shape."""

    N_V = 15
    N_C = 4

    @pytest.mark.parametrize(
        "code",
        [
            "UNI",
            "IC",
            "IANC",
            "EUCLID",
            "EUCLID_1D",
            "EUCLID_2D",
            "EUCLID_3D",
            "EUCLID_5D",
            "GAUSS",
            "LADDER",
            "SPHEROID",
            "PERTURB",
            "UNANIMOUS",
            "UFR",
            "VMF_HC",
            "VMF_HS",
        ],
    )
    def test_generator_produces_correct_shape(self, code: str):
        builder = get_generator_builder(code)
        profile = builder(self.N_V, self.N_C, seed=42, iteration=0)
        assert profile.preferences_ut.shape == (self.N_V, self.N_C)
        assert profile.n_v == self.N_V
        assert profile.n_c == self.N_C

    @pytest.mark.parametrize("code", ["UNI", "IC", "IANC"])
    def test_labels_are_human_readable(self, code: str):
        builder = get_generator_builder(code)
        profile = builder(10, 3, seed=0, iteration=0)
        assert profile.labels_candidates == ["Candidate 1", "Candidate 2", "Candidate 3"]


class TestMakeGeneratorBuilder:
    """Tests for the make_generator_builder helper."""

    def test_custom_builder_correct_shape(self):
        from svvamp import GeneratorProfileEuclideanBox

        builder = make_generator_builder(GeneratorProfileEuclideanBox, box_dimensions=[1.0, 1.0, 1.0, 1.0])
        profile = builder(n_v=20, n_c=5, seed=7)
        assert profile.preferences_ut.shape == (20, 5)
        assert profile.labels_candidates == [f"Candidate {i + 1}" for i in range(5)]

    def test_caller_can_override_defaults(self):
        from svvamp import GeneratorProfileEuclideanBox

        builder = make_generator_builder(GeneratorProfileEuclideanBox, box_dimensions=[1.0])
        # Override box_dimensions at call time
        profile = builder(n_v=10, n_c=3, seed=0, box_dimensions=[2.0, 3.0])
        assert profile.preferences_ut.shape == (10, 3)


#  Reproducibility


class TestReproducibility:
    """Same (seed, iteration) → same profile; different iteration → different profile."""

    def test_same_seed_same_result(self):
        builder = get_generator_builder("UNI")
        p1 = builder(20, 5, seed=123, iteration=0)
        p2 = builder(20, 5, seed=123, iteration=0)
        np.testing.assert_array_equal(p1.preferences_ut, p2.preferences_ut)

    def test_different_iteration_different_result(self):
        builder = get_generator_builder("UNI")
        p1 = builder(20, 5, seed=123, iteration=0)
        p2 = builder(20, 5, seed=123, iteration=1)
        assert not np.array_equal(p1.preferences_ut, p2.preferences_ut)


# DataInstance.from_generator


class TestDataInstanceFromGenerator:
    """Test the ``DataInstance.from_generator`` class method."""

    def test_creates_valid_instance(self):
        di = DataInstance.from_generator("UNI", n_v=10, n_c=3, seed=42)
        assert di.n_voters == 10
        assert di.n_candidates == 3
        assert di.profile is not None
        assert di.candidates.tolist() == ["Candidate 1", "Candidate 2", "Candidate 3"]

    def test_file_path_empty_when_generated(self):
        di = DataInstance.from_generator("IC", n_v=5, n_c=3, seed=0)
        assert di.file_path == ""


# Parquet round-trip


class TestParquetRoundTrip:
    """Generate → save parquet → reload → compare."""

    def test_save_and_load_parquet(self, tmp_path):
        di = DataInstance.from_generator("UNI", n_v=20, n_c=4, seed=99)
        path = str(tmp_path / "test_profile.parquet")
        di.save_parquet(path)

        di2 = DataInstance(path)

        np.testing.assert_array_almost_equal(di.data, di2.data)
        assert di.candidates.tolist() == di2.candidates.tolist()
        assert di2.n_voters == 20
        assert di2.n_candidates == 4

    def test_save_csv_round_trip(self, tmp_path):
        di = DataInstance.from_generator("IC", n_v=10, n_c=3, seed=7)
        path = str(tmp_path / "test_profile.csv")
        di.save_csv(path)

        # CSV round-trip is lossy (no row-index), but columns should match
        import pandas as pd

        df = pd.read_csv(path)
        assert list(df.columns) == di.candidates.tolist()
        assert len(df) == 10

"""Load or generate election profiles and persist them."""

from __future__ import annotations

import os
from csv import reader
from pathlib import Path

import numpy as np
import pandas as pd
from svvamp import Profile

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class DataInstance:
    """Encapsulates an election profile (utility matrix + candidate labels).

    `DataInstance`` can be created in three ways:

     **From an existing file** (CSV or Parquet)::

           di = DataInstance("path/to/data.csv")

     **From a generator** (wraps svvamp ``GeneratorProfile*``)::

           di = DataInstance.from_generator(
               model_code="UNI", n_v=101, n_c=5, seed=42, iteration=0
           )

     **From a raw Profile**::

           di = DataInstance.from_profile(profile)
    """

    def __init__(self, file_path: str):
        try:
            self.candidates, raw = self.get_data(file_path)
            self.data, self._orig_min, self._orig_max = self._normalize(raw)
            self.profile = self.build_profile(self.candidates, self.data)
            self.file_path = file_path
            self.model = None
        except Exception as e:
            raise ValueError(f"Error initializing DataInstance: {e}") from e

    # -------------------------------------------------- normalization helpers

    @staticmethod
    def _normalize(data: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Min-max normalize a utility matrix to [0, 1].

        The transformation is a global affine map that preserves every
        relative difference in the original data, making it fully
        reversible via :meth:`denormalize`.

        Args:
            data: 2-D array of shape ``(n_voters, n_candidates)``.

        Returns:
            A tuple ``(normalized, orig_min, orig_max)``.
        """
        dmin: float = float(data.min())
        dmax: float = float(data.max())
        spread = dmax - dmin
        if spread > 0.0:
            normalized = (data - dmin) * (1.0 / spread)  # one division
        else:
            # all utilities identical → perfect indifference
            normalized = np.full_like(data, 0.5)
        return normalized, dmin, dmax

    def denormalize(self) -> np.ndarray:
        """Restore the original (pre-normalization) utility values.

        Returns:
            2-D array with the same shape as ``self.data`` containing
            the utilities on their original scale.
        """
        spread = self._orig_max - self._orig_min
        if spread > 0.0:
            return self.data * spread + self._orig_min
        return np.full_like(self.data, self._orig_min)

    # --------------------------------------------------------- class methods

    @classmethod
    def from_generator(
        cls,
        model_code: str,
        n_v: int,
        n_c: int,
        *,
        seed: int = 0,
        iteration: int = 0,
        **extra_params: object,
    ) -> DataInstance:
        """Generate an election profile using a registered generator.

        Args:
            model_code: Registered generator short code (e.g. ``"UNI"``).
            n_v: Number of voters.
            n_c: Number of candidates.
            seed: Base random seed for reproducibility.
            iteration: Iteration index (added to *seed*).
            **extra_params: Model-specific keyword arguments forwarded to
                the generator builder.

        Returns:
            A new ``DataInstance`` whose profile was generated in-memory.
        """
        from vote_simulation.models.data_generation.generator_registry import (
            get_generator_builder,
        )

        builder = get_generator_builder(model_code)
        profile: Profile = builder(n_v, n_c, seed=seed, iteration=iteration, **extra_params)

        instance = object.__new__(cls)
        instance.candidates = np.asarray(profile.labels_candidates, dtype=str)
        raw = np.asarray(profile.preferences_ut, dtype=np.float64)
        instance.data, instance._orig_min, instance._orig_max = cls._normalize(raw)
        instance.profile = Profile(
            preferences_ut=instance.data,
            labels_candidates=profile.labels_candidates,
        )
        instance.file_path = ""  # not loaded from disk
        return instance

    @classmethod
    def from_profile(cls, profile: Profile, file_path: str = "") -> DataInstance:
        """Wrap an existing ``svvamp.Profile`` into a ``DataInstance``.

        Args:
            profile: An existing ``svvamp.Profile`` object.
            file_path: Optional file path associated with the profile.

        Returns:
            A new ``DataInstance`` wrapping the provided profile.
        """
        instance = object.__new__(cls)
        instance.candidates = np.asarray(profile.labels_candidates, dtype=str)
        raw = np.asarray(profile.preferences_ut, dtype=np.float64)
        instance.data, instance._orig_min, instance._orig_max = cls._normalize(raw)
        instance.profile = Profile(
            preferences_ut=instance.data,
            labels_candidates=profile.labels_candidates,
        )
        instance.file_path = file_path
        return instance

    # loaders

    def get_csv(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load candidate labels and utility matrix from a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            A tuple containing:
                - candidates: 1-D array of candidate names.
                - data: 2-D array of shape (n_voters, n_candidates) with utility values.
        """
        try:
            candidates_list: list[str] = []
            rows: list[list[float]] = []

            with open(file_path, encoding="utf-8", newline="") as fh:
                csv_reader = reader(fh)
                next(csv_reader, None)

                for row in csv_reader:
                    if len(row) < 2:
                        raise ValueError("CSV file must contain at least one data column.")
                    candidates_list.append(row[0].strip('"'))
                    rows.append([float(value) for value in row[1:]])

            if not rows:
                raise ValueError("CSV file must contain at least one row.")

            candidates = np.asarray(candidates_list, dtype=str)
            data = np.asarray(rows, dtype=np.float64).T  # rows = voters, columns = candidates

        except Exception as e:
            raise ValueError(f"Error reading the file : {e}") from e

        return candidates, data

    def get_parquet(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load candidate labels and utility matrix from a Parquet file.

        The Parquet file is expected to have one column per candidate
        (column name = candidate label) and one row per voter.

        Args:
            file_path: Path to the Parquet file.

        Returns:
            A tuple containing:
                - candidates: 1-D array of candidate names.
                - data: 2-D array of shape (n_voters, n_candidates) with utility values.
        """
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                raise ValueError("Parquet file is empty.")
            candidates = np.asarray(df.columns.tolist(), dtype=str)
            data = df.to_numpy(dtype=np.float64)  # (n_voters, n_candidates)
        except Exception as e:
            raise ValueError(f"Error reading parquet file: {e}") from e
        return candidates, data

    def get_data(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load data from a CSV or Parquet file.

        Args:
            file_path: Path to the data file.

        Returns:
            candidates: 1-D array of candidate names.
            data: 2-D array of shape ``(n_voters, n_candidates)``.
        """
        if not os.path.isfile(file_path):
            raise ValueError("Invalid file path. Please provide a valid file path.")

        if file_path.endswith(".csv"):
            return self.get_csv(file_path)

        if file_path.endswith(".parquet"):
            return self.get_parquet(file_path)

        raise ValueError("Unable to load data from provided file path.")

    # profile builder

    def build_profile(self, candidates: np.ndarray, data: np.ndarray) -> Profile:
        """Build a ``svvamp.Profile`` from candidate labels and utility matrix."""
        return Profile(preferences_ut=data, labels_candidates=candidates.tolist())

    def save_parquet(self, file_path: str) -> str:
        """Persist the utility matrix to a Parquet file.

        Creates parent directories if needed. The file contains one column
        per candidate and one row per voter.

        Args:
            file_path: Destination path (should end in ``.parquet``).

        Returns:
            The resolved absolute path of the written file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data, columns=self.candidates.tolist())
        df.to_parquet(str(path), index=False)
        return str(path.resolve())

    def save_csv(self, file_path: str) -> str:
        """Persist the utility matrix to a CSV file (same layout as input).

        Args:
            file_path: Destination path (should end in ``.csv``).

        Returns:
            The resolved absolute path of the written file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data, columns=self.candidates.tolist())
        df.to_csv(str(path), index=False)
        return str(path.resolve())

    @property
    def n_voters(self) -> int:
        """Number of voters in this instance."""
        return int(self.data.shape[0])

    @property
    def n_candidates(self) -> int:
        """Number of candidates in this instance."""
        return int(self.data.shape[1])
    
    @staticmethod
    def _cluster_order(matrix: np.ndarray, axis: int, method: str = "average", metric: str = "euclidean") -> np.ndarray:
        """Return the reordered indices of rows or columns via hierarchical clustering.

        Args:
            matrix: 2-D array ``(n_voters, n_candidates)``.
            axis: 0 = cluster rows (voters), 1 = cluster columns (candidates).
            method: Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
            metric: Distance metric passed to ``scipy.cluster.hierarchy.linkage``.

        Returns:
            1-D array of reordered indices.
        """
        n_items = matrix.shape[axis]
        if n_items <= 1 or not _HAS_SCIPY:
            return np.arange(n_items)

        # For column clustering, transpose so rows = candidates
        data = matrix.T if axis == 1 else matrix
        lnk = linkage(data, method=method, metric=metric)
        return leaves_list(lnk)

    

    def plot_heatmap(
        self,
        *,
        cluster_columns: bool = False,
        cluster_rows: bool = True,
        method: str = "average",
        metric: str = "euclidean",
        cmap: str = "viridis",
        title: str | None = None,
        save_path: str | None = None,
        show: bool = True,
    ) -> dict:
        """Visualize the utility matrix as a heatmap with optional hierarchical clustering.

        Values are already in [0, 1]. Columns (candidates) are reordered by
        hierarchical clustering by default so that similar preference profiles
        appear next to each other.

        Args:
            cluster_columns: Reorder candidates by hierarchical clustering (default True).
            cluster_rows: Reorder voters by hierarchical clustering (default False).
            method: Linkage method (``"average"``, ``"ward"``, ``"complete"``, …).
            metric: Distance metric (``"euclidean"``, ``"cosine"``, …).
            cmap: Matplotlib colormap name.
            title: Figure title. Defaults to model code if available.
            save_path: If provided, save the figure to this path.
            show: Whether to call ``plt.show()``.

        Returns:
            Dict with keys ``ordered_matrix``, ``row_order``, ``col_order``.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plot_heatmap(). Install it with: pip install matplotlib")

        if not _HAS_SCIPY and (cluster_columns or cluster_rows):
            print("[Warning] scipy not found. Clustering disabled. Install with: pip install scipy")

        matrix = self.data  # already in [0, 1]

        row_order: np.ndarray = (
            self._cluster_order(matrix, axis=0, method=method, metric=metric)
            if cluster_rows
            else np.arange(matrix.shape[0])
        )
        col_order: np.ndarray = (
            self._cluster_order(matrix, axis=1, method=method, metric=metric)
            if cluster_columns
            else np.arange(matrix.shape[1])
        )

        ordered = matrix[row_order][:, col_order]
        ordered_candidates = self.candidates[col_order]

        # --- figure sizing
        fig_w = max(8, ordered.shape[1] * 0.5)
        fig_h = max(5, ordered.shape[0] * 0.08)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(
            ordered,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )

        # --- axes labels
        if title is None:
            title = f"Profiles heatmap — {self.model_code}" if self.model_code else "Profiles heatmap"

        xlabel = "Candidates"
        if cluster_columns and _HAS_SCIPY:
            xlabel += " (clustered)"
        ylabel = "Voters"
        if cluster_rows and _HAS_SCIPY:
            ylabel += " (clustered)"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # candidate labels on x-axis (readable even with many candidates)
        ax.set_xticks(np.arange(len(ordered_candidates)))
        ax.set_xticklabels(ordered_candidates, rotation=45, ha="right", fontsize=8)

        # hide individual voter ticks when there are many
        if ordered.shape[0] <= 30:
            ax.set_yticks(np.arange(ordered.shape[0]))
            ax.set_yticklabels(row_order, fontsize=7)
        else:
            ax.set_yticks([])

        plt.colorbar(im, ax=ax, label="Normalized utility [0, 1]")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()

        return {
            "ordered_matrix": ordered,
            "row_order": row_order,
            "col_order": col_order,
        }

if __name__ == "__main__":
    # Example usage:
    di = DataInstance.from_generator(model_code="UNI", n_v=100, n_c=5, seed=42)
    di = DataInstance.from_generator(model_code="EUCLID_1D", n_v=100, n_c=5, seed=42)
    di = DataInstance.from_generator(model_code="EUCLID_2D", n_v=100, n_c=15, seed=42)
    di = DataInstance.from_generator(model_code="EUCLID_5D", n_v=100, n_c=15, seed=42)
    print("Candidates:", di.candidates)
    print("Utility matrix shape:", di.data.shape)
    di.plot_heatmap(cluster_columns=True, cluster_rows=False, title="Uniform Model Heatmap")
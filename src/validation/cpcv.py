"""Combinatorial Purged Cross-Validation (CPCV).

López de Prado, *Advances in Financial Machine Learning*, ch. 7.

Standard k-fold CV leaks information when training samples are
correlated with test samples (which is the rule, not the exception, in
finance: overlapping return windows, autocorrelation, label-horizon
spillover). CPCV fixes this in two ways:

1. **Combinatorial groups.** Instead of one test group per fold, pick
   *n_test* groups out of *n_splits*. The number of "paths" through the
   data grows binomially, giving you many more OOS estimates from the
   same dataset.
2. **Purge + embargo.** Drop training samples whose label horizon
   overlaps the test set (purge), plus an additional fraction either
   side as a buffer (embargo).

Returns ``(train_indices, test_indices)`` tuples. Use them to slice
features / labels / returns.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np


class CombinatorialPurgedKFold:
    """CPCV split generator.

    Args:
        n_splits: number of contiguous groups the dataset is divided into.
        n_test_splits: number of groups withheld as test in each combination.
        embargo_pct: fraction of total samples to drop as embargo on each
            side of every test group. Typical 0.01-0.02.

    Yields ``(train_idx, test_idx)`` numpy arrays. ``test_idx`` is the
    union of the chosen test groups; ``train_idx`` is everything else
    minus the embargo zone.
    """

    def __init__(
        self,
        n_splits: int = 8,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("0 < n_test_splits < n_splits")
        if not 0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def get_n_splits(self) -> int:
        """Total number of split combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield combinatorial purged train/test index tuples."""
        if n_samples < self.n_splits:
            raise ValueError(
                f"n_samples={n_samples} must be >= n_splits={self.n_splits}"
            )

        # Group boundaries — split contiguously into n_splits chunks.
        bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)
        groups = [
            np.arange(bounds[i], bounds[i + 1]) for i in range(self.n_splits)
        ]
        embargo_size = int(n_samples * self.embargo_pct)

        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([groups[i] for i in sorted(test_combo)])

            # Embargo zone: a window around each test group.
            embargo_mask = np.zeros(n_samples, dtype=bool)
            for i in test_combo:
                start = max(0, bounds[i] - embargo_size)
                end = min(n_samples, bounds[i + 1] + embargo_size)
                embargo_mask[start:end] = True

            # Train = everything not in embargo zone.
            train_mask = ~embargo_mask
            train_idx = np.where(train_mask)[0]

            yield train_idx, test_idx


def generate_splits(
    n_samples: int,
    *,
    n_splits: int = 8,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convenience wrapper — return all combinations as a materialised list."""
    cpcv = CombinatorialPurgedKFold(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        embargo_pct=embargo_pct,
    )
    return list(cpcv.split(n_samples))

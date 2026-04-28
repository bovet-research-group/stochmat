"""
Reusable test helpers for stochmat and downstream packages.

This module exposes lightweight factories for generating test inputs
(clusterings, etc.) so that downstream packages can exercise the same
patterns without duplicating boilerplate.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def make_clusterings(
    N: int,
    sizes1: Sequence[int],
    sizes2: Sequence[int],
    seed: int | None = 42,
) -> tuple[list[set[int]], list[set[int]]]:
    """Build two random partitions of ``{0, ..., N-1}``.

    Each partition is created by shuffling the node indices and slicing them
    into clusters with the requested sizes.

    Parameters
    ----------
    N : int
        Total number of nodes.
    sizes1, sizes2 : sequence of int
        Cluster sizes for the first and second partition. The sum of each
        sequence must not exceed ``N``; remaining nodes are dropped.
    seed : int or None, optional
        Seed for the random number generator. Use ``None`` for non-deterministic
        behaviour.

    Returns
    -------
    tuple of (list[set[int]], list[set[int]])
        Two clusterings, each a list of disjoint sets of node indices.
    """
    rng = np.random.default_rng(seed)

    def _partition(sizes: Iterable[int]) -> list[set[int]]:
        nodes = rng.permutation(N)
        out: list[set[int]] = []
        start = 0
        for size in sizes:
            out.append(set(nodes[start:start + size].tolist()))
            start += size
        return out

    return _partition(sizes1), _partition(sizes2)


def make_equal_clusterings(
    N: int,
    k1: int,
    k2: int,
    seed: int | None = 42,
) -> tuple[list[set[int]], list[set[int]]]:
    """Build two random partitions with (approximately) equal cluster sizes.

    Convenience wrapper around :func:`make_clusterings` that splits ``N`` nodes
    into ``k1`` and ``k2`` clusters of size ``N // k1`` and ``N // k2``
    respectively. Trailing nodes are dropped if ``N`` is not divisible.

    Parameters
    ----------
    N : int
        Total number of nodes.
    k1, k2 : int
        Number of clusters in the two partitions.
    seed : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    tuple of (list[set[int]], list[set[int]])
        Two clusterings, each a list of disjoint sets of node indices.
    """
    return make_clusterings(N, [N // k1] * k1, [N // k2] * k2, seed=seed)

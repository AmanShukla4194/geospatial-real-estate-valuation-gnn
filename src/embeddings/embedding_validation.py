"""
Spatial Embedding Validation.

Validates the generated spatial feature embeddings.

Author:
    Aman Shukla
"""

from __future__ import annotations

import numpy as np

from src.embeddings.spatial_embeddings import (
    build_spatial_embeddings,
)


def validate_embeddings():

    print("=" * 70)
    print("Validating Spatial Embeddings")
    print("=" * 70)

    embeddings = build_spatial_embeddings()

    print()

    print("Running validation checks...\n")

    if embeddings is None:
        raise ValueError(
            "Embedding matrix was not created."
        )

    print("✓ Embedding matrix exists")

    if embeddings.ndim != 2:

        raise ValueError(
            "Embedding matrix should be two-dimensional."
        )

    print("✓ Matrix dimension validation passed")

    if np.isnan(
        embeddings
    ).any():

        raise ValueError(
            "NaN values detected."
        )

    print("✓ NaN validation passed")

    if np.isinf(
        embeddings
    ).any():

        raise ValueError(
            "Infinite values detected."
        )

    print("✓ Infinite value validation passed")

    print()

    print("Embedding Statistics")
    print("-" * 70)

    print(
        f"Nodes               : {embeddings.shape[0]:,}"
    )

    print(
        f"Embedding Dimension : {embeddings.shape[1]}"
    )

    print(
        f"Mean                : {embeddings.mean():.6f}"
    )

    print(
        f"Standard Deviation  : {embeddings.std():.6f}"
    )

    print("-" * 70)

    print()

    print(
        "Spatial embedding validation completed successfully."
    )

    print("=" * 70)

    return embeddings


if __name__ == "__main__":

    validate_embeddings()
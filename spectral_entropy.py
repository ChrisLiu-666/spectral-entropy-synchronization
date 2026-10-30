"""Scale-dependent spectral-entropy descriptors for undirected networks.

The definitions implemented here correspond to Eqs. (2)--(5) and the
definitions of S_e^* and S_A in the accompanying manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.integrate import trapezoid


@dataclass(frozen=True)
class SpectralEntropyResult:
    """Spectral-entropy curves and the two summary descriptors."""

    beta: np.ndarray
    entropy: np.ndarray
    specific_heat: np.ndarray
    beta_star: float
    entropy_star: float
    entropy_area: float


def laplacian_eigenvalues(graph: nx.Graph, weight: str | None = "weight") -> np.ndarray:
    """Return the sorted eigenvalues of the combinatorial graph Laplacian."""
    if graph.is_directed():
        raise nx.NetworkXNotImplemented("Only undirected graphs are supported.")
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph must contain at least one node.")

    laplacian = nx.laplacian_matrix(graph, weight=weight).astype(float).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    # Remove harmless negative roundoff near the zero eigenvalue.
    eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
    return eigenvalues


def laplacian_eigendecomposition(
    graph: nx.Graph, weight: str | None = "weight"
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted Laplacian eigenvalues and orthonormal eigenvectors.

    Eigenvectors are stored column-wise. The row order follows
    ``list(graph.nodes())``.
    """
    if graph.is_directed():
        raise nx.NetworkXNotImplemented("Only undirected graphs are supported.")
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph must contain at least one node.")

    laplacian = nx.laplacian_matrix(graph, weight=weight).astype(float).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
    return eigenvalues, eigenvectors


def spectral_entropy_curve(
    eigenvalues: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gibbs spectral entropy S_e(beta) and specific heat C(beta).

    Parameters
    ----------
    eigenvalues:
        One-dimensional array containing the full Laplacian spectrum,
        including every zero eigenvalue.
    beta:
        Positive inverse-temperature values.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array.")
    if beta.ndim != 1 or beta.size == 0 or np.any(beta <= 0):
        raise ValueError("beta must be a non-empty array of positive values.")
    if np.any(eigenvalues < -1e-10):
        raise ValueError("Laplacian eigenvalues must be non-negative.")

    eigenvalues = np.maximum(eigenvalues, 0.0)
    shifted = eigenvalues - eigenvalues.min()
    weights = np.exp(-beta[:, None] * shifted[None, :])
    probabilities = weights / weights.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_probabilities = np.where(probabilities > 0, np.log2(probabilities), 0.0)
    entropy = -np.sum(probabilities * log_probabilities, axis=1)

    mean = probabilities @ eigenvalues
    mean_square = probabilities @ np.square(eigenvalues)
    variance = np.maximum(mean_square - np.square(mean), 0.0)
    specific_heat = np.square(beta) * variance / np.log(2.0)
    return entropy, specific_heat


def _integrate_effective_range(
    beta: np.ndarray,
    normalized_entropy: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Integrate over every contiguous component of the effective range."""
    selected = np.flatnonzero(mask)
    if selected.size < 2:
        return 0.0

    total = 0.0
    for component in np.split(selected, np.where(np.diff(selected) > 1)[0] + 1):
        if component.size >= 2:
            total += float(
                trapezoid(normalized_entropy[component], x=np.log(beta[component]))
            )
    return total


def spectral_entropy_descriptors(
    graph: nx.Graph,
    beta: np.ndarray | None = None,
    mu: float = 0.05,
    weight: str | None = "weight",
) -> SpectralEntropyResult:
    """Compute S_e(beta), C(beta), beta*, S_e*, and S_A for a graph.

    ``beta_star`` is the location of the dominant specific-heat peak.
    ``entropy_area`` integrates S_e/log2(N) over the grid points satisfying
    C(beta) >= mu * max(C), using d(log beta).
    """
    if not 0 < mu < 1:
        raise ValueError("mu must lie strictly between 0 and 1.")
    if beta is None:
        beta = np.logspace(-3, 3, 301)
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1 or np.any(np.diff(beta) <= 0):
        raise ValueError("beta must be a strictly increasing one-dimensional array.")

    eigenvalues = laplacian_eigenvalues(graph, weight=weight)
    entropy, specific_heat = spectral_entropy_curve(eigenvalues, beta)
    peak_index = int(np.argmax(specific_heat))

    if graph.number_of_nodes() == 1:
        normalized_entropy = np.zeros_like(entropy)
    else:
        normalized_entropy = entropy / np.log2(graph.number_of_nodes())

    heat_max = float(specific_heat[peak_index])
    effective_mask = specific_heat >= mu * heat_max if heat_max > 0 else np.zeros_like(
        specific_heat, dtype=bool
    )
    entropy_area = _integrate_effective_range(beta, normalized_entropy, effective_mask)

    return SpectralEntropyResult(
        beta=beta,
        entropy=entropy,
        specific_heat=specific_heat,
        beta_star=float(beta[peak_index]),
        entropy_star=float(entropy[peak_index]),
        entropy_area=entropy_area,
    )

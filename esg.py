"""Dynamic Entropy-Sensitivity Greedy (ESG) node selection.

At each round, ESG computes entropy-sensitivity scores from the current
principal submatrix, selects the node with the largest score, and deletes the
selected node's row and column before the next round.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from spectral_entropy import spectral_entropy_curve


@dataclass(frozen=True)
class ESGRound:
    """Information recorded for one ESG selection round."""

    beta_star: float
    selected_node: Hashable
    selected_score: float
    scores: dict[Hashable, float]


@dataclass(frozen=True)
class ESGResult:
    """Ordered selected nodes and complete round-by-round diagnostics."""

    ranking: list[Hashable]
    rounds: list[ESGRound]


def _validate_graph(graph: nx.Graph) -> None:
    if graph.is_directed() or graph.is_multigraph():
        raise nx.NetworkXNotImplemented(
            "ESG requires a simple, connected, undirected graph."
        )
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph must contain at least one node.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Self-loops are not supported.")
    if not nx.is_connected(graph):
        raise ValueError("ESG is defined here for connected graphs.")


def entropy_sensitivity_from_matrix(
    matrix: np.ndarray,
    beta: float,
) -> np.ndarray:
    r"""Return diagonal-perturbation entropy sensitivities for a matrix.

    For eigenpairs ``(lambda_i, x_i)`` of the current symmetric principal
    submatrix, this computes

    ``c_r = beta^2/ln(2) * sum_i p_i * (<lambda>_beta-lambda_i) * x_i(r)^2``.

    The input may be the original graph Laplacian or a grounded principal
    submatrix obtained after deleting previously selected rows and columns.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("matrix must be a non-empty square array.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("matrix must be symmetric.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number.")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues[0] < -1e-9:
        raise ValueError("matrix must be positive semidefinite.")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    shifted = eigenvalues - eigenvalues.min()
    gibbs_weights = np.exp(-beta * shifted)
    probabilities = gibbs_weights / gibbs_weights.sum()
    mean_eigenvalue = float(probabilities @ eigenvalues)
    modal_coefficients = probabilities * (mean_eigenvalue - eigenvalues)
    return (
        beta**2
        / np.log(2.0)
        * (np.square(eigenvectors) @ modal_coefficients)
    )


def entropy_sensitivity_scores(
    graph: nx.Graph,
    beta: float,
    weight: str | None = "weight",
) -> dict[Hashable, float]:
    """Return ESG scores for all nodes of the original graph Laplacian."""
    _validate_graph(graph)
    nodes = list(graph.nodes())
    laplacian = nx.laplacian_matrix(graph, nodelist=nodes, weight=weight).toarray()
    values = entropy_sensitivity_from_matrix(laplacian, beta)
    return {node: float(values[i]) for i, node in enumerate(nodes)}


def _validate_beta_grid(beta: np.ndarray | None) -> np.ndarray:
    if beta is None:
        beta = np.logspace(-3, 3, 301)
    beta = np.asarray(beta, dtype=float)
    if (
        beta.ndim != 1
        or beta.size == 0
        or np.any(~np.isfinite(beta))
        or np.any(beta <= 0)
        or np.any(np.diff(beta) <= 0)
    ):
        raise ValueError("beta must be a finite, positive, strictly increasing array.")
    return beta


def _characteristic_beta(matrix: np.ndarray, beta: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    _, specific_heat = spectral_entropy_curve(eigenvalues, beta)
    return float(beta[int(np.argmax(specific_heat))])


def esg_ranking(
    graph: nx.Graph,
    beta: np.ndarray | None = None,
    n_select: int | None = None,
    weight: str | None = "weight",
) -> ESGResult:
    """Run dynamic ESG by repeatedly deleting the selected row and column.

    At each round:

    1. compute ``beta_star`` from the dominant specific-heat peak of the
       current principal submatrix;
    2. compute the entropy-sensitivity score of every remaining node;
    3. select the largest score, retaining current node order for exact ties;
    4. delete the selected row and column.

    Original graph labels are retained in the output.
    """
    _validate_graph(graph)
    beta = _validate_beta_grid(beta)
    nodes: list[Hashable] = list(graph.nodes())

    if n_select is None:
        n_select = len(nodes)
    if not isinstance(n_select, (int, np.integer)) or not 0 <= n_select <= len(nodes):
        raise ValueError("n_select must be between zero and the number of nodes.")

    current_matrix = nx.laplacian_matrix(
        graph, nodelist=nodes, weight=weight
    ).astype(float).toarray()
    remaining_nodes = nodes.copy()
    rounds: list[ESGRound] = []

    for _ in range(int(n_select)):
        beta_star = _characteristic_beta(current_matrix, beta)
        score_values = entropy_sensitivity_from_matrix(current_matrix, beta_star)
        selected_index = int(np.argmax(score_values))
        selected_node = remaining_nodes[selected_index]
        scores = {
            node: float(score_values[i]) for i, node in enumerate(remaining_nodes)
        }
        rounds.append(
            ESGRound(
                beta_star=beta_star,
                selected_node=selected_node,
                selected_score=float(score_values[selected_index]),
                scores=scores,
            )
        )

        current_matrix = np.delete(
            np.delete(current_matrix, selected_index, axis=0),
            selected_index,
            axis=1,
        )
        remaining_nodes.pop(selected_index)

    return ESGResult(
        ranking=[round_result.selected_node for round_result in rounds],
        rounds=rounds,
    )

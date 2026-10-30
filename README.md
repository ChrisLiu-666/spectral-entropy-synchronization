# Spectral entropy and ESG

This repository contains the two algorithmic components and six network
datasets released with the
manuscript *Full-Spectrum Organization of Network Synchronization:
Scale-Dependent Entropy and Control*:

1. scale-dependent Gibbs spectral entropy and the descriptors
   \(S_e(\beta)\), \(\mathcal C(\beta)\), \(\beta^\ast\), \(S_e^\ast\), and
   \(S_A\);
2. the Entropy-Sensitivity Greedy (ESG) node score and ranking.

The repository intentionally provides these reusable algorithmic functions,
not the complete simulation, data-processing, or figure-generation pipeline.

## Files

- `spectral_entropy.py` - Laplacian spectrum, entropy curve, specific heat,
  characteristic scale, and integrated entropy descriptor.
- `esg.py` - first-order node entropy-sensitivity scores and ESG ordering.
- `example.py` - a minimal executable example.
- `tests/test_functions.py` - numerical and consistency tests.
- `requirements.txt` - Python dependencies.
- `data/` - edge lists for the six networks used in the pinning experiments.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Quick start

```bash
python example.py
```

Both modules accept a connected, simple, undirected `networkx.Graph`.

```python
import networkx as nx

from esg import esg_ranking
from spectral_entropy import spectral_entropy_descriptors

G = nx.read_edgelist("network.txt", nodetype=int)

entropy = spectral_entropy_descriptors(G, mu=0.05)
print(entropy.beta_star, entropy.entropy_star, entropy.entropy_area)

esg = esg_ranking(G, n_select=G.number_of_nodes() // 2)
print(esg.ranking)
print([round_.beta_star for round_ in esg.rounds])
```

## Network data

The `data` directory contains whitespace-separated edge lists. Load them as
simple, undirected, unweighted graphs:

```python
G = nx.read_edgelist(
    "data/Celegans.txt", nodetype=int, create_using=nx.Graph
)
```

| File | Nodes | Edges | Description |
| --- | ---: | ---: | --- |
| `Celegans.txt` | 297 | 2,148 | C. elegans neural network |
| `Jazz.txt` | 198 | 2,742 | Jazz musicians network |
| `Email.txt` | 1,133 | 5,451 | Email communication network |
| `NS_GC.txt` | 379 | 914 | Network-science coauthorship component |
| `WS_500_0.3.txt` | 500 | 1,000 | Watts-Strogatz instance |
| `ER_500_0.01.txt` | 496 | 1,202 | Largest connected component of an ER instance |

All six supplied files are connected and contain no self-loops. See
`data/README.md` for provenance and preprocessing notes.

By default, both calculations use 301 logarithmically spaced values of
`beta` in `[1e-3, 1e3]`, matching the grid described in the Supplementary
Data. A custom positive, strictly increasing grid can be supplied.

## Spectral-entropy definitions

For Laplacian eigenvalues \(\lambda_i\),

```text
p_i(beta) = exp(-beta lambda_i) / sum_j exp(-beta lambda_j)
S_e(beta) = -sum_i p_i(beta) log2 p_i(beta)
C(beta)   = beta^2 Var_beta(lambda) / ln(2).
```

`beta_star` is the dominant peak of `C(beta)`, `entropy_star` is
`S_e(beta_star)`, and `entropy_area` is the trapezoidal approximation of
`S_e/log2(N)` over `C(beta) >= mu * max(C)`, integrated with respect to
`log(beta)`.

## ESG definition

For the current principal submatrix, at its characteristic inverse
temperature \(\beta^\ast\), ESG assigns remaining node \(r\) the first-order
entropy sensitivity

```text
c_r(beta) = beta^2/ln(2) * sum_i p_i(beta)
            * (<lambda>_beta - lambda_i) * x_i(r)^2.
```

Here `x_i(r)` is component `r` of a normalized eigenvector of the current
matrix. The implementation remains valid for repeated eigenvalues: summing
an orthonormal basis within each repeated eigenspace is equivalent to the
spectral-projector expression in the manuscript.

ESG is implemented as a dynamic greedy procedure. At every round it
recalculates `beta_star` and all remaining-node scores, selects the largest
score, and deletes the selected node's row and column. The resulting
principal submatrix is used in the next round. Original graph node labels are
preserved. Exact ties retain the current node order. The returned `rounds`
records each round's `beta_star`, selected score, and all candidate scores.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests check entropy limits, finiteness, the initial zero-sum sensitivity
identity, agreement with a finite-difference entropy derivative, and the
round-by-round principal-submatrix update.

## Scope of the availability statement

This release supports calculation of the manuscript's spectral-entropy
descriptors and ESG node rankings and provides the six networks used in the
pinning comparison. It does not include the complete dynamical-simulation or
figure-generation pipeline.

## Citation

Please cite the accompanying manuscript when using these functions.

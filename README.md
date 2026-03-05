# Polyakov Gram-to-Graph Test Cascade

This repository contains the code, run configurations, and reproducibility artifacts for the manuscript:

**From Polyakov Worldsheet Ensembles to Effective Geometry: A Reproducible Gram→Graph Test Cascade**

The project implements a conservative numerical pipeline from Euclidean Polyakov-worldsheet samples to fair normalized Gram matrices, graph-derived distances, and DPH diagnostics. It includes robustness sweeps, threshold sensitivity analysis, and negative controls.

---

## Scope

This repository is intended as a reproducible **diagnostic engine**, not as a claim of fundamental spacetime reconstruction.

The goal is to test when a Gram structure, derived from localized worldsheet states, supports a stable, metric-like coarse geometry under explicit and controlled constructions.

The present baseline is deliberately minimal:
- Euclidean lattice worldsheet
- target dimension \(D=1\)
- moderate lattice size
- limited but reproducible sampling
- explicit robustness and control experiments

This is a proof-of-concept methodology paper with an emphasis on:
- operational definitions
- failure modes
- reproducibility
- conservative interpretation

---

## Main pipeline

The implemented pipeline is:

1. **Euclidean worldsheet sampling** on a lattice
2. **Window-state construction** along the \(\tau\)-direction
3. **Fair normalized Gram matrix** computation
4. **Distance mapping** via  
   \[
   y_{ij} = -\log(\mathrm{clip}(|K_{ij}|,\varepsilon,1))
   \]
5. **Graph construction** using k-nearest neighbors
6. **All-pairs shortest paths**
7. **DPH diagnostics**
8. **Robustness sweeps** over \((W_\tau,\mathrm{stride})\)
9. **\(K_{\min}\)-sensitivity analysis**
10. **Negative controls**

---

## Repository structure

```text
code/worldsheet_polyakov/      Main analysis code
outputs/worldsheet_polyakov/   Run outputs, K files, summaries, plots
docs/                          Notes and manuscript-related material
tools/                         Helper scripts for sweeps and controls


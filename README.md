# CONFHIT: Conformal Generative Design with Oracle-Free Guarantees

[ConfHit](https://arxiv.org/abs/2603.07371) is a model-agnostic framework that provides **validity guarantees** for conditional generative models in resource-constrained settings (e.g. drug discovery) **without oracle access**. It addresses:

1. **Certification** — Given a batch of generated candidates, guarantee at confidence level \(1-\alpha\) that it contains at least one *hit* (e.g. a molecule satisfying a desired property).
2. **Design** — Refine the batch to a **compact certified set** while preserving that guarantee

The method uses **density-ratio weighted conformal p-values** to correct for distribution shift between calibration and generated data, and a **conformal nested testing** procedure for the design problem (Algorithm 1 in the paper). No experimental oracle is required; calibration uses historical labeled data only.

---

## Setup

1. **Conda env** (e.g. `chemprop` with chemprop, torch, pandas, sklearn, tqdm).

Run all commands from the repo root.

---

## Commands


### Design (conformal nested testing)

Refines each input’s generated batch to a compact certified set (Algorithm 1).

```bash

# Molcraft, N=10 (requires precomputed features; see Setup)
conda run -n chemprop python src/design_main.py --config config/molcraft_N10.json
```

### Certification

Certifies whether a given batch contains at least one hit at level \(\alpha\) (Section 3.1 in the paper).

```bash
conda run -n chemprop python src/certification_main.py --config config/molcraft_N10.json
```

### Budget analysis

Budget-constrained allocation across multiple inputs (Section 4.6 / Appendix D in the paper).

```bash
conda run -n chemprop python src/budget_analysis.py --config config/molcraft_N10_budget.json
```

---

## Layout

| Path | Description |
|------|-------------|
| `src/design_main.py` | Design pipeline (nested testing). |
| `src/certification_main.py` | Certification-only pipeline. |
| `src/budget_analysis.py` | Budget analysis across inputs. |
| `src/design.py` | Core: conformal p-values, nested testing (`design_test`). |
| `src/certification.py` | Certification analysis. |
| `src/models.py` | Property predictor (Chemprop binary MPNN). |
| `src/kde_utils.py` | KDE for density-ratio estimation. |
| `config/` | JSON configs (qed_hgraph N=7, molcraft N=10). |
| `data/calib/` | Calibration data (`qed_calib.csv`, `dock_calib.csv`). |
| `data/generated_samples/` | Generated candidate batches. |
| `trained_models/` | Property model checkpoint (e.g. `binary_qed.ckpt`). |

---

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{laghuvarapu2026confhitconformalgenerativedesign,
      title={ConfHit: Conformal Generative Design with Oracle Free Guarantees}, 
      author={Siddhartha Laghuvarapu and Ying Jin and Jimeng Sun},
      year={2026},
      eprint={2603.07371},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.07371}, 
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

# Explainable DDoS Detection with Multi-Expert Transformer Ensembles

> **Transformer-based DDoS detection prioritizing interpretability and uncertainty estimation on the CIC-DDoS2019 dataset.**

## Motivation

Existing DDoS detection models on CIC-DDoS2019 achieve 99.99% accuracy using Random Forest or XGBoost — but they are black boxes. A SOC analyst receiving thousands of alerts per day needs to know:

1. **Why** was this traffic flagged?
2. **How confident** is the model in this prediction?
3. **What type** of attack is this?

This project explores whether transformer-based architectures with specialized attention mechanisms can answer these questions — trading raw accuracy for explainability and deployment readiness.

## Architecture

```
Input: Network flow sequences (batch, 64 flows, 43 features)
  │
  ├── Feature Embedding (43 → 128)
  ├── Protocol Embedding (1 → 64)
  └── Temporal Embedding (1 → 64)
  │
  ▼ Concatenate + Project → (batch, 64, 128)
  │
  ├── Expert 1: Temporal Attention  (8 heads, 7 layers)  — timing patterns
  ├── Expert 2: Protocol Attention  (4 heads, 6 layers)  — protocol-specific behavior
  └── Expert 3: Feature Attention   (16 heads, 8 layers) — feature importance
  │
  ▼ Dynamic Meta-Learning Gating (learned expert weighting per sample)
  │
  ▼ Uncertainty Estimation (trained on prediction correctness)
  │
  ├── Binary Head:     attack vs normal
  ├── Multiclass Head: 13 attack types
  └── Anomaly Head:    continuous anomaly score
```

## Results

Results below are from training on CIC-DDoS2019 with 3,052 training sequences and 760 test sequences. All numbers are from actual notebook runs, see notebooks 03–05 for full training logs.

### Attention Mechanism Comparison (NB03, 10 epochs, depth=4)

| Expert Type | Binary F1 | MC F1 | Anomaly Corr | Combined |
|---|---|---|---|---|
| Cross-Modal | 0.9816 | 0.9321 | 0.9666 | **0.9538** |
| Feature | 0.9868 | 0.9250 | 0.9711 | 0.9528 |
| Temporal | 0.9804 | 0.9269 | 0.9574 | 0.9490 |
| Protocol | 0.9842 | 0.9225 | 0.9617 | 0.9488 |

### Multi-Task Extension (NB03b, 30 epochs, depth=6)

| Expert Type | Binary F1 | MC F1 | Anomaly Corr | Combined |
|---|---|---|---|---|
| Cross-Modal | 0.9882 | 0.9455 | 0.9750 | **0.9642** |
| Temporal | 0.9856 | 0.9483 | 0.9691 | 0.9637 |
| Feature | 0.9895 | 0.9388 | 0.9672 | 0.9597 |
| Protocol | 0.9855 | 0.9364 | 0.9672 | 0.9573 |

### Comparison with Existing Work

| Method | Binary F1 | MC F1 | Explainable? | Uncertainty? | Multi-task? |
|---|---|---|---|---|---|
| Random Forest / XGBoost | ~0.97 | ~0.88 | No | No | No |
| CNN-based / ViT hybrids | ~0.99 | — | No | No | No |
| Time Series Transformer (TST) | ~0.96 | ~0.90 | Limited | No | No |
| **Ours (Single Expert, NB03b)** | **~0.99** | **~0.95** | **Yes** | **No** | **Yes** |
| **Ours (Ensemble, NB04)** | **~0.98** | **~0.93** | **Yes** | **Yes** | **Yes** |

Individual attention experts and the full ensemble are competitive with or exceed tabular and CNN-based baselines while providing attention-based explainability, calibrated uncertainty estimation, and simultaneous multi-task outputs.

## Key Contributions

1. **Explainability-first multi-expert ensemble** — An ensemble of three transformer experts (temporal, protocol, feature) combined through a learned gating mechanism that provides concept-level attribution: analysts can observe which expert (and thus which behavioural category) drives each detection decision.

2. **Comparative study of attention mechanisms** — A systematic evaluation of four attention variants (temporal, protocol, feature, cross-modal) for DDoS detection, showing that cross-modal attention achieves the highest combined score while all variants remain competitive with tabular baselines.

3. **Integrated uncertainty estimation** — A dedicated uncertainty head trained on prediction correctness, providing a calibrated confidence signal for escalating low-confidence predictions without resorting to Bayesian inference at test time.

4. **Multi-task learning framework** — A joint multi-task setup with learned task weights that simultaneously optimises binary detection, 13-class attack classification, and anomaly severity scoring from a shared representation.

## Project Structure

```
DDoS-Transformer-Detection/
├── README.md
├── data/
│   └── README.md                      # Dataset description and setup
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA on preprocessed sequences
│   ├── 02_sequence_creation.ipynb     # Tabular → sequence transformation
│   ├── 03_attention_exploration.ipynb  # Attention type comparison (multi-task)
│   ├── 03b_multitask_extension.ipynb  # Deeper multi-task training
│   ├── 04_ensemble_model.ipynb        # Final ensemble architecture + training
│   └── 05_evaluation.ipynb            # Results, ablation, visualizations
├── src/
│   ├── __init__.py
│   ├── dataset.py                     # Sequence creation + Dataset classes
│   ├── models.py                      # All model architectures
│   ├── training.py                    # Training loops + loss functions
│   └── evaluation.py                  # Metrics + attention visualization
└── research_draft.pdf                 # Research report (IEEE format)             
```

## Setup & Reproduction

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset

This project uses the [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) dataset. See [data/README.md](data/README.md) for download and preprocessing instructions.

### Running

1. Place preprocessed CSV files in `data/` (see data README)
2. Run notebooks in order: `01_data_exploration.ipynb` → `05_evaluation.ipynb`
3. Or use `src/` modules directly:

```python
from src.dataset import create_flow_based_sequences, MultiTaskDataset
from src.models import AdvancedEnsembleTransformer
from src.training import train_advanced_ensemble
from src.evaluation import evaluate_model
```

## Limitations & Future Work

### Limitations

- **Simulated data:** CIC-DDoS2019 is generated in a controlled environment; real-world traffic contains more noise and concept drift.
- **Limited cross-dataset evaluation:** We have not validated the approach on other datasets (e.g., UNSW-NB15, CSE-CIC-IDS2018).
- **No attribution explainability:** The embedding layer mixes features before attention, preventing direct feature-level attribution without post-hoc methods.
- **Sequence creation sensitivity:** Choices such as sequence length (64) and stride (32) were not exhaustively optimised and may affect performance.

### Future Work

- **Cross-dataset evaluation** on UNSW-NB15, CSE-CIC-IDS2018, and real traffic captures.
- **Post-hoc attribution** using integrated gradients or SHAP to complement expert-level explanations.
- **Online learning** for adaptation to concept drift in live networks.
- **Sequence length optimisation** via hyperparameter search.
- **Lightweight deployment:** knowledge distillation to a smaller model for edge use.
## License

This project is for academic and research purposes.

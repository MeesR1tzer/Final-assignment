# XAI for Fraud Detection: SHAP vs CEM

A code demonstration for the Explainable AI course (2025/26), comparing two explanation methods — SHAP and CEM — applied to a Random Forest classifier trained on the ULB Credit Card Fraud Detection dataset.

## Research Question

*To what extent does CEM provide more actionable insights for fraud investigators compared to SHAP in highly imbalanced financial datasets?*

## Project Structure

```
.
├── notebook.ipynb        # Main notebook: data loading, model training, SHAP & CEM explanations
├── README.md             # This file
└── output/
    ├── shap_summary_plot.jpg         # Global SHAP feature importance
    ├── shap_force_plot.jpg           # Local SHAP explanation for one transaction
    ├── shap_sparsity.jpg             # SHAP sparsity analysis across 50 fraud cases
    ├── shap_vs_cem_comparison.jpg    # Side-by-side feature comparison
    └── actionability_metrics.csv    # Sparsity results table
```

## Dataset

**ULB Credit Card Fraud Detection** — available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

- 284,807 transactions, 492 fraud cases (0.17% fraud rate)
- Features V1–V28 are PCA-transformed (anonymous), plus `Time` and `Amount`
- Download `creditcard.csv` and place it in the project root before running the notebook

## Requirements

Install dependencies using pip:

```bash
pip install numpy pandas scikit-learn imbalanced-learn shap matplotlib
```

Tested with Python 3.9.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Download the dataset from Kaggle and place `creditcard.csv` in the project root.

3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```

4. Run all cells in order (`Kernel > Restart & Run All`).

## What the Notebook Does

| Step | Description |
|------|-------------|
| Data loading & preprocessing | Loads the dataset, applies `MinMaxScaler`, splits into train/test |
| Class imbalance handling | Applies SMOTE **only to the training set** to avoid data leakage |
| Model training | Trains a `RandomForestClassifier` (100 trees) |
| Model evaluation | Reports ROC-AUC, recall, and precision for the fraud class |
| SHAP explanations | Uses `TreeExplainer` to compute local and global feature attributions |
| CEM explanations | Applies a tabular adaptation of CEM to find Pertinent Positives (PP) and Pertinent Negatives (PN) for three fraud cases |
| Actionability comparison | Compares mean sparsity (number of features) between SHAP and CEM |

## Results Summary

| Method | Mean features for sufficiency | Metric |
|--------|-------------------------------|--------|
| SHAP | 7.3 | Min top-k features for 90% fraud score |
| CEM (PP) | 5.0 | L1-regularised optimisation |

CEM produced sparser explanations on average, suggesting it provides more concise and actionable outputs for fraud investigators.

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*. https://doi.org/10.48550/arXiv.1705.07874
- Dhurandhar, A. et al. (2018). Explanations based on the missing: Towards contrastive explanations with pertinent negatives. *NeurIPS 31*. https://doi.org/10.48550/arXiv.1802.07623
- Machine Learning Group - ULB (2018). Credit Card Fraud Detection Dataset. Kaggle. https://www.kaggle.com/mlg-ulb/creditcardfraud
- Almalki, F., & Masud, M. (2025). Financial fraud detection using explainable AI and stacking ensemble methods. *IEEE Access*. https://doi.org/10.48550/arXiv.2505.10050

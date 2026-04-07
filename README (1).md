# XAI for Fraud Detection: SHAP vs CEM

Comparing SHAP and CEM explanation methods on a Random Forest classifier trained on the ULB Credit Card Fraud Detection dataset.

## Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the same folder as `xai_fraud_creditcard.py`.

## Installation

1. Install [Python 3.9+](https://www.python.org/downloads/) and [VS Code](https://code.visualstudio.com/)
2. Install the required packages by running this in your terminal:

```bash
pip install numpy pandas scikit-learn imbalanced-learn shap matplotlib scipy
```

## Usage

1. Open the project folder in VS Code
2. Open a terminal (`Ctrl+` `)
3. Run the script:

```bash
python xai_fraud_creditcard.py
```

The script will print model evaluation metrics to the terminal and save the following output files in the same folder:

- `shap_summary_plot.png` — global SHAP feature importance across 50 fraud transactions
- `shap_force_plot.png` — local SHAP explanation for a single transaction
- `shap_sparsity.png` — distribution of features needed for 90% score sufficiency
- `shap_vs_cem_comparison.png` — side-by-side SHAP vs CEM feature comparison
- `actionability_metrics.csv` — mean sparsity results for both methods

## References

- Lundberg & Lee (2017). A unified approach to interpreting model predictions. https://doi.org/10.48550/arXiv.1705.07874
- Dhurandhar et al. (2018). Explanations based on the missing. https://doi.org/10.48550/arXiv.1802.07623
- ULB (2018). Credit Card Fraud Detection. https://www.kaggle.com/mlg-ulb/creditcardfraud

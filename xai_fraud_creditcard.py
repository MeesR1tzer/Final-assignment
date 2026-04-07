import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.optimize import minimize
import shap

#Load and preprocess the dataset

df = pd.read_csv("creditcard.csv")
print(f"Dataset shape: {df.shape} | Fraud rate: {df['Class'].mean():.4%}")

X = df.drop(columns=["Class", "Time"]) #Drop "Time" because its irrelevant for fraud detection 
y = df["Class"]

scaler = StandardScaler() 
X["Amount"] = scaler.fit_transform(X[["Amount"]]) #Scale "Amount" because it has a wide range and can dominate the model if left unscaled

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) #Stratify to maintain class distribution in train/test split

X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"Resampled training size: {X_train_res.shape[0]}") 
#SMOTE is used to balance the training data by generating synthetic samples of the minority class (fraud) to improve model performance on imbalanced datasets.

# Train Random Forest

model = RandomForestClassifier(
    n_estimators=200, #200 trees for more stable predictions
    max_depth=10, #Limit depth to prevent overfitting 
    random_state=42,
    n_jobs=-1 #uses all available CPU cores on mac for faster training
)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}") #ROC-AUC evaluates the model at every possible threshold

# SHAP explanations

fraud_sample = X_test[y_test == 1].head(50) #Select 50 fraud transactions for explanation 


explainer = shap.TreeExplainer(model) #Creates a SHAP explainer specifically for tree-based models
shap_values = explainer.shap_values(fraud_sample) #computes SHAP values for all 50 fraud transactions across all features

if isinstance(shap_values, list): #older versions of SHAP return a list of arrays for each class, newer versions return a single array with shape (samples, features, classes)
    shap_fraud = shap_values[1]
else:
    shap_fraud = shap_values[:,:, 1]

shap.summary_plot(shap_fraud, fraud_sample, plot_type="dot", show=False) #plots all 50 transactions as dots 
plt.title("SHAP Summary Plot - Fraud Predictions") 
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("shap_summary_plot.png saved ")

shap.force_plot( #shows the push/pull breakdoown for a single transaction 
    explainer.expected_value[1], shap_fraud[0], fraud_sample.iloc[0],
    matplotlib=True, show=False, figsize=(20, 3)
)
plt.tight_layout()
plt.savefig("shap_force_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("shap_force_plot.png saved")

# SHAP Sparsity

def shap_sufficiency(model, instance, shap_vals, threshold=0.90):
    original_score = model.predict_proba(instance.values.reshape(1, -1))[0][1] # Get the original fraud score for the instance & reshapes the instance into a 2D array and gets the first row 
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1] #sorts features by their SHAP values in descending order (abs makes it so negative values are also considered important)
    for k in range(1, len(shap_vals) + 1): #Loops from k=1 up to the total number of features
        top_k_idx = sorted_idx[:k] #Takes the indices of the top k most important features
        masked_vals = np.zeros(len(instance)) #creates a blank transaction with all features set to zero
        masked_vals[top_k_idx] = instance.values[top_k_idx] #fills in only the top-k features with their real values
        score = model.predict_proba(pd.DataFrame([masked_vals], columns=instance.index))[0][1] #Gets the fraud score for this masked transaction
        if score >= threshold * original_score: #Checks if the score is at least 90% of the original score
            return k
    return len(shap_vals)

sparsity_scores = [shap_sufficiency(model, fraud_sample.iloc[i], shap_fraud[i]) 
                   for i in range(len(fraud_sample))]#Calculates the sparsity score for each of the 50 fraud transactions using list comprehension

print(f"SHAP mean features for 90% sufficiency: {np.mean(sparsity_scores):.1f}")

plt.figure(figsize=(8, 4))
plt.hist(sparsity_scores, bins=range(1, max(sparsity_scores)+2),
         color="steelblue", edgecolor="white", alpha=0.85)
plt.axvline(np.mean(sparsity_scores), color="crimson", linestyle="--",
            label=f"Mean = {np.mean(sparsity_scores):.1f}")
plt.xlabel("Features needed for 90% sufficiency")
plt.ylabel("Count (fraud transactions)")
plt.title("SHAP Sparsity Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("shap_sparsity.png", dpi=150, bbox_inches="tight")
plt.close()
print("shap_sparsity.png saved")

# CEM explanations
#what is the minimal evidence that makes this fraud?
def cem_pertinent_positive(model, instance, lam=0.05, max_features=5): #lam is the regularization strength & max_features limits the output at 5 features
    def objective(mask):#mask is a vector of values between 0 and 1. one per feature
        score = model.predict_proba(pd.DataFrame([instance.values * mask], columns=instance.index))[0][1] 
        return -score + lam * np.sum(np.abs(mask))#multiplies the transaction's feature values by the mask, then asks the model for the fraud probability & objective is to minimize so -score is means maximizing the fraud score 
    result = minimize(objective, x0=np.ones(len(instance)) * 0.5, #x0=0.5 starts every feature at half-strength as a neutral starting point
                      method="L-BFGS-B", bounds=[(0, 1)] * len(instance), #bounds ensure the mask values stay between 0 and 1
                      options={"maxiter": 200})#L-BFGS-B is a gradient-based optimizer that handles bounds efficiently & maxiter=200 limits the number of iterations
    top_idx = np.argsort(result.x)[::-1][:max_features]# after optimization, result.x contains the final mask values & sort them descending and take the top 5 indices
    return {instance.index[i]: round(instance.values[i], 4) for i in top_idx} #Returns a dictionary of feature name and actual feature value for the top 5
#what would have had to be different for this not to be fraud
def cem_pertinent_negative(model, instance, lam=0.05, max_features=5):
    def objective(delta): # how much to add or subtract from each feature value
        score = model.predict_proba(pd.DataFrame([instance.values + delta], columns=instance.index))[0][1]
        return score + lam * np.sum(np.abs(delta)) #here minimize score directly, the goal is to have the fraud probility to as low as possible
    result = minimize(objective, x0=np.zeros(len(instance)),
                      method="L-BFGS-B", options={"maxiter": 200})
    top_idx = np.argsort(np.abs(result.x))[::-1][:max_features] #Sorts by the absolute size of the perturbation
    return {instance.index[i]: round(result.x[i], 4)
            for i in top_idx if abs(result.x[i]) > 1e-3} #filters out features where the pertupation was close to zero

pp_counts = {}
cem_sparsity = []
for i in range(3): #runs both functions on the first 3 fraud transactions and prints the results
    instance = fraud_sample.iloc[i]
    pp = cem_pertinent_positive(model, instance)
    pn = cem_pertinent_negative(model, instance)
    cem_sparsity.append(len(pp)) #records how many features the PP found for each transaction
    for feat in pp:
        pp_counts[feat] = pp_counts.get(feat, 0) + 1 #counts how many times each feature was identified as a PP across the 3 transactions
    print(f"Transaction {i+1} | PP: {list(pp.keys())} | PN: {list(pn.keys())}")

#Comparison plots
mean_shap = np.abs(shap_fraud).mean(axis=0) #takes the absolute SHAP value across all 50 fraud transactions and averages it for each feature
top10_idx = np.argsort(mean_shap)[-10:] #first sorts the features by their mean SHAP value and takes the indices of the top 10 features for the bar plot


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(fraud_sample.columns[top10_idx], mean_shap[top10_idx],
             color="steelblue", alpha=0.85)
axes[0].set_xlabel("Mean |SHAP value|")
axes[0].set_title("SHAP - Top 10 Features")

if pp_counts:
    axes[1].barh(list(pp_counts.keys()), list(pp_counts.values()),
                 color="darkorange", alpha=0.85)
axes[1].set_xlabel("Frequency as Pertinent Positive")
axes[1].set_title("CEM - PP Feature Frequency")
axes[1].set_xlim(0, 3.5)

plt.suptitle("SHAP vs. CEM - Feature Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_vs_cem_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("shap_vs_cem_comparison.png saved")

# Metrics 

summary = pd.DataFrame({
    "Method": ["SHAP", "CEM (PP)"],
    "Mean features for sufficiency": [round(np.mean(sparsity_scores), 1), round(np.mean(cem_sparsity), 1)],
    "Metric": ["Min top-k for 90% score", "L1-regularized optimization"]
})
summary.to_csv("actionability_metrics.csv", index=False)
print("actionability_metrics.csv saved")
print(summary.to_string(index=False))

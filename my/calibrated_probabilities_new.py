# This script extracts the model predictions and the calibrated probabilities produced by the ResPredAI run for Target4.
# It generates calibration and ROC plots, and saves the extracted probabilities.
# It's possibile to compare the theoretical probabilities ('prob_Target4.csv') with the predicted probabilities. 
# To extract the predicted probabilities, run the 'feature-importance' command of respredai: python -m respredai.cli feature-importance --output my/out_run_LR_3x2_new --model LR --target Target4 --direction
# The extracted probabilities will be saved in my/out_run_LR_3x2_new_int/feature_importance/Target4


import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import roc_curve, auc


# save the output for Target4
lr_target4 = joblib.load("my/out_run_LR_3x2_new_int/models/LR_Target4_models.joblib")
print(lr_target4.keys())
print(lr_target4["metrics"].keys())

# y_true, y_pred, y_prob
y_true = lr_target4["metrics"]["all_y_true"]
y_pred = lr_target4["metrics"]["all_y_pred"]
y_prob = lr_target4["metrics"]["all_y_prob"]
y_prob = [np.round(arr, 6) for arr in y_prob]
y_prob_0 = [arr[0] for arr in y_prob]
y_prob_1 = [arr[1] for arr in y_prob]

all_y = pd.DataFrame({
    "y_true": y_true, "y_prob_1": y_prob_1, "y_prob_0": y_prob_0, "y_pred": y_pred})
print(all_y.head())


# true classes
print(f"True Resistant: {sum(y_true)}")
print(f"True Susceptible: {10000 - sum(y_true)}")

# calibrated probabilities
print(f"Cal. Prob. Resistant: {round(sum(y_prob_1), 6)}")
print(f"Cal. Prob. Susceptible: {round(sum(y_prob_0), 6)}")

# predicted classes (1/0)
print(f"Predicted Resistant: {sum(y_pred)}") 
print(f"Predicted Rusceptible: {10000 - sum(y_pred)}")

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# outer test folds' thresholds 
print(lr_target4["fold_thresholds"])

# order of obs in the folds
ordine = lr_target4["metrics"]["all_test_indices"]
ordine = np.array(ordine)
# how the obs are divided in the folds
print(lr_target4["fold_test_data"])

# file theoretical prob.
# prob. of Target4 (last in the for loop of "generate_data.py")
prob = pd.read_csv("my/prob_Target4.csv")
prob = np.array(prob)
print(prob)
prob = prob[ordine] # same order
print(prob) 

# scatter plot 
plt.scatter(prob, y_prob_1, alpha=0.1)
plt.savefig("my/calibrated_prob_Target4_new_int.png", dpi=300, bbox_inches="tight")
plt.close()

# histogram
plt.hist(y_prob_1)
plt.savefig("my/hist_Target4_new_int.png", dpi=300, bbox_inches="tight")
plt.close()

# roc
fpr, tpr, _ = roc_curve(y_true, y_prob_1)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(
    fpr,
    tpr,
    label=f"ROC curve (AUC = {roc_auc:.3f})",
    linewidth=2
)
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("my/roc_curve_Target4_new_int.png", dpi=300, bbox_inches="tight")
plt.close()


# with var4
data = pd.read_csv("my/test_data.csv")
data = data.iloc[ordine] # same order

r_a = data["var4a"] == 1  # obs with var4a=1
y_prob_1_4a = np.array(y_prob_1)[r_a] # estimated prob. that outcome = 1 for obs with var4a = 1
plt.scatter(prob[r_a], np.array(y_prob_1_4a), alpha=0.1) # theoretical prob. vs estimated prob. 
plt.savefig("my/calibrated_prob_Target4_4a.png", dpi=300, bbox_inches="tight")
plt.close()

r_b = data["var4b"] == 1
y_prob_1_4b = np.array(y_prob_1)[r_b]
plt.scatter(prob[r_b], np.array(y_prob_1_4b), alpha=0.1)
plt.savefig("my/calibrated_prob_Target4_4b.png", dpi=300, bbox_inches="tight")
plt.close()

r_c = data["var4c"] == 1
y_prob_1_4c = np.array(y_prob_1)[r_c]
plt.scatter(prob[r_c], np.array(y_prob_1_4c), alpha=0.1)
plt.savefig("my/calibrated_prob_Target4_4c.png", dpi=300, bbox_inches="tight")
plt.close()

# remember var4 has been dropped 


# regressions
print(lr_target4["fold_models"])
# the regressions have intercepts? 
print(lr_target4["fold_models"][0].estimator.estimator.fit_intercept)
print(lr_target4["fold_models"][1].estimator.estimator.fit_intercept)
print(lr_target4["fold_models"][2].estimator.estimator.fit_intercept)


#######################################################################################################################################
# outer folds intercepts 
# (extracted after changing respredai/core/workflow.py) --> unsure!! changes not committed!! 
# do-able when changes are commited
#intercepts = lr_target4["metrics"]["outer_lr_intercepts"]
#print("Outer LR intercepts:")
#for i, intercept in enumerate(intercepts, start=1):
    #print(f"Outer fold {i}: {intercept}")
#######################################################################################################################################


# file estimated prob.
y_prob_1 = np.array(y_prob_1)
print(y_prob_1)

# add column with n. fold to the df
fold_test_data = lr_target4["fold_test_data"]
lista = []
for tupla in fold_test_data: 
    df_fold = tupla[0]
    lista.extend(df_fold.index.tolist())
print(type(fold_test_data[0][0]))

y_prob_1 = (
    pd.DataFrame({"y_prob_1": y_prob_1})
    .assign(
        fold=lambda x: np.select(
            [
                x.index.isin(fold_test_data[0][0].index),
                x.index.isin(fold_test_data[1][0].index),
                x.index.isin(fold_test_data[2][0].index),
            ],
            ["1", "2", "3"],
            default=None
    )
))

print(y_prob_1.head())
y_prob_1.to_csv("my/y_prob_1.csv", index = False)




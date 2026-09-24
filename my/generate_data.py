import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler 

np.random.seed(42)
n = 10000

# features
data = pd.DataFrame({
    "var1": np.random.normal(size=n),
    "var2": np.random.uniform(0, 1, size=n),
    "var3": np.random.binomial(1, np.random.uniform(0, 1, size=n)),
    "var5": np.random.normal(loc=10, scale=3, size=n),   
    "var6": np.random.uniform(0, 100, size=n),           
})
var4 = np.random.binomial(3,np.random.uniform(0, 1, size=n))
data["var4a"] = (var4 == 0).astype(int)
data["var4b"] = (var4 == 1).astype(int)
data["var4c"] = (var4 == 2).astype(int)
data["var4d"] = (var4 == 3).astype(int)
print(data.head())

continuous_features = ["var1", "var2", "var5", "var6"]
categorical_features = ["var3", "var4a", "var4b", "var4c", "var4d"]

X_cont = StandardScaler().fit_transform(data[continuous_features])
X_cat = data[categorical_features].to_numpy()
X = np.column_stack([X_cont, X_cat])
print(X)
print(type(X))

# target variables: each patient is resistant/susceptible to each antibiotic class
antibiotic_classes = ["Target1", "Target2", "Target3", "Target4"]

# coeff --> logit --> prob --> outcome 
data_cols = continuous_features + categorical_features
for antibiotic_class in antibiotic_classes:
    coeff = np.random.normal(size=X.shape[1]) 

    coeff_a = coeff[data_cols.index("var4a")]
    coeff_b = coeff[data_cols.index("var4b")]
    coeff_c = coeff[data_cols.index("var4c")]
    coeff_d = coeff[data_cols.index("var4d")]

    logit = X @ coeff
    # print(f"logit: {logit}")

    # print(pd.Series(logit).describe()) 
    prob = 1 / (1 + np.exp(-logit)) 
    prob_Target4 = pd.DataFrame(prob)
 
    # outcome (1/0)
    data[antibiotic_class] = (prob > np.random.uniform(0, 1, size=n)).astype(int)
    print("\n")

    # new coeff.
    coeff_a_new = coeff_a - coeff_d
    coeff_b_new = coeff_b - coeff_d
    coeff_c_new = coeff_c - coeff_d
    intercept_new = coeff_d

    new_coeff = {
        "var1": coeff[data_cols.index("var1")],
        "var2": coeff[data_cols.index("var2")],
        "var5": coeff[data_cols.index("var5")],
        "var6": coeff[data_cols.index("var6")],
        "var3": coeff[data_cols.index("var3")],
        "var4a": coeff_a_new,
        "var4b": coeff_b_new,
        "var4c": coeff_c_new,
        "intercept": intercept_new
    }

    sorted_coeff = dict(sorted(new_coeff.items(), key=lambda item: item[1], reverse=True))
    print(f"{antibiotic_class}: {sorted_coeff}")

print(data.head())

# contingencies tables
print(pd.crosstab(var4, data["Target1"], normalize=True))
print(pd.crosstab(var4, data["Target2"], normalize=True))
print(pd.crosstab(var4, data["Target3"], normalize=True))
print(pd.crosstab(var4, data["Target4"], normalize=True))

# resistant/susceptible per Target
print(data[antibiotic_classes].apply(pd.Series.value_counts))

# drop var4d (intercept)
data = data.drop(columns=["var4d"])
print(data.head())

# csv data
folder = Path(__file__).parent
data.to_csv(folder / "test_data.csv", index=False)
print(f"path: {folder / 'test_data.csv'}")

# csv prob
prob_Target4.to_csv(folder / "prob_Target4.csv", index = False)
print(f"path: {folder / 'prob_Target4.csv'}")



##########################################################################################################################################

#import statsmodels.formula.api as smf
#import statsmodels.api as sm

#model = sm.Logit(data["Target1"], X).fit()
#dir(model)
#model.__class__
#model.params
#model.bse

#from sklearn.metrics import roc_auc_score
#auc = roc_auc_score(data["Target1"], model.predict())
# auc

#np.quantile(prob[data["Target1"]==0], 0.981)
#np.quantile(prob[data["Target1"]==1], 0.016)

# funzione che max F1score
# np.quantile(prob[data["Target4"]==1], 1-0.916)
# np.quantile(prob[data["Target4"]==0], 0.916)

# np.mean((data["Target4"])[prob<=0.1])

#from sklearn.metrics import precision_recall_curve
#precision, recall, thresholds = precision_recall_curve(data["Target4"],prob)
#f1 = 2 * precision * recall / (precision + recall)
#idx = np.nanargmax(f1)

#best_threshold = thresholds[idx]
#best_f1 = f1[idx]

#print("Best threshold:", best_threshold)
#print("Best F1:", best_f1)
#print(recall[thresholds==best_threshold])
#recall = recall[1:]
#print(recall)

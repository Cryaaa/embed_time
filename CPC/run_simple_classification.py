# %%
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

table_location = Path("/mnt/efs/dlmbl/G-et/tabular_data")
version = 2
model_name = "ben_model_03_pp_norm"
out_tabular_data = table_location / model_name
out_tabular_data.mkdir(exist_ok=True)


latent_spaces = {
    "version_0_high_LR_latents":pd.read_csv(
        out_tabular_data / f"version_{str(0)}" / "latent_dimensions_table.csv"
    ),
    "version_0_high_LR_context":pd.read_csv(
        out_tabular_data / f"version_{str(0)}" / "context_dimensions_table.csv"
    ),
    "version_1_low_LR_latents":pd.read_csv(
        out_tabular_data / f"version_{str(1)}" / "latent_dimensions_table.csv"
    ),
    "version_1_low_LR_context":pd.read_csv(
        out_tabular_data / f"version_{str(1)}" / "context_dimensions_table.csv"
    ),
    "version_2_low_LR&Dims_latents":pd.read_csv(
        out_tabular_data / f"version_{str(2)}" / "latent_dimensions_table.csv"
    ),
    "version_2_low_LR&Dims_context":pd.read_csv(
        out_tabular_data / f"version_{str(2)}" / "context_dimensions_table.csv"
    ),
}

# %%
latent_spaces["version_2_low_LR&Dims_latents"].head()
# %%

gt_keys = ["Dev Outcome","Time","Axes","Run","Plate","ID","Unique Plate"]
results = {}

model_name =["SVC","RF","LDA"]

models = [
    SVC(C=40, kernel='rbf'),
    RandomForestClassifier(random_state=1,n_jobs=10,n_estimators=500,max_features=300),
    LDA(solver='svd'),
]

for name, df in latent_spaces.items():
    gt_df = df[gt_keys]
    df = pd.concat([df[df["Time"]==i].reset_index().drop(gt_keys,axis=1) for i in range(4)]+[gt_df],axis=1).dropna()
    
    results[name] = {}
    y = df["Dev Outcome"]
    X = df.drop(gt_keys,axis=1)
    print(len(X))
    
    for mod_name, model in zip(model_name,models):
        print(f"{name}, {mod_name}")

        # configure the cross-validation procedure
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        # execute the nested cross-validation
        scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv_outer, n_jobs=10)   

        # report performance
        print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        results[name][mod_name] = scores

# %%
all_data = []
for name, diction in results.items():
    for k, v in diction.items():
        scores = v
        all_data.append(pd.DataFrame({"Accuracy":scores,"Classifier":np.full(len(scores), k),"Feature Set":np.full(len(scores), name)}))
    
all_classifier_results_df = pd.concat(all_data,axis=0,ignore_index=True)
all_classifier_results_df.to_csv(out_tabular_data/"classification results per_plate_normalized.csv")
all_classifier_results_df
# %%
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

#out_fig_2 = home_directory + "/" + r"Plots\Fig 2"

sns.set()
fig, ax = plt.subplots(figsize=(5,3))

ax = sns.barplot(all_classifier_results_df,y="Accuracy",x="Feature Set",hue="Classifier", ax = ax, width = 0.8,saturation=1,errorbar=("sd",1),capsize=0.1,errwidth=1)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_ylim([0.4,0.8])
plt.xticks(rotation=90)

plt.savefig(out_tabular_data / "Morph Prediction.pdf", format="pdf", bbox_inches="tight")
plt.show()


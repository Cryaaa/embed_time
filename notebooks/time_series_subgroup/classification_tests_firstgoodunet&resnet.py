# %%
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
tabular_data = "/mnt/efs/dlmbl/G-et/tabular_data"

latent_spaces = {
    "UNet_20z_Old_Normalisation":pd.read_csv(
        Path(tabular_data)/"UNet_VAE_01_old_normalisation.csv"
    ),
    "UNet_20z_New_Normalisation":pd.read_csv(
        Path(tabular_data)/"UNet_VAE_02_new_normalisation.csv"
    ),
    "Resnet18_26000z_Old_Normalisation":pd.read_csv(
        Path(tabular_data)/"LinearVAE_01_bicubic_latents_w_annot.csv"
    ),
    "Resnet18_26000z_New_Normalisation":pd.read_csv(
        Path(tabular_data)/"LinearVAE_02_bicubic_latents_w_annot.csv"
    ),
}

df = latent_spaces["UNet_20z_Old_Normalisation"]
grouped_by_well = df.groupby(["Run","Plate","ID"])
n_samples = len(grouped_by_well)

# %%
group_dict = grouped_by_well.groups
group_keys = list(group_dict.keys())
group_keys[0]
# %
labels = [lab[0] for lab in grouped_by_well["Label"].unique().to_numpy()]
len(labels)
# %%
from sklearn.metrics import balanced_accuracy_score
gt_keys = ["Label","Time","Axes","Run","Plate","ID"]
results = {}

model_name =["SVC","RF","LDA"]

models = [
    SVC(C=40, kernel='rbf'),
    RandomForestClassifier(random_state=1,n_jobs=10,n_estimators=500,max_features=300),
    LDA(solver='svd'),
]

for name, df in latent_spaces.items():
    results[name] = {}
    y = df[gt_keys]
    X = df.drop(gt_keys,axis=1)
    grouped_by_well = df.groupby(["Run","Plate","ID"])
    n_samples = len(grouped_by_well)
    group_dict = grouped_by_well.groups
    group_keys = list(group_dict.keys())
    labels = [lab[0] for lab in grouped_by_well["Label"].unique().to_numpy()]

    for mod_name, model in zip(model_name,models):
        print(f"{name}, {mod_name}")

        # configure the cross-validation procedure
        cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        # execute the nested cross-validation
        scores = []
        for i, (idx_keys_train,idx_keys_test) in enumerate(cv_outer.split(range(n_samples),labels)):
            train_keys = []
            for j in idx_keys_train:
                train_keys.append(group_keys[j])
            train_indices_df = np.concat(
                [group_dict[key] for key in train_keys]
            )
            y_train = y.iloc[train_indices_df]["Label"]=="good"
            test_keys = []
            for j in idx_keys_test:
                test_keys.append(group_keys[j])
            test_indices_df = np.concat(
                [group_dict[key] for key in test_keys]
            )
            y_test=y.iloc[test_indices_df]["Label"] == "good"

            model.fit(X.iloc[train_indices_df],y_train)
            predictions = model.predict(X.iloc[test_indices_df])
            score = balanced_accuracy_score(
                y_true=y_test,
                y_pred=predictions
            )
            print(score)
            scores.append(score)

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
all_classifier_results_df
# %%
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

#out_fig_2 = home_directory + "/" + r"Plots\Fig 2"

sns.set()
fig, ax = plt.subplots(figsize=(5,3))

ax = sns.barplot(all_classifier_results_df,y="Accuracy",x="Feature Set",hue="Classifier", ax = ax, width = 0.8,saturation=1,errorbar=("sd",1),capsize=0.1,errwidth=1)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_ylim([0.4,1])
plt.xticks(rotation=90)

#plt.savefig(f"{out_fig_2}/Morph Prediction.pdf", format="pdf", bbox_inches="tight")
plt.show()
# %%

# %%
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
tabular_data = "/mnt/efs/dlmbl/G-et/tabular_data"

latent_spaces_unet = {
    "UNet_20z_Old_Normalisation":pd.read_csv(Path(tabular_data)/"20240901_UNet_20z_Latent_Space_OldNorm.csv"),
    "UNet_20z_New_Normalisation":pd.read_csv(Path(tabular_data)/"20240901_UNet_20z_Latent_Space_GoodNorm.csv"),
}

annotations_unet = {
    "UNet_20z_Old_Normalisation":pd.read_csv(Path(tabular_data)/"20240901_UNet_20z_Latent_Space_OldNorm_Annotations.csv"),
    "UNet_20z_New_Normalisation":pd.read_csv(Path(tabular_data)/"20240901_UNet_20z_Latent_Space_GoodNorm_Annotations.csv"),
}
# %%
gt_keys = ["Label","Time"]
results = {}

model_name =["SVC","RF","LDA"]

models = [
    SVC(C=40, kernel='rbf'),
    RandomForestClassifier(random_state=1,n_jobs=1,n_estimators=500,max_features=300),
    LDA(solver='svd'),
]

for name, df in latent_spaces_unet.items():
    results[name] = {}
    labels = annotations_unet[name]
    y = labels["Label"].to_numpy() ==  "good"
    X = df
    
    for mod_name, model in zip(model_name,models):
        print(f"{name}, {mod_name}")

        # configure the cross-validation procedure
        cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        # execute the nested cross-validation
        scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv_outer, n_jobs=10)
        # report performance
        print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        results[name][mod_name] = scores

# %%
latent_spaces_resnet = {
    "Resnet18_26000z_Old_Normalisation":pd.read_csv((Path(tabular_data)/"20240901_Resnet_26000z_Latent_Space_OldNorm.csv")),
    "Resnet18_26000z_New_Normalisation":pd.read_csv((Path(tabular_data)/"20240901_Resnet_26000z_Latent_Space_GoodNorm.csv")),
}

annotations_resnet = {
    "Resnet18_26000z_Old_Normalisation":pd.read_csv((Path(tabular_data)/"20240901_Resnet_26000z_Latent_Space_OldNorm_Annotations.csv")),
    "Resnet18_26000z_New_Normalisation":pd.read_csv((Path(tabular_data)/"20240901_Resnet_26000z_Latent_Space_GoodNorm_Annotations.csv")),
}

models = [
    SVC(C=5, kernel='rbf'),
    RandomForestClassifier(random_state=1,n_jobs=1,n_estimators=500,max_features=300),
    LDA(solver='svd'),

]

for name, df in latent_spaces_resnet.items():
    results[name] = {}
    labels = annotations_resnet[name]
    y = labels["Label"].to_numpy() ==  "good"
    X = df
    
    for mod_name, model in zip(model_name,models):
        print(f"{name}, {mod_name}")

        # configure the cross-validation procedure
        cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
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

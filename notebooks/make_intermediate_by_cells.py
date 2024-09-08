# %% Make an intermediate dataset
# 
import pandas as pd

location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_1168.csv"

metadata = pd.read_csv(location)
# %% [markdown]
# The metadata has gene, barcode, stage, cell_idx
# I want to split such that I have fewer cells per gene, stratifying them by split.

sample = metadata.groupby('gene', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=42))

# %%
print("Proportion of split in metadata")
metadata['split'].value_counts() / len(metadata)

# %%
print("Proportion of split in metadata, sampled")
sample['split'].value_counts() / len(sample)
# %%
print(f"Genes OG: {metadata['gene'].nunique()}, Genes Sampled: {sample['gene'].nunique()}")
# %% Save the sampled metadata
sample.to_csv("/mnt/efs/dlmbl/G-et/csv/dataset_split_1168_subsampled.csv", index=False)


# %%

# %% Make an intermediate dataset
# 
import pandas as pd

location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_806.csv"

metadata = pd.read_csv(location)
# %% [markdown]
# The metadata has gene, barcode, stage, cell_idx
# I want to split such that I have fewer genes, stratifying them by split.
# %%
# Sample a subset of genes
unique_genes = metadata['gene'].unique()
sampled_genes = pd.Series(unique_genes).sample(frac=0.02, random_state=42)

# Filter metadata to include only the sampled genes
filtered_metadata = metadata[metadata['gene'].isin(sampled_genes)]

# Ensure the stratification remains the same
train_sample = filtered_metadata[filtered_metadata['split'] == 'train']
val_sample = filtered_metadata[filtered_metadata['split'] == 'val']
test_sample = filtered_metadata[filtered_metadata['split'] == 'test']

# Combine the stratified samples
sample = pd.concat([train_sample, val_sample, test_sample])
# %%
print("Proportion of split in metadata")
metadata['split'].value_counts() / len(metadata)

# %%
print("Proportion of split in metadata, sampled")
sample['split'].value_counts() / len(sample)

# %%
print(f"Genes OG: {metadata['gene'].nunique()}, Genes Sampled: {sample['gene'].nunique()}")
# %% Save the sampled metadata
sample.to_csv(f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{sample['gene'].nunique()}_sampled.csv", index=False)


# %%

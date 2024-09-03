import pandas as pd
import numpy as np


metadata= pd.read_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast.csv")
filtered_metadata = metadata[metadata['Neuromast_ID'] == 0]

# Step 2: Initialize an empty list to store the balanced data
balanced_data = []

# Step 3: Group by 'timepoint' and process each group separately
for timepoint, group in filtered_metadata.groupby('T_value'):
    
    # Step 4: Find the counts for the specific cell types (e.g., 1, 2, 3)
    celltype_counts = group['Cell_Type'].value_counts()
    #print(celltype_counts)
    
    # Determine the minimum count among the three cell types
    min_count = celltype_counts.min()
    print(min_count)

    # Step 5: For each of the three cell types, sample `min_count` rows
    for cell_type in celltype_counts.index:
        sampled_rows = group[group['Cell_Type'] == cell_type].sample(n=min_count, random_state=42)
        balanced_data.append(sampled_rows)
        print(f"Sampled {len(sampled_rows)} rows for cell type {cell_type} in timepoint {timepoint}")

# Step 6: Combine all sampled rows into a single DataFrame
metadata_balanced_train = pd.concat(balanced_data)

# Step 7: Save the balanced DataFrame to a CSV file
metadata_balanced_train.to_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast_balanced_train.csv", index=False)

print("Balanced dataset saved as metadata_balanced_train.csv")

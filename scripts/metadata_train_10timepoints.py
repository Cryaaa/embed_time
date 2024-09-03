import pandas as pd
import numpy as np


metadata= pd.read_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast_balanced_train.csv")
filtered_metadata = metadata[metadata['Neuromast_ID'] == 0]

# Step 2: Initialize an empty list to store the balanced data
test_data = []
# Step 3: Define the specific T_values you want to filter by
target_t_values = [5, 50, 100, 150, 200, 250, 300, 350, 400, 450]

# Step 4: Filter the filtered_metadata DataFrame for the desired T_values
filtered_metadata = filtered_metadata[filtered_metadata['T_value'].isin(target_t_values)]

# Step 5: Group by 'timepoint' and process each group separately
for timepoint, group in filtered_metadata.groupby('T_value'):
    
    # Step 6: Append all the cell types (e.g., 1, 2, 3)
    test_data.append(group)  


# Step 6: Combine all sampled rows into a single DataFrame
metadata_test = pd.concat(test_data)

# Step 7: Save the balanced DataFrame to a CSV file
metadata_test.to_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast_train_T10.csv", index=False)

print("Balanced dataset saved as metadata_balanced_train.csv")
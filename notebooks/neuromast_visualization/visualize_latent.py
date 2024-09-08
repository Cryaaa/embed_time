# %%
# An attempt to create a reactive app to plot the latent space
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
from embed_time.dataset_static import ZarrCellDataset
from torchvision.transforms import v2
from embed_time.static_utils import read_config
from embed_time.neuromast import NeuromastDatasetTest, NeuromastDatasetTrain
import numpy as np
from sklearn.decomposition import PCA

# %% 
location = "/mnt/efs/dlmbl/G-et/data/neuromast/models/"

dataset = NeuromastDatasetTrain()
num_samples = 5
latents = np.load("/mnt/efs/dlmbl/G-et/data/neuromast/models/train_latents_big.npy")
class_information = np.load("/mnt/efs/dlmbl/G-et/data/neuromast/models/train_metadata_big.npy")

assert len(latents) == len(dataset) * num_samples
assert len(class_information) == len(dataset) * num_samples
# %%
# Randomly sample a subset of indices
# np.random.seed(0)
# subsample = 5000
# data_indices = np.random.choice(len(dataset), subsample, replace=False)
# # %%
# indices = [[i]* num_samples for i in data_indices]
# indices = np.array(indices).flatten()

# # %% Subsample the latents and class information
# latents = latents[indices]
# class_information = class_information[indices]

# # %% 
# assert len(latents) == subsample * num_samples
# assert len(class_information) == subsample * num_samples
# %% Run pca
pca = PCA(n_components=2)
data = pca.fit_transform(latents.reshape(latents.shape[0], -1))

df = pd.DataFrame(data, columns=['pc0', 'pc1'])
df["cell type"] = class_information

# %%
app = Dash()
app.layout = html.Div([
    html.H1(children='Latent Space Visualization', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            id='image',
            figure=px.imshow(
                dataset[0][0][0],
                color_continuous_scale='gray'
            ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False),
            style={'width': '400px', 'height': '400px'}
        ),
        dcc.Graph(
            id='latent-space',
            figure=px.scatter(df, x='pc0', y='pc1', color='cell type', opacity=0.3),
            style={'width': '900px', 'height': '600px'}
        )
    ], style={'display': 'flex', 'flexDirection': 'row'})
])

@app.callback(
    Output('image', 'figure'),
    Input('latent-space', 'clickData'),
)
def update_image(clickData):
    if clickData is None:
        return px.imshow(
            dataset[0][0][0],
            color_continuous_scale='gray'
        ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False)
    
    point_index = clickData['points'][0]['pointIndex']
    # Get the sample in the ordered_df
    point_index = indices[point_index]
    print(f"Should switch to {point_index}")
    return px.imshow(
        dataset[point_index][0][0],
        color_continuous_scale='gray'
    ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False)


if __name__ == '__main__':
    app.run(debug=True)

# %%

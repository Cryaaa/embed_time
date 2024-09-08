# %%
# An attempt to create a reactive app to plot the latent space
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
from embed_time.dataset_static import ZarrCellDataset
from torchvision.transforms import v2
from embed_time.static_utils import read_config
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

# %% Load the dataset
# Define the metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']
crop_size = 96
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
dataset = "benchmark_nontargeting_barcode_with_cct2"
csv_file = f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{dataset}.csv"
label_type = 'gene'
balance_classes = True

experiment_directory = Path(f"/mnt/efs/dlmbl/G-et/da_testing/vgg2d_{dataset}/{label_type}_{balance_classes}")

latents_file = experiment_directory / "vgg_latents.npy"
latents = np.load(latents_file)

data_df = pd.read_csv(csv_file)
data_df = data_df[data_df['split'] == 'val'].reset_index()
class_names = data_df[label_type].sort_values().unique().tolist()
num_classes = len(class_names)

# %% Run pca
# TODO should we scale?
pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
data = pipeline.fit_transform(latents)
metadata_options = ['gene', 'barcode', 'stage'] 

# %% Load the training dataset
# Create the dataset
dataset_mean, dataset_std = read_config(yaml_file_path)
dataset = ZarrCellDataset(
    parent_dir = '/mnt/efs/dlmbl/S-md/',
    csv_file = csv_file, 
    split='val',
    channels=[0, 1, 2, 3], 
    mask='min', 
    normalizations=normalizations,
    interpolations=None, 
    mean=dataset_mean, 
    std=dataset_std
)

def turn_into_rgb(image, channel_index=0):
    """
    Turns 4 channel, channel-first tensor from the dataset into a single channel, channel-last numpy array
    """
    return np.transpose(image.numpy(), (1, 2, 0))[:, :, channel_index]


df = pd.DataFrame(data, columns=['pc0', 'pc1'])
for opt in metadata_options:
    df[opt] = data_df[opt].tolist()

app = Dash()
app.layout = html.Div([
    html.H1(children='Latent Space Visualization', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='channel-dropdown',
                options=[{'label': f'Channel {i}', 'value': i} for i in range(4)],
                value=0,
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Graph(
                id='image',
                figure=px.imshow(
                    turn_into_rgb(dataset[0]["cell_image"]),
                    color_continuous_scale='gray'
                ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False),
                style={'width': '400px', 'height': '400px'}
            ),
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='color-dropdown',
                    options=[{'label': f'{label}', 'value': label} for label in metadata_options],
                    value="gene",
                    clearable=False,
                    style={'width': '150px'}
                ),
                dcc.Graph(
                    id='latent-space',
                    figure=px.scatter(df, x='pc0', y='pc1', color='gene', opacity=0.3),
                    style={'width': '900px', 'height': '600px'}
                )
            ]
        )
    ], style={'display': 'flex', 'flexDirection': 'row'})
])

@app.callback(
    Output('image', 'figure'),
    [Input('latent-space', 'clickData'), Input('channel-dropdown', 'value')]
)
def update_image(clickData, channel_index):
    if clickData is None:
        return px.imshow(
            turn_into_rgb(dataset[0]["cell_image"], channel_index),
            color_continuous_scale='gray'
        ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False)
    
    point_index = clickData['points'][0]['pointIndex']
    # Get the sample in the ordered_df
    return px.imshow(
        turn_into_rgb(dataset[point_index]["cell_image"], channel_index),
        color_continuous_scale='gray'
    ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False)


# Callback to change what we color the latent space points by
@app.callback(
    Output('latent-space', 'figure'),
    [Input('color-dropdown', 'value')]
)
def update_latent_space(value):
    return px.scatter(df, x='pc0', y='pc1', color=value, opacity=0.3)
    # return px.scatter(df, x='pc0', y='pc1', color=value, opacity=0.5),



if __name__ == '__main__':
    app.run(debug=True)

# %%

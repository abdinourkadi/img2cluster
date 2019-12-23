# %%
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from skimage.io import imread_collection
import plotly.express as px
import pandas as pd
import os

filelist = []
labellist = []

for root, dirs, files in os.walk("..\\playground\\visualization\\data\\images\\train", topdown=False):
    for folder in dirs:
        folderpath = os.path.join(root, folder, '*.jpg')
        print(folderpath)
        images = imread_collection(folderpath)
        for i in images:
            filelist.append(i.ravel())
            labellist.append(folder)

dict_img = {'image': filelist, 'label': labellist}
df_img = pd.DataFrame(data=dict_img)
tsne_results = pd.read_csv('data\\tsne.csv')
x_values = tsne_results['x'].values
y_values = tsne_results['y'].values

# %%
fig = px.scatter(df_img, x=x_values, y=y_values, color=df_img['label'].values,
                 render_mode='webgl', width=900, height=1000) \
    .for_each_trace(lambda t: t.update(name=t.name.replace("color=", "")))

fig.update_traces(marker_line=dict(width=0, color='DarkSlateGray'))
fig.update_traces(marker=dict(size=2.5))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='T-SNE Viewer'),

    html.Div(children='''
        Img2Cluster: A web application for image clustering and labeling.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)

# %%
# -*- coding: utf-8 -*-
from skimage.io import imread_collection
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from waitress import serve
import pandas as pd
import flask
import dash
import os

file_list = []
label_list = []

for root, dirs, files in os.walk("data\\images\\", topdown=False):
    for folder in dirs:
        folder_path = os.path.join(root, folder, '*.jpg')
        print(folder_path)
        images = imread_collection(folder_path)
        for i in images:
            file_list.append(i.ravel())
            label_list.append(folder)

dict_img = {'image': file_list,
            'label': label_list}
df_img = pd.DataFrame(data=dict_img)

# %%
# csv is pickled and not human readable
tsne = pd.read_pickle('data\\tsne.csv')

fig = px.scatter(tsne, x='x', y='y', color=tsne['label'],
                 render_mode='webgl', width=700, height=800) \
    .for_each_trace(lambda t: t.update(name=t.name.replace("label=", "")))

fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'))
fig.update_traces(marker=dict(size=8))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='T-SNE Viewer'),

    html.Div(children='''
        Img2Cluster: A web application for image clustering and labeling. Two
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    # app.run_server(debug=False, dev_tools_hot_reload=False)
    serve(server, host='0.0.0.0', port=8080, threads=10)

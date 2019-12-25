# %%
# -*- coding: utf-8 -*-
from skimage.io import imread_collection
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from waitress import serve
import json

from dash.dependencies import Input, Output
import pandas as pd
import flask
import dash
import os

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)  # , external_stylesheets=external_stylesheets)

# file_list = []
# label_list = []
#
# for root, dirs, files in os.walk("data/images/", topdown=False):
#     for folder in dirs:
#         folder_path = os.path.join(root, folder, '*.jpg')
#         print(folder_path)
#         images = imread_collection(folder_path)
#         for i in images:
#             file_list.append(i.ravel())
#             label_list.append(folder)
#
# dict_img = {'image': file_list,
#             'label': label_list}
# df_img = pd.DataFrame(data=dict_img)
# add index, tsne x & y
# %%
# csv is pickled and not human readable
tsne = pd.read_pickle('data/tsne.csv')

fig = px.scatter(tsne, x='x', y='y', color=tsne['label'],
                 render_mode='webgl', height=900, width=900, hover_data=['index']) \
    .for_each_trace(lambda t: t.update(name=t.name.replace("label=", "")))

fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'))
fig.update_traces(marker=dict(size=8))

# %%
app.layout = html.Div(className="grid-container", children=[

    html.Div(className = 'title', children = [
        html.H1(children='header'),
        html.Div(children='div child'),
    ]),

    html.Div(className='sidebar', children='menu placeholder'),
    dcc.Graph(
        className='tsnezone',
        id='2d-tsne',
        figure=fig
    ),

    dcc.Graph(
        className='images',
        id='im-graph'
    )
])


@app.callback(Output('im-graph', 'figure'),
              [Input('2d-tsne', 'clickData')])
def display(click_data):
    if click_data:
        sample_idx = int(click_data['points'][0]['customdata'][0])
        sample_image = tsne['image'][sample_idx].reshape(28, 28)
        pic = px.imshow(sample_image, color_continuous_scale='gray')
        pic.update_layout(coloraxis_showscale=False)
        pic.update_layout(width=400, height=400)
        pic.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        return pic
    else:
        print("Caught TypeError")
        return {}


if __name__ == '__main__':
    # app.run_server(debug=True)#, host='0.0.0.0')
    serve(server, host='0.0.0.0', port=8080, threads=10)

# -*- coding: utf-8 -*-
from skimage.io import imread_collection
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import base64
import os

from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from waitress import serve
from io import BytesIO
from PIL import Image
import numpy as np
import flask
import json
import dash

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# path_list = []
# file_list = []
# label_list = []
#
# for root, dirs, files in os.walk("data\\images\\", topdown=False):
#     for folder in dirs:
#         folder_path = os.path.join(root, folder, '*.jpg')
#         print(folder_path)
#         images = imread_collection(folder_path)
#         path_list.extend(images.files)
#         for i in images:
#             file_list.append(i.ravel())
#             label_list.append(folder)
#
# # %%
# tsne_clf = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne_clf.fit_transform(file_list)
#
# # %%
# index_list = range(len(file_list))
# x_values = tsne_results[:, 0]
# y_values = tsne_results[:, 1]
#
# # %%
#
# dict_img = {'index': index_list,
#             'label': label_list,
#             'x': x_values,
#             'y': y_values,
#             'paths': path_list,
#             'image': file_list}
#
# # %%
# df_img = pd.DataFrame(data=dict_img)
# df_img.to_pickle('data\\tsne_pickle.csv')
tsne = pd.read_pickle('data\\tsne_pickle.csv')

# %%
fig = px.scatter(tsne, x='x', y='y', color=tsne['label'],
                 render_mode='webgl', height=700, width=600, hover_data=['index']) \
    .for_each_trace(lambda t: t.update(name=t.name.replace("label=", "")))
fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'), marker=dict(size=8))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# %%
app.layout = html.Div(className="grid-container", children=[

    html.Div(className='title', children=[
        html.H1(children='Image Dataset Viewer'),
        html.Div(children='View and label all of your images in one page'),
    ]),

    html.Div(className='sidebar', children='menu placeholder'),
    dcc.Graph(
        className='graph-panel',
        id='2d-tsne',
        figure=fig
    ),

    html.Div(className='image-panel',
             id='im-graph')
])


def numpy_to_b64(array):
    array = np.uint8(array)
    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64


@app.callback(
    Output('im-graph', 'children'),
    [Input('2d-tsne', 'selectedData')])
def display_selected_data(selectedData):
    if selectedData:
        item_list = []
        for i in selectedData['points']:
            select_idx = int(i['customdata'][0])
            image_np = tsne['image'][select_idx].reshape(28, 28).astype(np.float64)
            image_b64 = numpy_to_b64(image_np)
            img_src = 'data:image;base64,' + image_b64
            item = html.Img(src=img_src,
                            style={"height": "5vh", "border": "0", "float": "left"})
            item_list.append(item)
        return item_list
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True)
    # serve(server, host='0.0.0.0', port=8080, threads=10, debug=True)

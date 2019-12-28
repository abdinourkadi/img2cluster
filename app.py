# -*- coding: utf-8 -*-
from skimage.io import imread_collection
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import os

from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from waitress import serve
import flask
import dash

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

path_list = []
file_list = []
label_list = []

for root, dirs, files in os.walk("data\\images\\", topdown=False):
    for folder in dirs:
        folder_path = os.path.join(root, folder, '*.jpg')
        print(folder_path)
        images = imread_collection(folder_path)
        path_list.extend(images.files)
        for i in images[:10]:
            file_list.append(i.ravel())
            label_list.append(folder)

# %%
tsne_clf = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne_clf.fit_transform(file_list)

# %%
index_list = range(len(file_list))
x_values = tsne_results[:, 0]
y_values = tsne_results[:, 1]

# %%

dict_img = {'index': index_list,
            'label': label_list,
            'x': x_values,
            'y': y_values,
            'paths': path_list,
            'image': file_list}

# %%
df_img = pd.DataFrame(data=dict_img)
df_img.to_pickle('data\\tsne_pickle.csv')
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
        pic.update_layout(coloraxis_showscale=False, width=200, height=200)
        pic.update_layout(margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0))
        pic.update_xaxes(showticklabels=False)
        pic.update_yaxes(showticklabels=False)
        return pic
    else:
        print("Caught TypeError")
        return {}


if __name__ == '__main__':
    # app.run_server(debug=True)#, host='0.0.0.0')
    serve(server, host='0.0.0.0', port=8080, threads=10)

from flask import send_file
import zipfile
import base64
from util import numpy_to_b64, build_df, generate_fig, parse_contents
from dash.dependencies import Input, Output, State

import dash_html_components as html
import dash_core_components as dcc
from waitress import serve
import flask
import dash
import uuid
import time

import urllib.parse
import pandas as pd
import numpy as np
import json
import io
import plotly.express as px
from flask_caching import Cache

cache = Cache()

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)
cache = Cache(app.server, config={
    # 'CACHE_TYPE': 'redis',
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_THRESHOLD': 200
})

cache.clear()

fig = {}
starter_fig = px.scatter()
starter_fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'), marker=dict(size=8))
starter_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

my_session_id = '555'  # str(uuid.uuid4())
global_df = pd.DataFrame()
print("hi")

# %%
app.layout = html.Div(className="grid-container", children=[

    html.Div(className='title', children=[
        html.H1(children='Image Dataset Viewer')
    ]),

    html.Div(id="cluster-control-tab", className='sidebar', children=[
        dcc.Tabs(id="tabs-example", value='about-tab', children=[
            dcc.Tab(label='About', value='about-tab', children=[
                html.Div(className='about-text', children=[
                    html.H4(className='what-is', children='What is Img2Cluster?'),
                    html.P('Img2Cluster is a Dash based image dataset '
                           'viewer and labeler. Using dimensionality'
                           'reduction techniques, view and label your data.'
                           'Img2Cluster is a Dash based image dataset '
                           'viewer and labeler. Using dimensionality'
                           'reduction techniques, view and label your data.'
                           'Img2Cluster is a Dash based image dataset '
                           'viewer and labeler. Using dimensionality'
                           'reduction techniques, view and label your data.'
                           'Img2Cluster is a Dash based image dataset '
                           'viewer and labeler. Using dimensionality'
                           'reduction techniques, view and label your data.'
                           ),
                    html.P('Img2Cluster is a Dash based image dataset '
                           'viewer and labeler. Using dimensionality'
                           'reduction techniques, view and label your data.'
                           'Img2Cluster is a Dash based image dataset '
                           'viewer and labeler. Using dimensionality'
                           'reduction techniques, view and label your data.')
                ])]),

            dcc.Tab(label='Data', value='data-tab',
                    children=html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Data source'),
                            dcc.Dropdown(
                                id='data-dropdown',
                                options=[
                                    {'label': 'Demo Data', 'value': 'Demo Data'},
                                    {'label': 'Upload CSV', 'value': 'Upload CSV'},
                                    {'label': 'Upload Raw Images', 'value': 'Upload Images'}
                                ],
                                value='preloaded')
                        ]),
                        html.Div(id='uploaded-data', children=[
                            dcc.Upload(
                                id="upload-csv",
                                className='control-upload',
                                children=html.Div(
                                    [
                                        "Drag and drop your file here, or click to select"
                                        " your .csv from your local file directory"
                                    ]
                                ),
                                multiple=True),
                            dcc.Upload(
                                id="upload-images",
                                className='control-upload',
                                children=html.Div(
                                    [
                                        "Drag and Drop or "
                                        "click to import "
                                        ".PNG files here!"
                                    ]
                                ),
                                multiple=True),
                        ])
                    ])),
            dcc.Tab(label='Graph', value='graph-tab',
                    children=[
                        html.Div(className='control-tab', children=[
                            html.Div(className='app-controls-block', children=[
                                html.Div(className='app-controls-name', children='Label Selected Cluster'),
                                html.Div(dcc.Input(id='label-input', type='text')),
                                html.Button('Submit', id='label-submit'),
                            ]),
                            html.Div(className='app-controls-block', children=[
                                html.Div(className='app-controls-name', children='Export CSV with new labels'),
                                html.A(html.Button(
                                    id='download-button',
                                    className='control-download',
                                    children="Download Data"),
                                    id='download-link',
                                    href="",
                                    download="downloaded_data.csv")
                            ])
                        ])
                    ])
        ]),
    ]),

    dcc.Graph(
        className='graph-panel',
        id='2d-tsne',
        figure=fig
    ),

    html.Div(className='image-panel',
             id='im-graph'),
    html.Div(id='initial-labels', style={'display': 'none'}),
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(my_session_id, id='session-id', style={'display': 'none'})
])


def get_dataframe(session_id):
    @cache.memoize()
    def query_and_serialize_data(session_id):
        return global_df

    return query_and_serialize_data(session_id)


@app.callback(Output('initial-labels', 'children'),
              [Input('upload-csv', 'contents'),
               Input('upload-csv', 'filename')])
def upload_csv(file, filename):
    global global_df

    if file is not None:
        file = file[0]
        filename = filename[0]
        print(filename)

        if 'csv' in filename:
            # Assume that the user uploaded a CSV file

            _, content_string = file.split(',')
            decoded = base64.b64decode(content_string)

            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

            df = parse_contents(file, filename)
            df = build_df(df)

            global_df = df.copy(deep=True)

            json_list = json.dumps(df['label'].astype(str).values.tolist())
            return json_list

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            _, content_string = file.split(',')
            decoded = base64.b64decode(content_string)

            df = pd.read_excel(io.BytesIO(decoded))

            df = parse_contents(file, filename)
            df = build_df(df)

            global_df = df.copy(deep=True)

            json_list = json.dumps(df['label'].astype(str).values.tolist())
            return json_list

        elif 'zip' in filename:
            zip_file = zipfile.ZipFile("fake_server/zippy.zip", "w")
            zip_file.write(file)
            zip_file.close()

            with zipfile.ZipFile('zippy.zip', 'r') as zip_ref:
                zip_ref.extractall('fake_server/unzipped')

            print("unzipping successful")
    else:
        return None


@app.callback([Output('intermediate-value', 'children'),
               Output('download-link', 'href')],
              [Input('initial-labels', 'children'),
               Input('label-submit', 'n_clicks')],
              [State('2d-tsne', 'selectedData'),
               State('intermediate-value', 'children'),
               State('label-input', 'value')])
def label_cluster_and_update_download(initial_labels, n_clicks, selected_data, label_json, label):
    if initial_labels and n_clicks is None:
        temp_df = get_dataframe(my_session_id)
        label_list = json.loads(initial_labels)

        temp_df['label'] = label_list
        temp_df = temp_df[['paths', 'x', 'y', 'label']]
        temp_df.to_csv('output/downloadFile.csv', index=False)
        csv_string = '/dash/urlToDownload'
        return initial_labels, csv_string

    elif selected_data and (n_clicks > 0):
        label_list = json.loads(label_json)
        label = str(label)

        for i in selected_data['points']:
            select_idx = int(i['customdata'][0])
            label_list[select_idx] = label

        temp_df = get_dataframe(my_session_id)
        temp_df['label'] = label_list
        temp_df = temp_df[['paths', 'x', 'y', 'label']]
        temp_df.to_csv('output/downloadFile.csv', index=False)
        csv_string = '/dash/urlToDownload'
        return json.dumps(label_list), csv_string
    else:
        return label_json, None


@app.server.route('/dash/urlToDownload')
def download_csv():
    return send_file('output/downloadFile.csv',
                     mimetype='csv',
                     attachment_filename='downloadFile.csv',
                     as_attachment=True)


@app.callback(Output('2d-tsne', 'figure'),
              [Input('intermediate-value', 'children')])
def display_graph(label_json):  # , uploaded_df):
    if label_json:
        temp_df = get_dataframe(my_session_id)
        label_list = json.loads(label_json)
        temp_df['label'] = label_list
        return generate_fig(temp_df)
    else:
        return starter_fig


@app.callback(
    Output('upload-csv', 'style'),
    [Input('data-dropdown', 'value')])
def show_hide_csv_upload(selected_drop):
    if selected_drop == 'Upload CSV':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('upload-images', 'style'),
    [Input('data-dropdown', 'value')])
def show_hide_image_upload(selected_drop):
    if selected_drop == 'Upload Images':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('im-graph', 'children'),
    [Input('2d-tsne', 'selectedData')])
def display_selected_data(selected_data):
    if selected_data:

        df = get_dataframe(my_session_id)
        item_list = []

        for i in selected_data['points']:
            select_idx = int(i['customdata'][0])
            img_shape = df['shape'][select_idx]
            image_np = df['image'][select_idx]

            if len(img_shape) == 2:
                image_np = image_np.reshape(img_shape[0], img_shape[1])
            elif len(img_shape) == 3:
                image_np = image_np.reshape(img_shape[0], img_shape[1], img_shape[2])

            image_np = image_np.astype(np.float64)
            image_b64 = numpy_to_b64(image_np)
            img_src = 'data:image;base64,' + image_b64
            item = html.Img(src=img_src,
                            style={"height": "7vh",
                                   "padding": "5px",
                                   "display": "inline-block",
                                   "text-align-last": "left"})
            item_list.append(item)

        return item_list
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True)
    # serve(app.server,port=1001,threads=10)

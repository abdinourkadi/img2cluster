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

my_session_id = '555'  # str(uuid.uuid4())
print(my_session_id)
global_df = pd.DataFrame()

# %%
app.layout = html.Div(className="grid-container", children=[

    html.Div(className='title', children=[
        html.H1(children='Image Dataset Viewer'),
        html.Div(children='View and label all of your images in one page'),
    ]),

    html.Div(className='sidebar', children=[
        dcc.Tabs(id="tabs-example", value='about-tab', children=[
            dcc.Tab(label='About', value='about-tab', children=[
                html.Div(className='control-tab', children=[
                    html.H4(className='what-is', children="What is Img2Cluster?"),

                    html.P('Img2Cluster is a viz of data, and can be used '
                           'to highlight relationships between objects in a dataset '
                           '(e.g., genes that are located on different chromosomes '
                           'in the genome of an organism).'),
                    html.P('A Dash Img2Cluster graph consists of two main parts: the layout '
                           'and the tracks. '
                           'The layout sets the basic parameters of the graph, such as '
                           'radius, ticks, labels, etc; the tracks are graph layouts '
                           'that take in a series of data points to display.'),
                    html.P('The visualizations supported by Dash Circos are: heatmaps, '
                           'chords, highlights, histograms, line, scatter, stack, '
                           'and text graphs.'),
                    html.P('In the "Data" tab, you can opt to use preloaded datasets; '
                           'additionally, you can download sample data that you would '
                           'use with a Dash Circos component, upload that sample data, '
                           'and render it with the "Render" button.'),
                    html.P('In the "Graph" tab, you can choose the type of Circos graph '
                           'to display, control the size of the graph, and access data '
                           'that are generated upon hovering over parts of the graph. '),
                    html.P('In the "Table" tab, you can view the datasets that define '
                           'the parameters of the graph, such as the layout, the '
                           'highlights, and the chords. You can interact with Circos '
                           'through this table by selecting the "Chords" graph in the '
                           '"Graph" tab, then viewing the "Chords" dataset in the '
                           '"Table" tab.')])]),
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
                                        "Drag and Drop or "
                                        "click to import "
                                        ".CSV file here!"
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
                    children=[html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Label Selected Cluster'),
                            html.Div(dcc.Input(id='label-input', type='text')),
                            html.Button('Submit', id='label-submit'),
                        ])]),
                              html.Div(className='control-tab', children=[
                                  html.Div(className='app-controls-block', children=[
                                      html.Div(className='app-controls-name', children='Export CSV with new labels'),
                                      html.A(html.Button(
                                          id='download-button',
                                          className='control-download',
                                          children="Download Data"
                                      ),
                                          id='download-link',
                                          href="",
                                          download="downloaded_data.csv",
                                      )
                                  ]),
                              ])
                              ],
                    ),
        ]),
    ]),

    dcc.Graph(
        className='graph-panel',
        id='2d-tsne',
        figure=fig
    ),

    html.Div(className='image-panel',
             id='im-graph'),
    # html.Div(id='initial-df', style={'display': 'none'}),
    html.Div(id='initial-labels', style={'display': 'none'}),
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(my_session_id, id='session-id', style={'display': 'none'})  # json_list)
])


def get_dataframe(session_id):
    start = time.time()

    @cache.memoize()
    def query_and_serialize_data(session_id):
        return global_df  # .to_json()

    print("get dataframe time: ", ((time.time()) - start))
    return query_and_serialize_data(session_id)


@app.callback(Output('initial-labels', 'children'),
              [Input('upload-csv', 'contents'),
               Input('upload-csv', 'filename')])
def upload_csv(file, filename):
    if file is not None:
        df = parse_contents(file, filename)
        df = build_df(df)

        global global_df
        global_df = df.copy(deep=True)

        json_list = json.dumps(df['label'].astype(str).values.tolist())
        return json_list
    else:
        return None


@app.callback([Output('intermediate-value', 'children'),
               Output('download-link', 'href')],
              [  # Input('initial-df', 'children'),
                  Input('initial-labels', 'children'),
                  Input('label-submit', 'n_clicks')],
              [State('2d-tsne', 'selectedData'),
               State('intermediate-value', 'children'),
               State('label-input', 'value')])
def label_cluster_and_update_download(initial_labels, n_clicks, selectedData, label_json, label):
    # print('initial labels: ', initial_labels)
    if initial_labels and n_clicks is None:
        labelstart = time.time()
        temp_df = get_dataframe(my_session_id)
        label_list = json.loads(initial_labels)

        temp_df['label'] = label_list
        temp_df = temp_df[['paths', 'x', 'y', 'label']]
        csv_string = temp_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        print("initial_labels time: ", ((time.time()) - labelstart))
        return initial_labels, csv_string

    elif selectedData and (n_clicks > 0):
        selectstart = time.time()
        label_list = json.loads(label_json)
        label = str(label)

        for i in selectedData['points']:
            select_idx = int(i['customdata'][0])
            label_list[select_idx] = label

        temp_df = get_dataframe(my_session_id)
        temp_df['label'] = label_list
        temp_df = temp_df[['paths', 'x', 'y', 'label']]
        csv_string = temp_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        print("selectstart time: ", ((time.time()) - selectstart))
        return json.dumps(label_list), csv_string
    else:
        return label_json, None


@app.callback(Output('2d-tsne', 'figure'),
              [Input('intermediate-value', 'children')])
def display_graph(label_json):  # , uploaded_df):
    # print('label json: ', label_json)
    if label_json:
        display_time = time.time()
        # temp_df = pd.read_json(uploaded_df)
        temp_df = get_dataframe(my_session_id)
        label_list = json.loads(label_json)
        temp_df['label'] = label_list
        print("display time: ", ((time.time()) - display_time))
        return generate_fig(temp_df)
    else:
        return {}


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
def display_selected_data(selectedData):
    if selectedData:

        df = get_dataframe(my_session_id)
        time1 = time.time()
        # df['image'] = df['image'].apply(lambda x: np.array(x))
        time2 = time.time()
        print("if statement: ", time2 - time1)
        item_list = []

        for i in selectedData['points']:
            select_idx = int(i['customdata'][0])
            image_np = df['image'][select_idx].reshape(28, 28).astype(np.float64)
            image_b64 = numpy_to_b64(image_np)
            img_src = 'data:image;base64,' + image_b64
            item = html.Img(src=img_src,
                            style={"height": "5vh",
                                   "border": "0",
                                   "float": "left"})
            item_list.append(item)
            # time3 = time.time()
            # print("for loop total: ", time3 - time2)
        # time4 = time.time()
        # print("display selected data total:", time4 - time1)
        return item_list
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True)
    # serve(app.server,port=1001,threads=10)

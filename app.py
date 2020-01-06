from util import numpy_to_b64, build_df, generate_fig, parse_contents
from dash.dependencies import Input, Output, State

import dash_html_components as html
import dash_core_components as dcc
import flask
import dash

import urllib.parse
import pandas as pd
import numpy as np
import json

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# tsne = pd.read_csv('data\\input.csv')
# tsne = build_df(tsne)
# json_list = json.dumps(tsne['label'].values.tolist())

# %%
fig = {}#None  # generate_fig(tsne)

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
    html.Div(id='initial-df', style={'display': 'none'}),
    html.Div(id='initial-labels', style={'display': 'none'}),
    html.Div(id='df-value', style={'display': 'none'}),
    html.Div(id='intermediate-value', style={'display': 'none'})  # , children=json_list)
])


#
# @app.callback(Output('df-value', 'children'),
#               [Input('df-value', 'children')])
# def clean_df(uploaded_df):
#     df = build_df(pd.read_json(uploaded_df))
#     return df.to_json()


@app.callback([Output('initial-df', 'children'),
               Output('initial-labels', 'children')],
              [Input('upload-csv', 'contents'),
               Input('upload-csv', 'filename')])  # last one was state
def upload_csv(file, filename):
    if file is not None:
        df = parse_contents(file, filename)
        df = build_df(df)
        json_list = json.dumps(df['label'].astype(str).values.tolist())
        return df.to_json(), json_list
    else:
        return None, None


@app.callback([Output('intermediate-value', 'children'),
               Output('download-link', 'href')],
              [Input('initial-df', 'children'),
               Input('initial-labels', 'children'),
               Input('label-submit', 'n_clicks')],
              [State('2d-tsne', 'selectedData'),
               State('intermediate-value', 'children'),
               State('label-input', 'value'),
               State('df-value', 'children')])
def label_cluster_and_update_download(initial_df, initial_labels, n_clicks, selectedData, label_json, label, uploaded_df):
    if selectedData and (n_clicks == 0):
        temp_df = pd.read_json(initial_df)
        label_list = json.loads(initial_labels)

        temp_df['label'] = label_list
        temp_df = temp_df[['paths', 'x', 'y', 'label']]
        csv_string = temp_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return initial_labels, csv_string

    elif selectedData and n_clicks > 0:
        label_list = json.loads(label_json)
        label = str(label)

        for i in selectedData['points']:
            select_idx = int(i['customdata'][0])
            label_list[select_idx] = label

        # temp_df = tsne.copy(deep=True)
        temp_df = pd.read_json(uploaded_df)
        temp_df['label'] = label_list
        temp_df = temp_df[['paths', 'x', 'y', 'label']]
        csv_string = temp_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return json.dumps(label_list), csv_string
    else:
        return label_json, None


@app.callback(Output('2d-tsne', 'figure'),
              [Input('intermediate-value', 'children')],
              [State('df-value', 'children')])
def display_graph(label_json, uploaded_df):
    # temp_df = tsne.copy(deep=True)
    if label_json and uploaded_df:
        temp_df = pd.read_json(uploaded_df)
        label_list = json.loads(label_json)
        temp_df['label'] = label_list
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
    [Input('2d-tsne', 'selectedData')],
    [State('df-value', 'children')])
def display_selected_data(selectedData, uploaded_df):
    if selectedData and uploaded_df:
        df = pd.read_json(uploaded_df)
        item_list = []
        for i in selectedData['points']:
            select_idx = int(i['customdata'][0])
            # image_np = tsne['image'][select_idx].reshape(28, 28).astype(np.float64)
            image_np = df['image'][select_idx].reshape(28, 28).astype(np.float64)
            image_b64 = numpy_to_b64(image_np)
            img_src = 'data:image;base64,' + image_b64
            item = html.Img(src=img_src,
                            style={"height": "5vh",
                                   "border": "0",
                                   "float": "left"})
            item_list.append(item)
        return item_list
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True)

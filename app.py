import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

import numpy as np
import flask
import dash

from util import numpy_to_b64
from util import build_df

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

tsne = pd.read_csv('data\\blank_sample.csv')
tsne = build_df(tsne)

# %%
fig = px.scatter(tsne, x='x', y='y', color=tsne['label'],  # 820 700
                 render_mode='webgl', height=750, width=700, hover_data=['index']) \
    .for_each_trace(lambda t: t.update(name=t.name.replace("label=", "")))
fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'), marker=dict(size=8))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
                                      html.A(
                                          html.Button(
                                              id='download-button',
                                              className='control-download',
                                              children="Download Data"
                                          ),
                                          href="",
                                          download="labeled_data.csv",
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
             id='im-graph')
])


# TODO: implement after selecting/labeling is working
# @app.callback(
#     dash.dependencies.Output('download-button', 'href'),
#     [dash.dependencies.Input('field-dropdown', 'value')])
# def update_download_link(filter_value):
#     dff = filter_data(filter_value)
#     csv_string = dff.to_csv(index=False, encoding='utf-8')
#     csv_string = "data:text/csv;charset=utf-8," + urllib.quote(csv_string)
#     return csv_string


@app.callback(
    Output('upload-csv', 'style'),
    [Input('data-dropdown', 'value')])
def show_hide_uploaded(selected_drop):
    if selected_drop == 'Upload CSV':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}

#TODO rename this function, duplicate name with the one below it
@app.callback(
    Output('upload-images', 'style'),
    [Input('data-dropdown', 'value')])
def show_hide_uploaded(selected_drop):
    if selected_drop == 'Upload Images':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}


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
                            style={"height": "5vh",
                                   "border": "0",
                                   "float": "left"})
            item_list.append(item)
        return item_list
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True)
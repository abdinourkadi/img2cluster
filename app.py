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
strtest = 'data:image;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABxElEQVR4nL2SvWtTYRjFf+/HvfdNbntNtdJWO7kUF+ngUJSg4Ad2E7qLODj0v3ARBKEgKG6CdhU/QDcH0clJxUECihqbJaE1vc29TZO8j0PS0MTdsx6ec87zPAeG0AaIlKYECSMIARNbFAGBZhy6oCCheOpHus3hUc45cAGzq93Uyz+TWJi78kX8bnUhGmUiCNXcrZZs+5/zjJnaBMpvRNKPN+1EiUMjpIELTzuSrocKo/ZlLVoRUODu3h/fu8ykBooDMgCrDTySVJoXIzTMHExiHPpOlsmrs4A1CpgEQAWAYlHEr5UohpaSIp7qO6oOpidcbal7t9PisaXj0yvPaw+39nWVg/C9VMLC+Sd1SUUyeTZcxVh0UKlfW/rwS5oi0pCd6sENLZ9y+d6RXNr5V9kVud5/hVK92HZB92a9+Lc3lk+uvNiRc244iDGvJRdp3g8MBh5nvwf38Tps96jWp2klq92Nmcql2pnCRtwCiCGAxdq3RkMy6eTSkr2arPVVQ9Cw8DKeOP1uU0QambS38iQGUOB813O07op+vrxZPrH8YP1z0sz6eYwBp60D3BEiYEppNezGQJ0CxForFI5wvEP/C38B9LCmuPotr8gAAAAASUVORK5CYII='


def numpy_to_b64(array, scalar=True):
    if scalar:

        array = np.uint8(array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64


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
    html.Div(  # [html.Img(src=strtest)],#'data:image;base64,{}'.format(strtest))],
        className='images',
        id='im-graph',
        children="wtf is going on"
    )

    # html.Div(
    #         className='images',
    #         id='im-graph',
    #         children=[html.Img(src="data:image;base64, "
    #                                "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8"
    #                                "/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==")]
    #     )
])


@app.callback(Output(component_id='im-graph', component_property='children'),
              [Input('2d-tsne', 'clickData')])
def generate_image(click_data):
    if click_data:
        click_idx = int(click_data['points'][0]['customdata'][0])
        image_np = tsne['image'][click_idx].reshape(28, 28).astype(np.float64)
        image_b64 = numpy_to_b64(image_np)  # .decode()
        # return html.Img(
        #     src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABuUlEQVR4nL2Su2sUURTGf/feuXceu2OSlSBoI0FFsRC18IFi7HwgWgT8AxQRbMRGJAgWNqJgY2WtsbGyELUNCsEqIIKKwUIwj8V1N7vjZCeTY7Ezi7v2ft3h43e+78CBv6Q9BSEoNAMKqGHRpsq4c9hBE2vRVUxvGBkyt8CojS59+b03Mnj8o3FuJhs/Kvjlgn4ZnGP3N/l8wRCrYcxR9d7I0ouKA3Q04CkDp9NUJsES9QFQYH38V5JnV1GgoiJUKc95AOH22YYsKJQzaBMWrFfEz3Tl03GIYTAQTXhX1puH4cDJY0fObi1Ao0E5D563l+ewlfeLkq48m+pjxiqYzPP6+erb9UX5+lTkVrU0PQ0jL9tyndlEZP4M99eu+QAEYGHs4M+0fkIvdORehH9x9VGvSYrNfBrnQm9z8mCbW5pJvImHeasfiVPxdN5dvdOS5aPw+EPW3LWpF4gFzKlmJo20KUxJXZ6Uj6AJFBBf2Uhz6XzfNy+/buyk5grbGu1gx5xIR6S9svY6iKF3itKANTDxTrJE2h8v74HRcm9QFtt/uyVdOVSDSkniQ2DQDpxC4TMGZvj7/p/+AMJWiMiAw//fAAAAAElFTkSuQmCC",
        #     style={"height": "25vh", "display": "block", "margin": "auto"},
        # )
        finalstr = 'data:image;base64,' + image_b64
        print(finalstr)
        return html.Div(children='What the fuck')
        # return html.Img(
        #     src=finalstr#+ image_b64,
        #     #style={"height": "25vh", "display": "block", "margin": "auto"},
        # )
    else:
        print("Caught TypeError")
        return {}


# @app.callback(Output('im-graph', 'figure'),
#               [Input('2d-tsne', 'clickData')])
# def display(click_data):
#     if click_data:
#         sample_idx = int(click_data['points'][0]['customdata'][0])
#         sample_image = tsne['image'][sample_idx].reshape(28, 28)
#         pic = px.imshow(sample_image, color_continuous_scale='gray')
#         pic.update_layout(coloraxis_showscale=False, width=200, height=200)
#         pic.update_layout(margin=dict(
#             l=0,
#             r=0,
#             b=0,
#             t=0,
#             pad=0))
#         pic.update_xaxes(showticklabels=False)
#         pic.update_yaxes(showticklabels=False)
#         return pic
#     else:
#         print("Caught TypeError")
#         return {}


if __name__ == '__main__':
    # app.run_server(debug=True)#, host='0.0.0.0')
    serve(server, host='0.0.0.0', port=8080, threads=10)

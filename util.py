from skimage.io import imread_collection
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import base64
import os
import io


def parse_contents(file, filename):
    """
    parsing the contents of uploaded
    file and returning as dataframe
    """

    _, content_string = file.split(',')
    decoded = base64.b64decode(content_string)

    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        df = pd.read_excel(io.BytesIO(decoded))
    return df


def generate_fig(tsne):
    """
    generate the interactive graph given
    a dataframe with x,y,index, and label
    """
    fig = px.scatter(tsne, x='x', y='y', color=tsne['label'],
                     render_mode='webgl', hover_data=['index']) \
        .for_each_trace(lambda t: t.update(name=t.name.replace("label=", "")))
    fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'), marker=dict(size=8))
    fig.update_layout(
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def generate_master():
    """
    generate master df for local testing.
    reads in all 42000 MNIST images with
    labels by folder
    """

    path_list = []
    image_list = []
    label_list = []

    for root, dirs, files in os.walk("data/images/", topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder, '*.jpg')
            print(folder_path)
            images = imread_collection(folder_path)
            path_list.extend(images.files)

            for i in images:
                image_list.append(i.ravel())
                label_list.append(folder)

    tsne_clf = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne_clf.fit_transform(image_list)

    index_list = range(len(image_list))
    x_values = tsne_results[:, 0]
    y_values = tsne_results[:, 1]

    dict_img = {'index': index_list,
                'label': label_list,
                'x': x_values,
                'y': y_values,
                'paths': path_list,
                'image': image_list}

    df_img = pd.DataFrame(data=dict_img)
    return df_img


def build_df(df):
    """
    select the only 4 needed columns
    resent and append index column
    replace nans in label with blank string ''
    read all the images from the paths column and add to images column
    """

    df = df[['paths', 'x', 'y', 'label']]
    df = df.reset_index(drop=True)

    df['index'] = df.index.values
    df['label'] = df['label'].fillna('')
    image_list = []
    shape_list = []
    images = imread_collection(df['paths'].tolist())

    for i in images:
        shape_list.append(np.array(i.shape).tolist())
        image_list.append(i.ravel())

    df['image'] = image_list
    df['shape'] = shape_list
    return df


def numpy_to_b64(array):
    """
    converts image array to base64
    encoded string to be directly
    embedded in HTML page
    """
    array = np.uint8(array)
    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64

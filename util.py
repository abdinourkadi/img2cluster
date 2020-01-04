from skimage.io import imread_collection
from sklearn.manifold import TSNE
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import base64
import os


def generate_master():
    """
    generate master df for local testing.
    reads in all 42000 MNIST images with
    labels by folder
    """

    path_list = []
    image_list = []
    label_list = []

    for root, dirs, files in os.walk("data\\images\\", topdown=False):
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

    path_list = df['paths'].values
    image_list = []

    images = imread_collection(path_list)
    for i in images:
        image_list.append(i.ravel())

    df['image'] = image_list
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

import os
import time
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def save_plot(path):
    if not os.path.exists(path):
        plt.savefig(path)


def measure_time(start=None):
    end = time.time()
    return float(f'{(end - start):.1f}') if start is not None else float(f'{end:.1f}')


def format_float(number):
    return float(f'{number:.4f}')


def format_percent(number):
    return float(f'{(number * 100):.2f}')


def train_test_from(x, y, fraction=0.3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=fraction, random_state=42)
    return x_train, y_train, x_test, y_test


def ravel(column):
    return column.values.ravel() if isinstance(column, pd.DataFrame) else np.array(column).ravel()


def use_pca(train, test, dims, reverse=False):
    pca = PCA(n_components=dims)
    train = pca.fit_transform(train)
    variance = np.sum(pca.explained_variance_ratio_)
    test = pca.transform(test)
    if reverse:
        train = pca.inverse_transform(train)
        test = pca.inverse_transform(test)
    return train, test, variance


def all_model_params(model):
    replacement = ''
    for p, v in model.get_params().items():
        replacement += f'{p}={v}, '
    return f'{re.search(r'^(.*)\(', str(model)).group(1)}({replacement[:-2]})'

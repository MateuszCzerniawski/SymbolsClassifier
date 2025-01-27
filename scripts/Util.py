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


def collect_best_nets(input_dir, output_path, min_accuracy=0.95, net_label='NET', min_dim=20, max_dim=60):
    best = pd.DataFrame()
    names = [name for name in os.listdir(input_dir) if net_label in name]
    names = [name for name in names if min_dim <= int(re.search(r'(\d+)$', name).group(1)) <= max_dim]
    for filename in names:
        data = pd.read_csv(f'{input_dir}/{filename}', index_col=0)
        data = data[data['accuracy'] >= min_accuracy].drop('loss', axis=1).drop('mae', axis=1)
        data['accuracy'] = data['accuracy'].apply(lambda x: format_float(x))
        best = pd.concat([best, data], ignore_index=True)
    subset = list(best.columns)
    subset.remove('accuracy')
    subset.remove('time')
    best = best.drop_duplicates(subset=subset, keep='first')
    best = best.sort_values(by='accuracy', ascending=False)
    best.reset_index()
    best.to_csv(output_path, index=False)

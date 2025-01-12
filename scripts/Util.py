import os
import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


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

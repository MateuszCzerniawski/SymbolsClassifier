import numpy as np
import pandas as pd

import DataManipulator
import Visualizer
import Trees
import Neural
import Util

if __name__ == "__main__":
    print('processing data')
    DataManipulator.process('../data/images/raw', '../data/images/parsed', '../data/in_csv', [2, 4])
    print('visualising data')
    Visualizer.visualise_symbols('../data/images/parsed', '../data/images/symbols', f'../graphs/blended.png')
    print('loading data')
    x = DataManipulator.load('../data/in_csv/original_x', decompress=True)
    y = DataManipulator.load('../data/in_csv/original_y')
    y = y.values.ravel() if isinstance(y, pd.DataFrame) else np.array(y).ravel()
    data = Util.train_test_from(x, y)
    print('conducting tests for trees:')
    Trees.conduct_all(data)
    print('conducting tests for nets')
    y = Neural.categorise(y)
    data = Util.train_test_from(x, y)
    results = Neural.conduct_all(data, path='../results/nets')

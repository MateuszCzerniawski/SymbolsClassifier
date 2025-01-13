import itertools

import numpy as np
import pandas as pd

import Util
import multiprocessing as mp

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

criteria = ['gini', 'entropy', 'log_loss']
max_depths = [None, 10, 20, 30]
n_estimators = [25, 50, 100, 125, 150]
min_splits = [2, 5, 10]
min_leaves = [1, 2, 4, 5]
max_features = ['sqrt', 'log2', None]
max_leaf_nodes = [None, 10, 20, 50]
bootstrap = [True, False]
oob_score = [True, False]
learning_rates = [0.01, 0.1, 0.3]
subsamples = [0.6, 0.8, 1.0]
colsample_bytrees = [0.6, 0.8, 1.0]
gammas = [0, 1, 5]
min_child_weight = [1, 5, 10]
reg = [('l1', 0), ('l1', 0.1), ('l1', 1), ('l2', 1), ('l2', 10), ('l2', 100)]

model_hiperparameters = {
    'DecisionTreeClassifier': [criteria, max_depths, min_splits, min_leaves, max_features, max_leaf_nodes],
    'ExtraTreesClassifier': [criteria, n_estimators, max_depths, min_splits, min_leaves, max_features, bootstrap],
    'RandomForestClassifier': [criteria, n_estimators, max_depths, min_splits, min_leaves, max_features, bootstrap,
                               oob_score],
    'XGBClassifier': [n_estimators, max_depths, learning_rates, subsamples, colsample_bytrees, gammas,
                      min_child_weight, reg]
}


def combine_params(model):
    return list(itertools.product(*model_hiperparameters[model]))


def train_test_model(model, data):
    now = Util.measure_time()
    model.fit(data[0], data[1])
    predictions = model.predict(data[2])
    accuracy = accuracy_score(data[3], predictions)
    now = Util.measure_time(now)
    return model, predictions, accuracy, now


def use_decision_tree(args):
    data, params = args
    model = DecisionTreeClassifier() if params is None else \
        DecisionTreeClassifier(criterion=params[0], max_depth=params[1], min_samples_split=params[2],
                               min_samples_leaf=params[3], max_features=params[4], max_leaf_nodes=params[5])
    return train_test_model(model, data)


def use_extra_tree(args):
    data, params = args
    model = ExtraTreesClassifier() if params is None else \
        ExtraTreesClassifier(criterion=params[0], n_estimators=params[1], max_depth=params[2],
                             min_samples_split=params[3], min_samples_leaf=params[4],
                             max_features=params[5], bootstrap=params[6])
    return train_test_model(model, data)


def use_random_forest(args):
    data, params = args
    model = RandomForestClassifier() if params is None else \
        RandomForestClassifier(criterion=params[0], n_estimators=params[1], max_depth=params[2],
                               min_samples_split=params[3], min_samples_leaf=params[4],
                               max_features=params[5], bootstrap=params[6], oob_score=params[7])
    return train_test_model(model, data)


def use_xgboost(args):
    data, params = args
    model = None
    if params is None:
        model = XGBClassifier()
    elif params[7][0] == 'l1':
        mmodel = XGBClassifier(n_estimators=params[0], max_depth=params[1],
                               learning_rate=params[2], subsample=params[3], colsample_bytree=params[4],
                               gamma=params[5], min_child_weight=params[6], reg_alpha=params[7][1])
    else:
        model = XGBClassifier(n_estimators=params[0], max_depth=params[1],
                              learning_rate=params[2], subsample=params[3], colsample_bytree=params[4],
                              gamma=params[5], min_child_weight=params[6], reg_lambda=params[7][1])
    return train_test_model(model, data)


def conduct_tests(model, data, path=None):
    params = [(data, p) for p in combine_params(model)]
    function = {
        'DecisionTreeClassifier': use_decision_tree,
        'ExtraTreesClassifier': use_extra_tree,
        'RandomForestClassifier': use_random_forest,
        'XGBClassifier': use_xgboost
    }[model]
    with mp.Pool(processes=20) as pool:
        results = [list(i) for i in pool.map(function, params)]
        results = pd.DataFrame(results)
        if path is not None:
            results.to_csv(path)
        return results


def conduct_all(data):
    print('DecisionTreeClassifier')
    conduct_tests('DecisionTreeClassifier', data, path='../results/DecisionTree_res')
    print('ExtraTreesClassifier')
    conduct_tests('ExtraTreesClassifier', data, path='../results/ExtraTrees_res')
    print('RandomForestClassifier')
    conduct_tests('RandomForestClassifier', data, path='../results/RandomForest_res')
    print('XGBClassifier')
    conduct_tests('XGBClassifier', data, path='../results/XGB_res')


if __name__ == "__main__":
    import DataManipulator

    x = DataManipulator.load('../data/in_csv/original_x', decompress=True)
    y = DataManipulator.load('../data/in_csv/original_y')
    y = y.values.ravel() if isinstance(y, pd.DataFrame) else np.array(y).ravel()
    data = Util.train_test_from(x, y)
    conduct_all(data)

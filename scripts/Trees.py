import itertools
import sys

import pandas as pd

import Util
import multiprocessing as mp

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from scripts import DataManipulator

criteria = ['gini', 'entropy', 'log_loss']
max_depths = [None, 10, 20, 30]
n_estimators = [50, 100, 125, 150]
min_splits = [2, 5, 10]
min_leaves = [2, 4, 5]
max_features = ['sqrt', 'log2', None]
learning_rates = [0.01, 0.1, 0.3]
gammas = [0, 1, 5]
min_child_weight = [1, 5, 10]
reg = [('l1', 0.1), ('l1', 0.5), ('l1', 1), ('l2', 1), ('l2', 10), ('l2', 100)]

model_hiperparameters = {
    'DecisionTreeClassifier': [criteria, max_depths, min_splits, min_leaves, max_features],
    'ExtraTreesClassifier': [criteria, n_estimators, max_depths, min_splits, min_leaves, max_features],
    'RandomForestClassifier': [criteria, n_estimators, max_depths, min_splits, min_leaves, max_features],
    'XGBClassifier': [n_estimators, max_depths, learning_rates, gammas, min_child_weight, reg]
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
    data, params = args if len(args) == 2 else (args, None)
    model = DecisionTreeClassifier() if params is None else \
        DecisionTreeClassifier(criterion=params[0], max_depth=params[1], min_samples_split=params[2],
                               min_samples_leaf=params[3], max_features=params[4])
    return train_test_model(model, data)


def use_extra_tree(args):
    data, params = args if len(args) == 2 else (args, None)
    model = ExtraTreesClassifier() if params is None else \
        ExtraTreesClassifier(criterion=params[0], n_estimators=params[1], max_depth=params[2],
                             min_samples_split=params[3], min_samples_leaf=params[4],
                             max_features=params[5])
    return train_test_model(model, data)


def use_random_forest(args):
    data, params = args if len(args) == 2 else (args, None)
    model = RandomForestClassifier() if params is None else \
        RandomForestClassifier(criterion=params[0], n_estimators=params[1], max_depth=params[2],
                               min_samples_split=params[3], min_samples_leaf=params[4],
                               max_features=params[5])
    return train_test_model(model, data)


def use_xgboost(args):
    data, params = args if len(args) == 2 else (args, None)
    model = None
    if params is None:
        model = XGBClassifier()
    elif params[5][0] == 'l1':
        model = XGBClassifier(n_estimators=params[0], max_depth=params[1],
                              learning_rate=params[2], gamma=params[3],
                              min_child_weight=params[4], reg_alpha=params[5][1])
    else:
        model = XGBClassifier(n_estimators=params[0], max_depth=params[1],
                              learning_rate=params[2], gamma=params[3],
                              min_child_weight=params[4], reg_lambda=params[5][1])
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
        results = pd.DataFrame([[Util.all_model_params(i[0]), i[2], i[3]] for i in pool.map(function, params)])
        results.columns = ['model', 'accuracy', 'time']
        if path is not None:
            results.to_csv(path)
        return results


def conduct_all(data, output_dir, label=''):
    t = Util.measure_time()
    print('DecisionTreeClassifier')
    conduct_tests('DecisionTreeClassifier', data, path=f'{output_dir}/DT{label}')
    print(f'tested {len(combine_params('DecisionTreeClassifier'))} in {Util.format_float(Util.measure_time(t))}s')
    t = Util.measure_time()
    print('ExtraTreesClassifier')
    conduct_tests('ExtraTreesClassifier', data, path=f'{output_dir}/ET{label}')
    print(f'tested {len(combine_params('ExtraTreesClassifier'))} in {Util.format_float(Util.measure_time(t))}s')
    t = Util.measure_time()
    print('RandomForestClassifier')
    conduct_tests('RandomForestClassifier', data, path=f'{output_dir}/RF{label}')
    print(f'tested {len(combine_params('RandomForestClassifier'))} in {Util.format_float(Util.measure_time(t))}s')
    t = Util.measure_time()
    print('XGBClassifier')
    conduct_tests('XGBClassifier', data, path=f'{output_dir}/XGB{label}')
    print(f'tested {len(combine_params('XGBClassifier'))} in {Util.format_float(Util.measure_time(t))}s')


if len(sys.argv) > 1 and sys.argv[1] == "__trees__":
    sys.argv[1] = "consumed"
    print('preparing data for trees tests')
    x = DataManipulator.load('../data/in_csv/bilinear8_x', decompress=True)
    y = DataManipulator.load('../data/in_csv/y')
    y = Util.ravel(y)
    data = Util.train_test_from(x, y)
    conduct_all(data, '../results/bil8', label='bil8')
    print('conducting pca tests for trees')
    for i in range(10, min(data[0].shape) + 1, 5):
        train, test, variance = Util.use_pca(data[0], data[2], i)
        compressed = (train, data[1], test, data[3])
        print(f'dim={i} var={Util.format_float(variance)}')
        conduct_all(compressed, '../results/PCA', f'_pca{i}')

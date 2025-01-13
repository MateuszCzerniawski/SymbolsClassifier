import itertools

import keras
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import Sequential
import tensorflow
import multiprocessing as mp
import Util
from scripts import DataManipulator

pos_epochs = [10, 20, 50, 100, 200]
pos_batches = [16, 32, 64, 128, 256, 512]
pos_unit_counts = [256, 128, 64, 32, 16]
pos_dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
pos_reg_vals = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
pos_learning_rates = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
pos_optimizers = [Adam, SGD, RMSprop]
pos_regularizes = ['no_reg', 'l1', 'l2']
pos_activation_functions = ['relu', 'elu']
pos_loss_funcs = ['binary_crossentropy', 'mse', 'categorical_crossentropy']


def categorise(data, classes=12):
    return tensorflow.keras.utils.to_categorical(data - 1, num_classes=classes)


def build_net(inputs, output, units_counts, function='relu', dropouts=[], regularizer='l1', regularizer_vals=[]):
    regularizer = l2 if regularizer == 'l2' else l1
    if regularizer == 'no_reg':
        regularizer_vals = []
    dropout_index = 0
    regularize_index = 0
    model = Sequential()
    model.add(keras.Input(shape=inputs))
    model.add(Dense(units_counts[0], activation=function, kernel_regularizer=regularizer(regularizer_vals[0]))
              if len(regularizer_vals) > 0 else Dense(units_counts[0], activation=function))
    regularize_index += 1
    if len(dropouts) > 0:
        model.add(Dropout(dropouts[0]))
        dropout_index += 1
    for units in units_counts[1:]:
        if regularize_index < len(regularizer_vals):
            model.add(Dense(units, activation=function,
                            kernel_regularizer=regularizer(regularizer_vals[regularize_index])))
            regularize_index += 1
        else:
            model.add(Dense(units, activation=function))
        if dropout_index < len(dropouts):
            model.add(Dropout(dropouts[dropout_index]))
            dropout_index += 1
    model.add(Dense(output, activation='softmax',
                    kernel_regularizer=regularizer(regularizer_vals[regularize_index]))
              if regularize_index < len(regularizer_vals) else Dense(output, activation='softmax'))
    return model


def fit_evaluate(input):
    start = Util.measure_time()
    compiled_model, x_train, y_train, x_test, y_test, epochs, batch_size = input
    compiled_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    loss, accuracy, mae = compiled_model.evaluate(x_test, y_test)
    return compiled_model, loss, accuracy, mae, Util.measure_time(start)


def combine_nets(regularizer_vals=None, dropouts=None):
    def descending_arrays(array):
        def is_descending(arr):
            return all(arr[i] > arr[i + 1] for i in range(len(arr) - 1))

        descending = []
        for i in range(len(array)):
            descending.extend([list(arr) for arr in itertools.combinations_with_replacement(array, i + 1)])
        descending = [arr for arr in descending if is_descending(arr)]
        return descending

    tmp = []
    layers = descending_arrays(pos_unit_counts)
    if regularizer_vals == 'various':
        for units in layers:
            tmp.extend([[units, list(reg_vals)] for reg_vals in itertools.combinations(pos_reg_vals, len(units))])
    elif regularizer_vals == 'uniform':
        for units in layers:
            reg_vals = []
            for val in pos_reg_vals:
                reg_vals.append([val for i in range(len(units))])
            tmp.extend([units, vals] for vals in reg_vals)
    else:
        tmp.extend([[units, []] for units in layers])
    nets = []
    for config in tmp:
        drops = []
        if dropouts == 'various':
            drops = [drop for drop in itertools.combinations(pos_dropouts, len(config[0]))]
        elif dropouts == 'uniform':
            for d in pos_dropouts:
                drops.append([d for i in range(len(config[0]))])
        for drop in drops:
            nets.append({'units': config[0], 'reg_vals': config[1], 'dropouts': drop})
        if len(drops) == 0:
            nets.append({'units': config[0], 'reg_vals': config[1], 'dropouts': []})
    return nets


def combine_tests():
    nets = []
    for n in combine_nets(regularizer_vals='uniform', dropouts='uniform'):
        n['reg'] = 'l1'
        nets.append(n)
    for n in combine_nets(regularizer_vals='uniform', dropouts='uniform'):
        n['reg'] = 'l2'
        nets.append(n)
    for n in combine_nets():
        n['reg'] = 'no_reg'
        nets.append(n)
    params = list(
        itertools.product(pos_epochs, pos_batches, pos_learning_rates, pos_optimizers, pos_activation_functions,
                          pos_loss_funcs))
    nets = list(itertools.product(nets, params))
    tmp = []
    for n, p in nets:
        n['epochs'] = p[0]
        n['batch'] = p[1]
        n['learning_rate'] = p[2]
        n['optimiser'] = p[3]
        n['activation_func'] = p[4]
        n['loss_func'] = p[5]
        tmp.append(n)
    nets = tmp
    return nets


def use_net(input):
    data, in_out, params = input
    model = build_net(in_out[0], in_out[1], params['units'], function=params['activation_func'],
                      regularizer=params['reg'], regularizer_vals=params['reg_vals'])
    model.compile(optimizer=params['optimiser'](learning_rate=params['learning_rate']),
                  loss=params['loss_func'], metrics=['accuracy', 'mae'])
    t = Util.measure_time()
    model.fit(data[0], data[1], epochs=params['epochs'], batch_size=params['batch'], verbose=0)
    loss, accuracy, mae = model.evaluate(data[2], data[3])
    output = []
    for key in params.keys():
        val = params[key]
        if not isinstance(val, list):
            output.append(val)
        elif key == 'units':
            for i in val:
                output.append(i)
        else:
            output.append(val[0] if len(val) > 0 else 0)
    output.extend([accuracy, loss, mae, Util.measure_time(t)])
    print('done')
    return tuple(output)


def conduct_all(data, path=None):
    in_out = [(len(x.iloc[0]),), 12]
    tests = [(data, in_out, params) for params in combine_tests()]
    with mp.Pool(processes=20) as pool:
        results = [i for i in pool.map(use_net, tests)]
        results = pd.DataFrame(results)
        if path is not None:
            results.to_csv(path)
        return results

if __name__ == "__main__":
    # tests = combine_tests()
    # print(tests[0].keys())
    # print(len(tests))
    x = DataManipulator.load('../data/in_csv/bilinear4_x', decompress=True)
    y = DataManipulator.load('../data/in_csv/bilinear4_y')
    y = y.values.ravel() if isinstance(y, pd.DataFrame) else np.array(y).ravel()
    y = categorise(y)
    data = Util.train_test_from(x, y)
    # in_out = [(len(x.iloc[0]),), 12]
    # params = {
    #     'units': [256, 128, 64],
    #     'activation_func': 'elu',
    #     'reg': 'no_reg',
    #     'reg_vals': [],
    #     'optimiser': RMSprop,
    #     'learning_rate': 2e-4,
    #     'loss_func': 'binary_crossentropy',
    #     'epochs': 100,
    #     'batch': 128
    # }
    results = conduct_all(data, path='../results/nets')

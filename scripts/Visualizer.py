import math
import os
import re
import statistics
import warnings

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts import DataManipulator, Util, Trees

warnings.filterwarnings("ignore")

colors = ['blue', 'red', 'green', 'orange', 'purple', 'lime', 'black']
missing_dt_values = [('max_features', 'None'), ('min_samples_split', 2), ('max_depth', 'None'),
                     ('criterion', '\'gini\'')]
missing_et_values = [('n_estimators', 100), ('max_features', 'sqrt'), ('min_samples_split', 2), ('max_depth', 'None'),
                     ('criterion', '\'gini\'')]
missing_rf_values = missing_et_values


def visualise_symbols(parsed_images_dir, symbols_dir, output_path):
    blended = dict()
    for name in os.listdir(parsed_images_dir):
        arr = DataManipulator.img_to_array(cv2.imread(f'{parsed_images_dir}/{name}'))
        for s_name in os.listdir(symbols_dir):
            s = re.search(r'^(.*)\.[^\.]*$', s_name).group(1)
            n = re.search(r'_(.*)\.[^\.]*$', name).group(1)
            n = re.search(r'^(.*?)(?=\d+$)', n).group(1)
            if s == n:
                if s in blended:
                    blended[s] = np.add(blended[s], arr)
                else:
                    blended[s] = arr.copy()
    for key in blended:
        img = []
        arr = blended[key]
        size = int(math.sqrt(len(arr)))
        index = 0
        max_val = max(arr)
        for i in range(size):
            img.append([])
            for j in range(size):
                val = 255 - int(255 * (arr[index] / max_val))
                img[i].append([val, val, val])
                index += 1
        symbol = cv2.imread(f'{symbols_dir}/{key}.png')
        height, width, _ = symbol.shape
        img = cv2.resize(np.array(img, dtype=np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
        symbol = cv2.copyMakeBorder(symbol, 8, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.copyMakeBorder(img, 4, 8, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.vconcat([symbol, img])
        blended[key] = img
    final_img = cv2.hconcat([blended[key] for key in blended])
    text = 'ideal vs blended symbols'
    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2, 3)[0]
    final_img = cv2.copyMakeBorder(final_img, 2 * 40 + text_height, 40, 40, 40, cv2.BORDER_CONSTANT,
                                   value=[255, 255, 255])
    cv2.putText(final_img, text, ((final_img.shape[1] - text_width) // 2, text_height + 40), cv2.FONT_HERSHEY_COMPLEX,
                2, [0, 0, 0], 3, lineType=cv2.LINE_AA)
    cv2.imwrite(output_path, final_img)


def visualise_pca_variance(input_dir, output_path):
    y = DataManipulator.load(f'{input_dir}/y')
    all = dict()
    for name in os.listdir(input_dir):
        if 'x' in name:
            dims, vars = [], []
            print(name[:-2])
            x = DataManipulator.load(f'{input_dir}/{name}', decompress=True)
            x_train, y_train, x_test, y_test = Util.train_test_from(x, y)
            for i in range(1, 100, 10):
                train, test, variance = Util.use_pca(x_train, x_test, i)
                dims.append(i)
                vars.append(variance)
            all[name[:-2]] = (dims, vars)
    plt.close()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].set_title('original')
    axes[1].set_title('bilinear')
    axes[2].set_title('nearest')
    counters = [0, 0, 0]
    for name in all.keys():
        tmp = all[name]
        index = 1 if 'bilinear' in name else (2 if 'nearest' in name else 0)
        color = colors[counters[index]] if counters[index] < len(colors) else 'blue'
        axes[index].plot(tmp[0], tmp[1], label=name, color=color)
        counters[index] += 1
        axes[index].set_label(name)
        print(name)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    Util.save_plot(output_path)


def visualise_trees_results(input_path, output_path, missing=None):
    data = pd.read_csv(input_path)
    param_dict = dict()
    for index, row in data.iterrows():
        params = re.search(r'\(([^()]*|\([^()]*\))*\)', row['model']).group(0)[1:-1]
        params = params.replace(' ', '')
        if missing is not None:
            for m in missing:
                if m[0] not in params:
                    params = f'{m[0]}={m[1]},' + params
        params = [i.replace('\n', '') for i in params.split(',')]
        for p in params:
            try:
                param, val = p.split('=')
            except ValueError:
                continue
            if param not in param_dict:
                param_dict[param] = dict()
            elif val not in param_dict[param]:
                param_dict[param][val] = []
            else:
                param_dict[param][val].append((row['accuracy'], row['time']))
    plt.close()
    valid_keys = [p for p in param_dict.keys() if len(param_dict[p]) > 1]
    fig, axes = plt.subplots(1, len(valid_keys), figsize=(5 * len(valid_keys), 5))
    index = -1
    for param in valid_keys:
        index += 1
        tmp = 0
        axes[index].set_title(str(param))
        for val in param_dict[param]:
            accuracies = [i[0] for i in param_dict[param][val]]
            accuracies = pd.value_counts(pd.cut(accuracies, bins=10), sort=False)
            color = colors[tmp] if tmp < len(colors) else 'black'
            accuracies.plot(kind='bar', ax=axes[index], position=tmp, width=0.15, color=color, label=str(val))
            tmp += 1
        axes[index].legend()
    plt.tight_layout()
    Util.save_plot(output_path)


def visualise_nets_results(input_path, output_path):
    data = pd.read_csv(input_path)
    param_dict = {'layer2': dict(), 'layer3': dict(), 'reg_val': dict(), 'reg': dict(),
                  'epochs': dict(), 'batch': dict(), 'learning_rate': dict(), 'optimiser': dict()}
    for index, row in data.iterrows():
        for p in param_dict.keys():
            if row[p] not in param_dict[p]:
                param_dict[p][row[p]] = []
            else:
                param_dict[p][row[p]].append((row['accuracy'], row['time']))
    plt.close()
    fig, axes = plt.subplots(2, len(param_dict.keys()) // 2, figsize=(5 * len(param_dict.keys()), 3 * 5))
    index = -1
    for param in param_dict:
        index += 1
        tmp = 0
        axes[index // 4, index % 4].set_title(str(param))
        for val in param_dict[param]:
            accuracies = [i[0] for i in param_dict[param][val]]
            accuracies = pd.value_counts(pd.cut(accuracies, bins=10), sort=False)
            color = colors[tmp] if tmp < len(colors) else 'black'
            accuracies.plot(kind='bar', ax=axes[index // 4, index % 4], position=tmp, width=0.1, color=color,
                            label=str(val))
            tmp += 1
        axes[index // 4, index % 4].legend()
    plt.tight_layout()
    Util.save_plot(output_path)


def visualise_accuracy_to_time(input_dir, output_path, min_accuracy=None, max_time=None):
    models = {'DecisionTree': [], 'ExtraTree': [], 'RandomForest': [], 'XGB': [], 'net': []}
    for name in os.listdir(input_dir):
        for m in models:
            if m in name:
                for index, row in pd.read_csv(f'{input_dir}/{name}').iterrows():
                    models[m].append((row['accuracy'], row['time']))
    plt.close()
    plt.figure(figsize=(10, 10))
    index = 0
    for name, points in models.items():
        if min_accuracy is not None:
            points = [p for p in points if p[0] >= min_accuracy]
        if max_time is not None:
            points = [p for p in points if p[1] <= max_time]
        accuracies, times = [i[0] for i in points], [i[1] for i in points]
        color = colors[index] if index < len(colors) else 'black'
        index += 1
        plt.scatter(times, accuracies, label=name, color=color, alpha=0.2)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.title('model accuracy to training time')
    Util.save_plot(output_path)


def visualise_pca(input_dir, output_path, top=10):
    models = dict()
    for name in os.listdir(input_dir):
        model = re.search(r'^(.*)_', name).group(1)
        dim = re.search(r'(\d+)$', name).group(1)
        if model not in models:
            models[model] = dict()
        data = pd.read_csv(f'{input_dir}/{name}')
        data = data.sort_values(by='accuracy', ascending=False)
        models[model][dim] = (list(data['accuracy'][:top + 1]), list(data['time'][:top + 1]))
    tmp = dict()
    for model, dims in models.items():
        tmp[model] = [(int(dim), vals[0][0], statistics.mean(vals[0]), vals[1][0], statistics.mean(vals[1]))
                      for dim, vals in dims.items()]
        tmp[model].sort(key=lambda x: x[0])
    models = tmp
    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for model, info in models.items():
        dims = [i[0] for i in info]
        axes[0, 0].plot(dims, [i[1] for i in info], label=f'{model}')
        axes[0, 1].plot(dims, [i[2] for i in info], label=f'{model}')
        axes[1, 0].plot(dims, [i[3] for i in info], label=f'{model}')
        axes[1, 1].plot(dims, [i[4] for i in info], label=f'{model}')
    axes[0, 0].set_title('best accuracy')
    axes[0, 1].set_title('average accuracy')
    axes[1, 0].set_title('best time')
    axes[1, 1].set_title('average time')
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    plt.tight_layout()
    Util.save_plot(output_path)


def visualise_params_popularity(input_dir, output_dir, top=20):
    models = dict()
    all_dims = set()
    for name in os.listdir(input_dir):
        model = re.search(r'^(.*)_', name).group(1)
        dim = int(re.search(r'(\d+)$', name).group(1))
        if model not in models:
            models[model] = []
        insert_index = 0
        if len(models[model]) != 0:
            for i, n in enumerate(models[model]):
                insert_index = i + 1 if dim > int(re.search(r'(\d+)$', n).group(1)) else insert_index
        models[model].insert(insert_index, name)
        all_dims.add(dim)
    models = {k: {'files': v} for k, v in models.items()}
    models['NET'] = {'files': models['NET']['files'], 'layer2': dict(), 'layer3': dict(), 'reg_val': dict(),
                     'reg': dict(), 'epochs': dict(), 'batch': dict(), 'learning_rate': dict(), 'optimiser': dict()}
    all_dims = list(all_dims)
    for model in models.keys():
        names = models[model]['files']
        missing = {'DT': missing_dt_values, 'ET': missing_et_values, 'RF': missing_rf_values}
        missing = missing[model] if model in missing else None
        insert_index = -1
        for filepath in [f'{input_dir}/{filename}' for filename in names]:
            data = pd.read_csv(filepath).sort_values(by='accuracy', ascending=False)[:top]
            insert_index += 1
            if model in Trees.model_short:
                for index, row in data.iterrows():
                    params = re.search(r'\(([^()]*|\([^()]*\))*\)', row['model']).group(0)[1:-1]
                    params = params.replace(' ', '')
                    params = [i.replace('\n', '') for i in params.split(',')]
                    for p in params:
                        try:
                            param, val = p.split('=')
                            param=param.replace('\r','')
                        except ValueError:
                            continue
                        if param not in models[model]:
                            models[model][param] = dict()
                        if val not in models[model][param]:
                            models[model][param][val] = [0 for i in range(len(all_dims))]
                        models[model][param][val][insert_index] += 1
            else:
                for index, row in data.iterrows():
                    for param in models[model].keys():
                        if type(models[model][param]) != dict:
                            continue
                        if row[param] not in models[model][param]:
                            models[model][param][row[param]] = [0 for i in range(len(all_dims))]
                        models[model][param][row[param]][insert_index] += 1
        if missing is not None:
            for param, missing_value in missing:
                tmp = [top for i in range(len(all_dims))]
                for val, counts in models[model][param].items():
                    for i, v in enumerate(counts):
                        tmp[i] -= v
                models[model][param][missing_value] = tmp
    for model in models.keys():
        to_delete=['files']
        for key in models[model].keys():
            if len(models[model][key])<=1:
                to_delete.append(key)
        for key in to_delete:
            del models[model][key]
    for model, dims in models.items():
        plt.close()
        fig, axes = plt.subplots(1, len(dims.keys()), figsize=(5 * len(dims.keys()), 5))
        index = -1
        for param, vals in dims.items():
            index += 1
            for label, values in vals.items():
                axes[index].plot(all_dims, values, label=label)
            axes[index].legend()
            axes[index].set_title(param)
        plt.tight_layout()
        Util.save_plot(f'{output_dir}/hiperparams_{model}')


visualise_params_popularity('../results/PCA', '../graphs/PCA')

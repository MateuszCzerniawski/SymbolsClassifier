import math
import os
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt

from scripts import DataManipulator, Util


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
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'lime', 'black']
    axes[0].set_title('original')
    axes[1].set_title('bilinear')
    axes[2].set_title('nearest')
    counters=[0,0,0]
    for name in all.keys():
        tmp = all[name]
        index = 1 if 'bilinear' in name else (2 if 'nearest' in name else 0)
        color=colors[counters[index]] if counters[index]<len(colors) else 'blue'
        axes[index].plot(tmp[0], tmp[1], label=name,color=color)
        counters[index]+=1
        axes[index].set_label(name)
        print(name)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    Util.save_plot(output_path)

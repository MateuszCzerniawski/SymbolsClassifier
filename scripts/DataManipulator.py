import math
import os
import re

import cv2
import numpy as np
import pandas as pd

left, up, right, down = 443, 642, 5475, 5678
size, edge = 400, 20
symbols_names = ['ankha', 'loop', 'p', 'grass',
                 'rect', 'vawe', 'plate', 'eye',
                 'circ_center', 'circ_stripped', 'circ_x', 'scarab']


def parse_img(input_path, output_dir, names=symbols_names, output_label=None, shift=True):
    images, labels = [], []
    image = cv2.imread(input_path)
    all_x, all_y = range(left, right + 1, size + edge + 1), range(up, down + 1, size + edge + 1)
    coords = [(x, y) for x in all_x for y in all_y]
    for x, y in coords:
        img = darken(image[y:y + size, x:x + size])
        if shift:
            img = shift_to_center(img)
        if names is not None and output_label is not None:
            symbol_name = names[all_x.index(x)]
            cv2.imwrite(f'{output_dir}/{output_label}_{symbol_name}{all_y.index(y)}.png', img)
            labels.append(symbol_name)
        images.append(img)
    if names is not None and output_label is not None:
        return images, labels
    else:
        return images


def darken(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
    return image


def minimise(image, scale):
    scale = 1 / scale if scale < 1 else scale
    height, width, _ = image.shape
    width, height = int(width / scale), int(height / scale)
    nearest = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    bilinear = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return nearest, bilinear


def parse_all(input_dir, output_dir, names=symbols_names):
    images, labels = [], []
    for name in os.listdir(input_dir):
        img, l = parse_img(f'{input_dir}/{name}', output_dir, names=names, output_label=name)
        images.extend(img)
        labels.extend(l)
    return images, labels


def minimise_all(input_dir, scale, nearest_output_dir=None, bilinear_output_dir=None):
    all_nearest, all_bilinear = [], []
    if nearest_output_dir is not None and not os.path.exists(nearest_output_dir):
        os.mkdir(nearest_output_dir)
    if bilinear_output_dir is not None and not os.path.exists(bilinear_output_dir):
        os.mkdir(bilinear_output_dir)
    for name in os.listdir(input_dir):
        nearest, bilinear = minimise(cv2.imread(f'{input_dir}/{name}'), scale)
        if nearest_output_dir is not None:
            cv2.imwrite(f'{nearest_output_dir}/{name}_nearest.png', darken(nearest))
        if bilinear_output_dir is not None:
            cv2.imwrite(f'{bilinear_output_dir}/{name}_bilinear.png', darken(bilinear))
        all_nearest.append(nearest)
        all_bilinear.append(bilinear)
    return all_nearest, all_bilinear


def img_to_array(image):
    pixels = []
    tmp = [1 if pixel == 0 else 0 for pixel in image.flatten()]
    for i in range(0, len(tmp), 3):
        pixel = [tmp[i] == 0, tmp[i + 1] == 0, tmp[i + 2] == 0]
        pixels.append(int(0 if any(pixel) else 1))
    return pixels


def compress_to_ints(arr, width=None):
    width = int(math.sqrt(len(arr))) if width is None else width
    compressed = []
    arr = [arr[i:i + width] for i in range(0, len(arr), width)]
    for row in arr:
        compressed.append(int(''.join(map(str, row)), 2))
    return compressed


def decompress_from_ints(arr, width=None):
    width = len(arr) if width is None else width
    decompressed = []
    for row in arr:
        decompressed.extend(list(map(int, bin(row)[2:].zfill(width))))
    return decompressed


def center_of_image(image):
    sum_x, sum_y, count = 0, 0, 0
    for y in range(len(image)):
        row = image[y]
        for x in range(len(row)):
            pixel = row[y]
            black = False
            try:
                black = pixel == 0
            except:
                black = any([pixel[0] == 0, pixel[1] == 0, pixel[2] == 0])
            if black:
                sum_y += y
                sum_x += x
                count += 1
    return int(sum_x / count), int(sum_y / count)


def shift_image(image, shift):
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    dsize = (image.shape[1], image.shape[0])
    return cv2.warpAffine(image, matrix, dsize, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def shift_to_center(image):
    center = center_of_image(image)
    return shift_image(image, (200 - center[0], 200 - center[1]))


def dir_to_csv(input_dir, output_x, output_y):
    x, y = [], []
    for name in os.listdir(input_dir):
        arr = compress_to_ints(img_to_array(cv2.imread(f'{input_dir}/{name}')))
        x.append(arr)
        y.append(symbols_names.index(re.search(r"_(.*?)(?=\d)", name).group(1)))
    x, y = pd.DataFrame(x), pd.DataFrame(y)
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    x.to_csv(output_x, index=False)
    y.to_csv(output_y, index=False)
    return x, y


def load(path, decompress=False, width=None):
    df = pd.DataFrame(pd.read_csv(path))
    df.reset_index(drop=True, inplace=True)
    if decompress:
        tmp = []
        for i in range(len(df)):
            row = [int(j) for j in df.iloc[i]]
            tmp.append(decompress_from_ints(row, width=width))
        df = pd.DataFrame(tmp)
    return df


def process(input_dir, parsed_dir, csv_dir, minimisation_scales):
    parse_all(input_dir, parsed_dir)
    x, y = dir_to_csv(parsed_dir, f'{csv_dir}/original_x', f'{csv_dir}/original_y')
    for scale in minimisation_scales:
        nearest, bilinear = f'{parsed_dir}/../nearest{scale}', f'{parsed_dir}/../bilinear{scale}'
        minimise_all(parsed_dir, scale, nearest_output_dir=nearest, bilinear_output_dir=bilinear)
        dir_to_csv(nearest, f'{csv_dir}/nearest{scale}_x', f'{csv_dir}/nearest{scale}_y')
        dir_to_csv(bilinear, f'{csv_dir}/bilinear{scale}_x', f'{csv_dir}/bilinear{scale}_y')
    for name in os.listdir(csv_dir):
        if '_y' in name and ('bilinear' in name or 'nearest' in name):
            os.remove(f'{csv_dir}/{name}')
    os.remove(f'{csv_dir}/y')
    os.rename(f'{csv_dir}/original_y', f'{csv_dir}/y')
    return x, y

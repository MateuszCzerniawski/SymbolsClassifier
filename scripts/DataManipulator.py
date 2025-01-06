import os

import cv2
import pandas as pd

left, up, right, down = 443, 642, 5475, 5678
size, edge = 400, 20


def parse_img(input_path, output_dir, name=''):
    output = []
    image = cv2.imread(input_path)
    coords = [(x, y) for x in range(left, right + 1, size + edge + 1) for y in range(up, down + 1, size + edge + 1)]
    index = 1
    for x, y in coords:
        cropped = image[y:y + size, x:x + size]
        cropped = darken(cropped)
        cv2.imwrite(f'{output_dir}/{name}_{index}.png', cropped)
        index += 1
        output.append(cropped)
    return output


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


def parse_all(input_dir, output_dir):
    for name in os.listdir(input_dir):
        parse_img(f'{input_dir}/{name}', output_dir, name=name)


def minimise_all(input_dir, scale, nearest_output_dir=None, bilinear_output_dir=None):
    for name in os.listdir(input_dir):
        nearest, bilinear = minimise(cv2.imread(f'{input_dir}/{name}'), scale)
        if nearest_output_dir is not None:
            cv2.imwrite(f'{nearest_output_dir}/{name}_nearest.png', darken(nearest))
        if bilinear_output_dir is not None:
            cv2.imwrite(f'{bilinear_output_dir}/{name}_bilinear.png', darken(bilinear))


def img_to_array(image):
    return [1 if pixel == 0 else 0 for pixel in image.flatten()]


def img_dir_to_csv(input_dir, output_path):
    rows = []
    for name in os.listdir(input_dir):
        row = img_to_array(cv2.imread(f'{input_dir}/{name}'))
        rows.append(row)
        print(row)
    print('dataframe')
    df = pd.DataFrame(rows)
    print('saving')
    df.to_csv(output_path)
    print('done')
    return df


img_dir_to_csv('../data/images/parsed', '../data/in_csv/x_train')

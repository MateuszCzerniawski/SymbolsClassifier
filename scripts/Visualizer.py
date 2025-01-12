import math
import os
import re

import cv2
import numpy as np

from scripts import DataManipulator


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




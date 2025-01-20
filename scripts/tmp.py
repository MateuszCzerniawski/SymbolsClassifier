import os
import re
import statistics

import matplotlib.pyplot as plt
import pandas as pd

from scripts import DataManipulator, Util
import Trees
import Visualizer
import multiprocessing as mp

# accuracies = dict()
# for name in os.listdir('../results/PCA'):
#     data = pd.read_csv(f'../results/PCA/{name}')
#     data = data.sort_values(by='accuracy')
#     accuracies[int(name[len('DT_pca'):])] = list(data['accuracy'][:10])
# best = [(dim, vals[0]) for dim, vals in accuracies.items()]
# avg = [(dim, statistics.mean(vals)) for dim, vals in accuracies.items()]
# best.sort(key=lambda x: x[0])
# avg.sort(key=lambda x: x[0])
# for i in best:
#     print(i)
# plt.close()
# plt.plot([i[0] for i in best], [i[1] for i in best], label='best')
# plt.plot([i[0] for i in avg], [i[1] for i in avg], label='average')
# plt.legend()
# plt.show()
if __name__ == "__main__":
    x = DataManipulator.load('../data/in_csv/bilinear8_x', decompress=True)
    y = DataManipulator.load('../data/in_csv/y')
    y = Util.ravel(y)
    data = Util.train_test_from(x, y)
    Trees.conduct_tests('XGBClassifier', data, path='../results/bil8/XGB_res')
    Visualizer.visualise_trees_results('../results/bil8/XGB_res', '../graphs/XGBacc')

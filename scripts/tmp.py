import os
import statistics

import matplotlib.pyplot as plt
import pandas as pd

from scripts import DataManipulator, Util
import Trees
import multiprocessing as mp

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    # x = DataManipulator.load('../data/in_csv/bilinear8_x', decompress=True)
    # y = DataManipulator.load('../data/in_csv/y')
    # y = Util.ravel(y)
    # data = Util.train_test_from(x, y)
    # for i in range(10, min(data[0].shape) + 1, 5):
    #     train, test, variance = Util.use_pca(data[0], data[2], i)
    #     compressed = (train, data[1], test, data[3])
    #     print(f'dim={i} var={Util.format_float(variance)}')
    #     Trees.conduct_tests('DecisionTreeClassifier', compressed, path=f'../results/PCA/DT_pca{i}')
    accuracies = dict()
    for name in os.listdir('../results/PCA_DT'):
        data = pd.read_csv(f'../results/PCA_DT/{name}')
        data=data.sort_values(by='accuracy')
        accuracies[int(name[len('DT_pca'):])] = list(data['accuracy'][:10])
    best = [(dim, vals[0]) for dim, vals in accuracies.items()]
    avg = [(dim, statistics.mean(vals)) for dim, vals in accuracies.items()]
    best.sort(key=lambda x: x[0])
    avg.sort(key=lambda x: x[0])
    for i in best:
        print(i)
    plt.close()
    plt.plot([i[0] for i in best], [i[1] for i in best], label='best')
    plt.plot([i[0] for i in avg], [i[1] for i in avg], label='average')
    plt.legend()
    plt.show()

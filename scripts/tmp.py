from scripts import DataManipulator, Util
import Trees
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn')
    x = DataManipulator.load('../data/in_csv/bilinear8_x', decompress=True)
    y = DataManipulator.load('../data/in_csv/y')
    y = Util.ravel(y)
    data = Util.train_test_from(x, y)
    for i in range(10, min(data[0].shape) + 1, 5):
        train, test, variance = Util.use_pca(data[0], data[2], i)
        compressed = (train, data[1], test, data[3])
        print(f'dim={i} var={Util.format_float(variance)}')
        Trees.conduct_tests('DecisionTreeClassifier', compressed, path=f'../results/PCA/DT_pca{i}')

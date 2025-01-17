import subprocess

from scripts import DataManipulator, Visualizer

if __name__ == "__main__":
    print('processing data')
    DataManipulator.process('../data/images/raw', '../data/images/parsed', '../data/in_csv', [2, 4, 8])
    print('visualising data')
    Visualizer.visualise_symbols('../data/images/parsed', '../data/images/symbols', f'../graphs/blended.png')
    Visualizer.visualise_pca_variance('../data/in_csv', '../graphs/variancePCA.png')
    print('conducting tests for trees:')
    trees_result = subprocess.run(['../.venv/Scripts/python.exe', 'Trees.py', "__trees__"], text=True)
    print('conducting tests for nets')
    nets_result = subprocess.run(['../.venv/Scripts/python.exe', 'Neural.py', "__nets__"], text=True)

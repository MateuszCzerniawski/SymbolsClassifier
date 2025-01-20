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
    print('visualising results for trees')
    Visualizer.visualise_trees_results('../results/bil8/DecisionTree_res', '../graphs/hiperparameters/DTacc',
                                       missing=Visualizer.missing_dt_values)
    Visualizer.visualise_trees_results('../results/bil8/ExtraTrees_res', '../graphs/hiperparameters/ETacc',
                                       missing=Visualizer.missing_et_values)
    Visualizer.visualise_trees_results('../results/bil8/RandomForest_res', '../graphs/hiperparameters/RFacc',
                                       missing=Visualizer.missing_rf_values)
    Visualizer.visualise_trees_results('../results/bil8/XGB_res', '../graphs/hiperparameters/XGBacc')
    print('conducting tests for nets')
    nets_result = subprocess.run(['../.venv/Scripts/python.exe', 'Neural.py', "__nets__"], text=True)
    print('visualising results for nets')
    Visualizer.visualise_nets_results('../results/bil8/nets', '../graphs/nets_acc')
    print('visualising trees and nets comparisons')
    Visualizer.visualise_accuracy_to_time('../results/bil8', '../graphs/accuracy to time')
    Visualizer.visualise_accuracy_to_time('../results/bil8', '../graphs/accuracy to time (min acc)', min_accuracy=0.8)


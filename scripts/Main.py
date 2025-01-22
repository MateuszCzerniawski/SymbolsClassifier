import subprocess

from scripts import DataManipulator, Visualizer

if __name__ == "__main__":
    print('processing data')
    DataManipulator.process('../data/images/raw', '../data/images/parsed', '../data/in_csv', list(range(2, 21)))
    print('visualising data')
    Visualizer.visualise_symbols('../data/images/parsed', '../data/images/symbols', f'../graphs/blended.png')
    Visualizer.visualise_pca_variance('../data/in_csv', '../graphs/PCA/variancePCA.png')
    print('conducting tests for trees:')
    trees_result = subprocess.run(['../.venv/Scripts/python.exe', 'Trees.py', "__trees__"], text=True)
    print('visualising results for trees')
    Visualizer.visualise_trees_results('../results/bil/DT_bil8', '../graphs/hiperparameters/DTacc',
                                       missing=Visualizer.missing_dt_values)
    Visualizer.visualise_trees_results('../results/bil/ET_bil8', '../graphs/hiperparameters/ETacc',
                                       missing=Visualizer.missing_et_values)
    Visualizer.visualise_trees_results('../results/bil/RF_bil8', '../graphs/hiperparameters/RFacc',
                                       missing=Visualizer.missing_rf_values)
    Visualizer.visualise_trees_results('../results/bil/XGB_bil8', '../graphs/hiperparameters/XGBacc')
    print('conducting tests for NET_bil8')
    nets_result = subprocess.run(['../.venv/Scripts/python.exe', 'Neural.py', "__nets__"], text=True)
    print('visualising results for NET_bil8')
    Visualizer.visualise_nets_results('../results/bil/NET_bil8', '../graphs/nets_acc')
    print('visualising trees and NET_bil8 comparisons')
    Visualizer.visualise_accuracy_to_time('../results/bil8', '../graphs/accuracy to time')
    Visualizer.visualise_accuracy_to_time('../results/bil8', '../graphs/accuracy to time (min acc)', min_accuracy=0.8)
    print('visualising best accuracy for bilinear nad PCA tests')
    Visualizer.visualise_time_and_accuracy_over_dims('../results/PCA', '../graphs/PCA/PCAacc')
    Visualizer.visualise_time_and_accuracy_over_dims('../results/bil', '../graphs/bil/bil acc')
    print('visualising hiperparameters changes')
    Visualizer.visualise_params_popularity('../results/bil', '../graphs/bil')
    Visualizer.visualise_params_popularity('../results/PCA', '../graphs/PCA')

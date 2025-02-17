import subprocess

from scripts import DataManipulator, Visualizer, Util

if __name__ == "__main__":
    # print('processing data')
    # DataManipulator.process('../data/images/raw', '../data/images/parsed', '../data/in_csv',
    #                         minimisation_scales=list(range(2, 21)), filenames=['set1.png'])
    # print('visualising data')
    # Visualizer.visualise_symbols('../data/images/parsed', '../data/images/symbols', f'../graphs/blended.png')
    # Visualizer.visualise_pca_variance('../data/in_csv', '../graphs/PCA/variancePCA.png')
    # print('conducting tests for trees:')
    # trees_result = subprocess.run(['../.venv/Scripts/python.exe', 'Trees.py', "__trees__"], text=True)
    # print('visualising results for trees')
    # Visualizer.visualise_trees_results('../results/bil/DT_bil8', '../graphs/hiperparameters/DTacc',
    #                                    missing=Visualizer.missing_dt_values)
    # Visualizer.visualise_trees_results('../results/bil/ET_bil8', '../graphs/hiperparameters/ETacc',
    #                                    missing=Visualizer.missing_et_values)
    # Visualizer.visualise_trees_results('../results/bil/RF_bil8', '../graphs/hiperparameters/RFacc',
    #                                    missing=Visualizer.missing_rf_values)
    # Visualizer.visualise_trees_results('../results/bil/XGB_bil8', '../graphs/hiperparameters/XGBacc')
    # print('conducting tests for NET_bil8')
    # nets_result = subprocess.run(['../.venv/Scripts/python.exe', 'Neural.py', "__nets__"], text=True)
    # print('visualising results for NET_bil8')
    # Visualizer.visualise_nets_results('../results/bil/NET_bil8', '../graphs/nets_acc')
    # print('visualising trees and NET_bil8 comparisons')
    Visualizer.visualise_accuracy_to_time('../results/bil', '../graphs/accuracy to time', allowed_numbers=[8])
    # Visualizer.visualise_accuracy_to_time('../results/bil', '../graphs/accuracy to time (min acc)',
    #                                       allowed_numbers=[8], min_accuracy=0.85)
    # print('visualising best accuracy for bilinear and PCA tests')
    # Visualizer.visualise_time_and_accuracy_over_dims('../results/PCA', '../graphs/PCA/PCAacc')
    # Visualizer.visualise_time_and_accuracy_over_dims('../results/bil', '../graphs/bil/bil acc')
    # print('visualising hiperparameters changes')
    # Visualizer.visualise_params_popularity('../results/bil', '../graphs/bil')
    # Visualizer.visualise_params_popularity('../results/PCA', '../graphs/PCA')
    # print('finding best nets')
    Util.collect_best_nets('../results/PCA', '../results/best_candidates', min_accuracy=0.97)
    # print('processing full dataset')
    # DataManipulator.process('../data/images/raw', '../data/images/parsed', '../data/in_csv')
    # print('testing best nets')
    # nets_result = subprocess.run(['../.venv/Scripts/python.exe', 'Neural.py', "__best-nets__"], text=True)
    # print('visualising final results')
    Visualizer.visualise_time_and_accuracy_over_dims('../results/best', '../graphs/best nets acc')
    Visualizer.visualise_params_popularity('../results/best', '../graphs')
    Visualizer.visualise_accuracy_to_time('../results/best', '../graphs/best accuracy to time')
    Util.collect_best_nets('../results/best', '../results/best_nets', min_accuracy=0.93)



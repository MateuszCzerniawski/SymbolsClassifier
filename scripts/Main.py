import DataManipulator
import Visualizer

print('processing data')
DataManipulator.process('../data/images/raw', '../data/images/parsed', '../data/in_csv', [2, 4])
print('visualising data')
Visualizer.visualise_symbols('../data/images/parsed', '../data/images/symbols', f'../graphs/blended.png')

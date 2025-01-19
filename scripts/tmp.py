import Util
import DataManipulator
import Trees

params=Trees.combine_params('XGBClassifier')
x = DataManipulator.load('../data/in_csv/bilinear2_x', decompress=True)
y = DataManipulator.load('../data/in_csv/y')
y = Util.ravel(y)
data = Util.train_test_from(x, y)
print('start')
t = Util.measure_time()
Trees.use_xgboost(data)
t = Util.measure_time(t)
print(f'{t}s * {len(params)} / 14 = {Util.format_float(t*len(params)/14)}s')

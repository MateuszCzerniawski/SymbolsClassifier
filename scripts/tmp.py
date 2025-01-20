import Util
import DataManipulator
import Trees
import Neural

all = Neural.combine_tests()
e, b = set(), set()
for i in all:
    e.add(i['epochs'])
    b.add(i['batch'])
print(e)
print(b)

print(len(all)*20/60/14/60)

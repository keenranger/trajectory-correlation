import numpy as np
import pandas as pd
from module.beacon_parser import beacon_parser
from matplotlib import pyplot as plt


data_1233 = beacon_parser(pd.read_csv("./data/210219/1233o.csv", header=None))
data_1250 = beacon_parser(pd.read_csv("./data/210219/1233o.csv", header=None))
data_1251_1 = beacon_parser(pd.read_csv("./data/210219/1233o.csv", header=None))
data_1251_2 = beacon_parser(pd.read_csv("./data/210219/1233o.csv", header=None))


plt.figure()
plt.plot(data_1233[:,0], data_1233[:,23])
plt.show()
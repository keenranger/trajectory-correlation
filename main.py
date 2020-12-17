import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

color_list = ["r", "g", "b"]
scen_list = range(1, 10)
cs_list = []
hk_list = []
for scen in scen_list:
    cs_list.append(
        pd.read_csv("data/201215/cs/PDR_scen" + str(scen) + ".txt", header=None)
    )
    hk_list.append(
        pd.read_csv("data/201215/hk/PDR_scen" + str(scen) + ".txt", header=None)
    )
plt.figure()
for test_idx, color in enumerate(color_list):
    plt.plot(cs_list[test_idx][12], color+"--", label="test " + str(test_idx + 1))
    plt.plot(hk_list[test_idx][12], color, label="test " + str(test_idx + 1))
plt.title("beacon 13")
plt.legend()
plt.savefig("result/1to3_beacon13.png")
plt.figure()
for test_idx, color in enumerate(color_list):
    plt.plot(cs_list[test_idx][14], color+"--", label="test " + str(test_idx + 1))
    plt.plot(hk_list[test_idx][14], color, label="test " + str(test_idx + 1))
plt.title("beacon 14")
plt.legend()
plt.savefig("result/1to3_beacon14.png")
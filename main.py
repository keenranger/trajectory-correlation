import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("hello world!")
user_list = ["cs", "hk"]
color_list = ["r", "b"]
scen_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for scen in scen_list:
    plt.figure()
    for user, color in zip(user_list, color_list):
        pattern = pd.read_csv(
            "data/201215/" + user + "/PDR_scen" + str(scen) + ".txt", header=None
        )
        plt.plot(pattern[12], color, label="user")
    plt.title("beacon 13 scenario " + str(scen))
    plt.legend()
    plt.savefig("result/13-" + str(scen) + ".png")

for scen in scen_list:
    plt.figure()
    for user, color in zip(user_list, color_list):
        pattern = pd.read_csv(
            "data/201215/" + user + "/PDR_scen" + str(scen) + ".txt", header=None
        )
        plt.plot(pattern[14], color, label="user")
    plt.title("beacon 14 scenario " + str(scen))
    plt.legend()
    plt.savefig("result/14-" + str(scen) + ".png")
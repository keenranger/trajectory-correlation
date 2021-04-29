import pandas as pd
import numpy as np


def input_parser(data, data_length = 23, beacon_length = 10, total_beacon = 9):
    parsed = np.zeros([np.shape(data)[0], data_length + total_beacon]) - 200
    parsed[:, :data_length] = data[:, :data_length]

    for beacon in np.arange(beacon_length):
        col_idx = data_length+ beacon * 4
        having_beacon = data[data[:, col_idx] != 0]
        for idx in range(np.shape(having_beacon)[0]):
            beacon_id = int(having_beacon[idx, col_idx])
            if beacon_id == 10:
                beacon_id = 5
            beacon_rssi = having_beacon[idx, col_idx + 1]
            parsed[idx, data_length + beacon_id - 1] = beacon_rssi
    return parsed

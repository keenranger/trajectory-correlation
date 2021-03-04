import pandas as pd
import numpy as np


def beacon_parser(df):
    parsed = np.zeros([np.shape(df)[0], 23 + 9]) - 200
    parsed[:, :23] = df.loc[:, 1:23].to_numpy()

    for beacon in np.arange(7):
        idx = 24 + beacon * 4
        if df.dtypes[idx] == "object":
            having_beacon = df[df[idx] != "0"]
            for row in having_beacon.itertuples():
                parsed[row[0], 23 + int(row[idx + 1][-2:]) - 1] = row[idx + 2]
    return parsed

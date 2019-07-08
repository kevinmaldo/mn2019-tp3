#!/usr/bin/env python
import dataset
import airline
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import scores
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class Model:

    def __init__(self, data_range):
        self._coeffs = None
    
    def _build_lstq_matrix(self, df):
        features = []

        features.append(np.ones(df.shape[0]))
        features.append(df.index.astype(np.int64) / 10 ** 9)

        s = 365.0  # observations per annum
        n = int(s // 2)
        for j in [0, 1, 2]:
            factor = 2 * np.pi * j / s
            # days_number = (df.index - df.index[0]) / np.timedelta64(1, 'D')
            days_number = (df['Date'] - pd.Timestamp('1986-01-01')).dt.days
            features.append(np.sin(days_number * factor))
            features.append(np.cos(days_number * factor))

        # features.append(1 / (1 + np.minimum(df['Date'].dayofyear, abs(df['Date'].dayofyear - 365)) ** 2))

        features.append(df["OriginScore"])
        features.append(df["DestScore"])
        features.append(df["CarrierScore"])
        return np.stack(features, axis=1)

    def fit(self, df):
        A = self._build_lstq_matrix(df)
        self._coeffs = np.linalg.lstsq(A, df["Delayed"], rcond=None)[0]

    def predict(self, df):
        return self._build_lstq_matrix(df) @ self._coeffs

def main():
    ALL_DATA_RANGE = list(range(2002, 2009))
    model = Model(ALL_DATA_RANGE)
    training_years_count = -3
    training_years = ALL_DATA_RANGE[:training_years_count]
    test_years = ALL_DATA_RANGE[training_years_count:]
    train_df = dataset.get_pre_processed_data(training_years).reset_index()
    model.fit(train_df)
    print(model._coeffs)
    train_df["Prediction"] = model._build_lstq_matrix(train_df) @ np.array(model._coeffs)
    plt.figure()
    train_grouped = train_df.groupby(["Date"]).mean()
    sns.lineplot(train_grouped.index, train_grouped["Delayed"], color='r')
    sns.lineplot(train_grouped.index, train_grouped["Prediction"], color='b')
    plt.show()

    test_df = dataset.get_pre_processed_data(test_years)
    prediction = model.predict(test_df)
    print("RMSE: {}".format(math.sqrt(mean_squared_error(test_df["Delayed"], prediction))))

    test_df["Prediction"] = prediction
    grouped = test_df.groupby(["Date"]).mean()
    plt.figure()
    sns.lineplot(grouped.index, grouped["Delayed"], data=grouped, color='r')
    sns.lineplot(grouped.index, grouped["Prediction"], data=grouped, color='b')
    plt.show()

if __name__ == "__main__":
    main()

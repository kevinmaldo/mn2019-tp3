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

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class Model:

    def __init__(self, data_range):
        self._coeffs = None
    
    def _build_lstq_matrix(self, df):
        features = []

        features.append(np.ones(df.shape[0]))

        s = 365.0  # observations per annum
        n = int(s // 2)
        days_number = (df['Date'] - df["Date"].min()).dt.days
        features.append(days_number)            
        for j in [0, 1, 2, 3, 4, 5, 6, # some peaks per year
                  10, 11, 12, 13, 14, 15, # around one peak per month
                  21, 22, 23, 24, 25, 26, # around two peaks per month
                  50, 51, 52, 53, 54, 55, # around one peak per week
                  102, 103, 104, 105, 106, 107, # around two peaks per week
                  363, 364, 365, 366, 367 # around one peak per day
                  ]:
            factor = 2 * np.pi * j / s
            features.append(np.sin(days_number * factor))
            features.append(np.cos(days_number * factor))

        features.append(1 / (1 + np.minimum(df['Date'].dt.dayofyear, abs(df['Date'].dt.dayofyear - 365)) ** 2))

        features.append(df["OriginScore"])
        features.append(df["DestScore"])
        features.append(df["CarrierScore"])
        return np.stack(features, axis=1)

    def fit(self, df):
        A = self._build_lstq_matrix(df)
        self._coeffs = np.linalg.lstsq(A, df["Delayed"], rcond=None)[0]

    def predict(self, df):
        return self._build_lstq_matrix(df) @ self._coeffs

def _is_delayed(delayed_times):
    return (delayed_times >= 15).astype(np.float64)

def main():
    ALL_DATA_RANGE = list(range(2002, 2009))
    model = Model(ALL_DATA_RANGE)
    training_years_count = -3
    training_years = ALL_DATA_RANGE[:training_years_count]
    test_years = ALL_DATA_RANGE[training_years_count:]
    train_df = dataset.get_pre_processed_data(training_years).reset_index()
    model.fit(train_df)
    train_df["Prediction"] = model._build_lstq_matrix(train_df) @ np.array(model._coeffs)
    #plt.figure()
    train_grouped = train_df.groupby(["Date"]).mean()
    #sns.lineplot(train_grouped.index, train_grouped["Delayed"], color='r')
    #sns.lineplot(train_grouped.index, train_grouped["Prediction"], color='b')
    #plt.show()

    test_df = dataset.get_pre_processed_data(test_years)
    test_df = test_df[test_df["Date"] < '2008-09-01'] # remove the 2008 recesion
    prediction = model.predict(test_df)
    print("RMSE: {}".format(math.sqrt(mean_squared_error(_is_delayed(test_df["Delayed"]),
                                                         _is_delayed(prediction)))))

    test_df["Prediction"] = prediction
    grouped = test_df.groupby(["Date"]).mean()
    #plt.figure()
    #sns.lineplot(grouped.index, grouped["Delayed"], data=grouped, color='r')
    #sns.lineplot(grouped.index, grouped["Prediction"], data=grouped, color='b')
    #plt.show()

    final_df = pd.concat([train_df, test_df], sort=False)
    final_df["Prediction"] = model.predict(final_df)
    final_grouped = final_df.groupby(["Date"]).mean()
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(final_grouped.index, final_grouped["Delayed"], color='r', label="data", ax=ax)
    sns.lineplot(final_grouped.index, final_grouped["Prediction"], color='b', label="prediction", ax=ax)
    ax.axvline(x=test_df["Date"].min(), linestyle="--")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

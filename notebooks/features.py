import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import math
from scipy import signal

import sentiment # TODO: cambiar el nombre

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


sns.set()

def _load_original_csv():
    years = range(2003, 2005)
    columns = ["Year", "Month", "DayofMonth", "ArrDelay", "DepDelay"]
    dataframes = (pd.read_csv("../data/{}.csv".format(year), usecols=columns) for year in years)
    return pd.concat(dataframes, ignore_index=False)

def _set_date_index(df):
    df["Date"] = df["Year"].map("{:04d}".format) + "-" + \
                 df["Month"].map("{:02d}".format) + "-" + \
                 df["DayofMonth"].map("{:02d}".format)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop(columns=['Year', 'Month', 'DayofMonth'])
    df = df.set_index("Date")
    df = df.sort_index()
    return df

def _set_delayed(df):
    df["Delayed"] = ((df["ArrDelay"] >= 15) | (df["DepDelay"] >= 15)).astype(np.float64)
    df = df.drop(columns=["ArrDelay", "DepDelay"])
    return df

def _smooth_delayed(df):
    # convolucion?
    # butter filter?
    sns.lineplot(df.index, df["Cleaned"])
    df["Cleaned"] = signal.savgol_filter(df["Cleaned"], 5, 2)
    sns.lineplot(df.index, df["Cleaned"])
    plt.show()
    return df

def _build_lstsq_matrix(x):
    return np.stack([x.values, np.ones(x.shape[0])], axis=1)

def _trending(df):
    by_week = df.resample("W")["Cleaned"].mean()
    week_timestamps = by_week.index.astype(np.int64) // 10 ** 9
    coefs = np.linalg.lstsq(_build_lstsq_matrix(week_timestamps), by_week.values, rcond=None)[0]
    timestamps = (df.index.astype(np.int64) // 10 ** 9)
    linear_prediction = _build_lstsq_matrix(timestamps) @ np.array(coefs)
    df["WithoutTrend"] = df["Cleaned"] - linear_prediction
    df["Prediction"] += linear_prediction
    df["Cleaned"] -= linear_prediction

    #plt.figure()
    #sns.lineplot(x=df.index, y="Delayed", data=df)
    #sns.lineplot(x=df.index, y=linear_prediction)
    #plt.figure()
    #sns.lineplot(x=df.index, y="WithoutTrend", data=df)
    return df

def _year_periodicity(df):
    pol = np.polyfit(df.index.dayofyear, df.WithoutTrend, 7)
    year_prediction = np.polyval(pol, df.index.dayofyear)
    df["WithoutYearPeriod"] = df["WithoutTrend"] - year_prediction
    df["Prediction"] += year_prediction
    df["Cleaned"] -= year_prediction
    return df

def _week_periodicity(df):
    pol = np.polyfit(df.index.dayofweek, df["WithoutYearPeriod"], 5)
    week_prediction = np.polyval(pol, df.index.dayofweek)
    df["WithoutWeekPeriod"] = df["WithoutYearPeriod"] - week_prediction
    df["Prediction"] += week_prediction
    df["Cleaned"] -= week_prediction
    return df

def _init_prediction(df):
    df["Prediction"] = 0
    return df

def _init_cleaned(df):
    df["Cleaned"] = df["Delayed"]
    return df

def _plot_cleaned(df):
    pass

def _plot_prediction(df):
    print("RMSE: {}".format(math.sqrt(mean_squared_error(df["Delayed"], df["Prediction"]))))

def main():
    df = _load_original_csv()
    df = _set_date_index(df)
    df = _set_delayed(df)
    df = _init_prediction(df)
    df = _init_cleaned(df)
    df = _smooth_delayed(df)
    _plot_prediction(df)
    df = _trending(df)
    _plot_prediction(df)
    df = _year_periodicity(df)
    _plot_prediction(df)
    df = _week_periodicity(df)
    _plot_prediction(df)

if __name__ == "__main__":
    main()

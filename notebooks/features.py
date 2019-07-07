import pickle
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import math
from scipy import signal
from scipy.fftpack import fft
import os

import time

import airline

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Mirar horario intradia

# Friday's on summer
# Mirar holidays (thanksgiving y otras fechas)
# En 2008, recesion
# scores para los dias de la semana o hora del dia? asi no hacemos una periodica con alta frecuencia


# DONE:
# Score a aeropuertos
# Score a cada aerolinea
# No convertir Delayed en booleano, no perder la informacion de cuanta demora hay (diff entre 14 y 15 min)
# Ajustar e^(-x^2) o algo asi para sacar las hollidays

sns.set()

_PLOTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "informe", "plots")
if not os.path.exists(_PLOTS_FOLDER):
    os.makedirs(_PLOTS_FOLDER)

def _load_original_csv(years):
    columns = ["Year", "Month", "DayofMonth", "ArrDelay", "DepDelay", "UniqueCarrier"]
    dataframes = (pd.read_csv("../data/{}.csv".format(year), usecols=columns) for year in years)
    # return pd.concat(dataframes, ignore_index=False)
    return dataframes


def _set_date_index(df):
    df["Date"] = df["Year"].map("{:04d}".format) + "-" + \
                 df["Month"].map("{:02d}".format) + "-" + \
                 df["DayofMonth"].map("{:02d}".format)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop(columns=['Year', 'Month', 'DayofMonth'])
    return df


def _set_delayed(df):
    df["Delayed"] = np.maximum(0, np.maximum(df["ArrDelay"], df["DepDelay"]))
    df = df[~df["Delayed"].isna()]
    # df["Delayed"] = ((df["ArrDelay"] >= 15) | (df["DepDelay"] >= 15)).astype(np.float64)
    df = df.drop(columns=["ArrDelay", "DepDelay"])
    return df


def _group_by_date(df):
    df = df.groupby(["Date", "UniqueCarrier"])["Delayed"].mean().to_frame()
    df = df.reset_index()
    df = df.set_index("Date")
    return df


def _smooth_delayed(df):
    # convolucion?
    # butter filter?
    # f, (ax1, ax2) = plt.subplots(2, 1)
    # sns.lineplot(df.index, df["Cleaned"], color='b', label="raw", ax=ax1)
    # df["Cleaned"] = signal.savgol_filter(df["Cleaned"], 50001, 3)
    # sns.lineplot(df.index, df["Cleaned"], color='r', label="filtered", ax=ax2)
    # plt.show()
    return df


def _build_lstsq_matrix(x):
    return np.stack([x.values, np.ones(x.shape[0])], axis=1)


def _trending(df):

    day_timestamps = df.index.astype(np.int64) // 10 ** 9
    sns.scatterplot(day_timestamps, df["Cleaned"])
    lstq = airline.LeastSquaresClassifier()
    A = np.stack([day_timestamps, np.ones(x.shape[0])], axis=1)
    coefs = lstq.calculate(_build_lstsq_matrix(day_timestamps), df["Cleaned"], rcond=None)[0]
    timestamps = (df.index.astype(np.int64) // 10 ** 9)
    linear_prediction = _build_lstsq_matrix(timestamps) @ np.array(coefs)
    sns.lineplot(day_timestamps, linear_prediction).set_title("Linear trending")

    # TODO: se pisan los titulos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.scatterplot(df.index, df["Cleaned"], ax=ax1).set_title("Linear trending")
    sns.lineplot(df.index, linear_prediction, ax=ax1, color='r')
    sns.scatterplot(df.index, df["Cleaned"] - linear_prediction, ax=ax2).set_title("Without linear trending")

    ax1.set_xlim([df.index[0], df.index[-1]])
    ax2.set_xlim([df.index[0], df.index[-1]])
    fig.savefig(os.path.join(_PLOTS_FOLDER, "linear_trending.png"))
    plt.close()

    df["WithoutTrend"] = df["Cleaned"] - linear_prediction
    df["Prediction"] += linear_prediction
    df["Cleaned"] -= linear_prediction
    return df


def _second_periodic(df):
    peaks = 3
    N = df.shape[0]
    df["DaysNumber"] = (df.index - df.index[0]) / np.timedelta64(1, 'D')
    years = (df.index[-1] - df.index[0]) / np.timedelta64(1, 'Y')
    factor = 2 * np.pi * years * peaks / N
    functions = [np.sin(df["DaysNumber"] * factor), np.cos(df["DaysNumber"] * factor)]
    A = np.stack(functions, axis=-1)
    coeffs = np.linalg.lstsq(A, df["Cleaned"], rcond=None)[0]
    prediction = A @ coeffs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.scatterplot(df.index, df["Cleaned"], ax=ax1).set_title("Previous")
    sns.lineplot(df.index, prediction, ax=ax1, color='r')
    sns.scatterplot(df.index, df["Cleaned"] - prediction, ax=ax2).set_title("Without prediction")
    # sns.lineplot(df.index, df["Cleaned"] - prediction, ax=ax2)

    ax1.set_xlim([df.index[0], df.index[-1]])
    ax2.set_xlim([df.index[0], df.index[-1]])
    for idx, date in enumerate(df.index):
        if idx == 0 or df.index[idx].year != df.index[idx - 1].year:
            ax1.axvline(x=date, color='k', linestyle='--')
            ax2.axvline(x=date, color='k', linestyle='--')
    fig.savefig(os.path.join(_PLOTS_FOLDER, "second_periodic.png"))

    df["Cleaned"] -= prediction
    return df


def _first_periodic(df):
    for peaks in [1, 2, 3]:
        N = df.index.nunique()
        df["DaysNumber"] = (df.index - df.index[0]) / np.timedelta64(1, 'D')
        years = (df.index[-1] - df.index[0]) / np.timedelta64(1, 'Y')
        factor = 2 * np.pi * years * peaks / N
        functions = [np.sin(df["DaysNumber"] * factor), np.cos(df["DaysNumber"] * factor)]
        A = np.stack(functions, axis=1)
        coeffs = np.linalg.lstsq(A, df["Cleaned"], rcond=None)[0]
        prediction = A @ coeffs

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
        sns.scatterplot(df.index, df["Cleaned"], ax=ax1).set_title("Previous")
        sns.lineplot(df.index, prediction, ax=ax1, color='r')

        sns.scatterplot(df.index, df["Cleaned"] - prediction, ax=ax2).set_title("Without prediction")

        ax1.set_xlim([df.index[0], df.index[-1]])
        ax2.set_xlim([df.index[0], df.index[-1]])
        for idx, date in enumerate(df.index):
            if idx == 0 or df.index[idx].year != df.index[idx - 1].year:
                ax1.axvline(x=date, color='k', linestyle='--')
                ax2.axvline(x=date, color='k', linestyle='--')
        fig.savefig(os.path.join(_PLOTS_FOLDER, "peaks_{}.png".format(peaks)))
        df["Cleaned"] -= prediction
    return df


def _sum_of_periodic(df):
    functions = []
    for peaks in [1, 2, 3]:
        N = df.index.nunique()
        df["DaysNumber"] = (df.index - df.index[0]) / np.timedelta64(1, 'D')
        years = (df.index[-1] - df.index[0]) / np.timedelta64(1, 'Y')
        factor = 2 * np.pi * years * peaks / N
        functions.append(np.sin(df["DaysNumber"] * factor))
        functions.append(np.cos(df["DaysNumber"] * factor))
    A = np.stack(functions, axis=1)
    coeffs = np.linalg.lstsq(A, df["Cleaned"], rcond=None)[0]
    prediction = A @ coeffs
    print("Sum of periodics RMSE: {}".format(math.sqrt(mean_squared_error(df["Cleaned"], prediction))))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.scatterplot(df.index, df["Cleaned"], ax=ax1).set_title("Previous")
    sns.lineplot(df.index, prediction, ax=ax1, color='r')

    sns.scatterplot(df.index, df["Cleaned"] - prediction, ax=ax2).set_title("Without prediction")

    ax1.set_xlim([df.index[0], df.index[-1]])
    ax2.set_xlim([df.index[0], df.index[-1]])
    for idx, date in enumerate(df.index):
        if idx == 0 or df.index[idx].year != df.index[idx - 1].year:
            ax1.axvline(x=date, color='k', linestyle='--')
            ax2.axvline(x=date, color='k', linestyle='--')
    fig.savefig(os.path.join(_PLOTS_FOLDER, "sum_of_periodics.png"))
    return df


def _hollidays(df):
    functions = [1 / (1 + np.minimum(df.index.dayofyear, abs(df.index.dayofyear - 365)) ** 2)]
    A = np.stack(functions, axis=1)
    coeffs = np.linalg.lstsq(A, df["Cleaned"], rcond=None)[0]
    prediction = A @ coeffs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.scatterplot(df.index, df["Cleaned"], ax=ax1).set_title("Previous")
    sns.lineplot(df.index, prediction, ax=ax1, color='r')

    sns.scatterplot(df.index, df["Cleaned"] - prediction, ax=ax2).set_title("Without prediction")

    ax1.set_xlim([df.index[0], df.index[-1]])
    ax2.set_xlim([df.index[0], df.index[-1]])
    for idx, date in enumerate(df.index):
        if idx == 0 or df.index[idx].year != df.index[idx - 1].year:
            ax1.axvline(x=date, color='k', linestyle='--')
            ax2.axvline(x=date, color='k', linestyle='--')
    fig.savefig(os.path.join(_PLOTS_FOLDER, "hollydays_peak.png"))
    return df


def _plot_within_year(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
    sns.lineplot(df.index.month, df["Cleaned"], ax=ax1).set_title("Grouped by month of year")
    sns.boxplot(df.index.month, df["Cleaned"], ax=ax2).set_title("Boxplot")
    months = df.resample("M")["Cleaned"].mean()
    sns.scatterplot(months.index, months.values, ax=ax3).set_title("TODO")
    ax3.set_xlim([months.index[0], months.index[-1]])
    fig.savefig(os.path.join(_PLOTS_FOLDER, "within_year.png"))


def _plot_within_month(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
    sns.lineplot(df.index.day, df["Cleaned"], ax=ax1).set_title("Grouped by day of month")
    sns.boxplot(df.index.day, df["Cleaned"], ax=ax2).set_title("Boxplot")
    sns.scatterplot(df.index[:91], df["Cleaned"][:91], ax=ax3).set_title("Zoomed first months")
    ax3.set_xlim([df.index[0], df.index[90]])
    fig.savefig(os.path.join(_PLOTS_FOLDER, "within_month.png"))


def _plot_within_week(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
    sns.lineplot(df.index.dayofweek, df["Cleaned"], ax=ax1).set_title("Grouped by day of week")
    sns.boxplot(df.index.dayofweek, df["Cleaned"], ax=ax2).set_title("Boxplot")
    sns.scatterplot(df.index[:22], df["Cleaned"][:22], ax=ax3).set_title("Zoomed first months")
    ax3.set_xlim([df.index[0], df.index[21]])
    fig.savefig(os.path.join(_PLOTS_FOLDER, "within_week.png"))


def _within_week_periodic(df):
    peaks = 52 * 2
    N = df.index.nunique()
    df["DaysNumber"] = (df.index - df.index[0]) / np.timedelta64(1, 'D')
    years = (df.index[-1] - df.index[0]) / np.timedelta64(1, 'Y')
    factor = 2 * np.pi * years * peaks / N
    functions = [np.sin(df["DaysNumber"] * factor), np.cos(df["DaysNumber"] * factor)]
    A = np.stack(functions, axis=1)
    coeffs = np.linalg.lstsq(A, df["Cleaned"], rcond=None)[0]
    prediction = A @ coeffs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    limit = 7 * 8 * 10
    sns.scatterplot(df.index[:limit], df[:limit]["Cleaned"], ax=ax1).set_title("Previous")
    sns.lineplot(df.index[:limit], prediction[:limit], ax=ax1, color='r')

    sns.scatterplot(df.index, df["Cleaned"] - prediction, ax=ax2).set_title("Without prediction")

    ax1.set_xlim([df.index[0], df.index[limit]])
    ax2.set_xlim([df.index[0], df.index[-1]])
    for date in df.index:
        if date.is_year_start:
            ax1.axvline(x=date, color='k', linestyle='--')
            ax2.axvline(x=date, color='k', linestyle='--')
    fig.savefig(os.path.join(_PLOTS_FOLDER, "within_week_periodic.png".format(peaks)))
    df["Cleaned"] -= prediction
    return df


def _year_periodicity(df):
    fig = plt.figure()
    sns.scatterplot(df.index.dayofyear, df.WithoutTrend)
    pol = np.polyfit(df.index.dayofyear, df.WithoutTrend, 7)
    year_prediction = np.polyval(pol, df.index.dayofyear)
    sns.lineplot(df.index.dayofyear, year_prediction)
    df["WithoutYearPeriod"] = df["WithoutTrend"] - year_prediction
    df["Prediction"] += year_prediction
    df["Cleaned"] -= year_prediction
    plt.close()
    return df


def _week_periodicity(df) -> pd.DataFrame:
    plt.figure()
    sns.boxplot(df.index.dayofweek, df["WithoutYearPeriod"])
    plt.close()
    plt.figure()
    sns.scatterplot(df.index.dayofweek, df["WithoutYearPeriod"])
    pol = np.polyfit(df.index.dayofweek, df["WithoutYearPeriod"], 5)
    week_prediction = np.polyval(pol, df.index.dayofweek)
    sns.lineplot(df.index.dayofweek, week_prediction, color='r')
    plt.close()
    df["WithoutWeekPeriod"] = df["WithoutYearPeriod"] - week_prediction
    df["Prediction"] += week_prediction
    df["Cleaned"] -= week_prediction
    return df


def _month_periodicity(df):
    plt.figure()
    sns.boxplot(df.index.day, df["WithoutYearPeriod"])
    plt.close()
    plt.figure()
    sns.boxplot(df.index.month, df["WithoutYearPeriod"])
    plt.close()
    return df


def _init_prediction(df):
    df["Prediction"] = 0
    return df


def _init_cleaned(df):
    #df["Cleaned"] = df.groupby(df.index)["Delayed"].transform(np.mean)
    df["Cleaned"] = df["Delayed"]
    return df


def _plot_cleaned(df):
    pass


def _build_least_squares_matrix(df):
    features = []

    features.append(np.ones(df.shape[0]))
    features.append(df.index.astype(np.int) / 10 ** 9)

    for peaks in [1, 2, 3, 4, 12 * 4, 12 * 4 * 2]:
        N = df.index.nunique()
        df["DaysNumber"] = (df.index - df.index[0]) / np.timedelta64(1, 'D')
        years = (df.index[-1] - df.index[0]) / np.timedelta64(1, 'Y')
        factor = 2 * np.pi * years * peaks / N
        # features.append(np.sin(df["DaysNumber"]*factor))
        # features.append(np.cos(df["DaysNumber"]*factor))

    s = 365.0  # observations per annum
    n = int(s // 2)
    df["ts"] = df.index.astype(np.int) / 10 ** 9
    for j in range(0, n + 1):
        factor = 2 * np.pi * j / s
        df["DaysNumber"] = (df.index - df.index[0]) / np.timedelta64(1, 'D')
        features.append(np.sin(df["DaysNumber"] * factor))
        features.append(np.cos(df["DaysNumber"] * factor))

    for peaks in [1, 2, 3, 4, 12 * 4, 12 * 4 * 2]:
        N = df.index.nunique()
        df["DaysNumber"] = (df.index - df.index[0]) / np.timedelta64(1, 'D')
        years = (df.index[-1] - df.index[0]) / np.timedelta64(1, 'Y')
        factor = 2 * np.pi * years * peaks / N
        # features.append(np.sin(df["DaysNumber"]*factor)**2)
        # features.append(np.cos(df["DaysNumber"]*factor)**2)

    # for grade in [1, 2, 3, 4, 5]:
    # features.append(df.index.dayofweek**grade)

    features.append(1 / (1 + np.minimum(df.index.dayofyear, abs(df.index.dayofyear - 365)) ** 2))
    return np.stack(features, axis=1)


def _plot_prediction(df):
    print("RMSE: {}".format(math.sqrt(mean_squared_error(df["Delayed"], df["Prediction"]))))


def _plot_frequencies(df):
    pass


def _train(df):
    A = _build_least_squares_matrix(df)
    coeffs = np.linalg.lstsq(A, df["Delayed"], rcond=None)[0]
    prediction = _build_least_squares_matrix(df) @ coeffs

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(df.index, df["Delayed"], color='r', label='Delayed', ax=ax)
    sns.lineplot(df.index, prediction, color='b', label='Prediction', ax=ax)
    ax.set_xlim([df.index[0], df.index[-1]])
    for date in df.index:
        if date.is_year_start:
            ax.axvline(x=date, color='k', linestyle='--')
    print("Final fitting RMSE: {}".format(math.sqrt(mean_squared_error(df["Delayed"], prediction))))
    fig.savefig(os.path.join(_PLOTS_FOLDER, "final_fiting.png"))
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(20, 10))
    records_limit = 90
    sns.lineplot(df.index[:records_limit], df["Delayed"][:records_limit], color='r', label='Delayed', ax=ax2)
    sns.lineplot(df.index[:records_limit], prediction[:records_limit], color='b', label='Prediction', ax=ax2)
    ax2.set_xlim([df.index[0], df.index[records_limit]])
    fig2.savefig(os.path.join(_PLOTS_FOLDER, "final_fiting_zoomed_first_weeks.png"))
    plt.close()

    return coeffs


def cached_df(prefix: str):
    def decorator(f: Callable[[int], pd.DataFrame]):
        def wraped(year: int):
            filename = os.path.join(
                os.path.dirname(__file__), '..', 'data', prefix + str(year) + '.csv'
            )
            if os.path.isfile(filename):
                df = pd.read_csv(filename)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.reset_index()
                df = df.set_index("Date")
                return df
            else:
                df = f(year)
                df.to_csv(filename)
                return df
        return wraped
    return decorator


@cached_df('processed')
def _get_pre_processed_year(year) -> pd.DataFrame:
    columns = ["Year", "Month", "DayofMonth", "ArrDelay", "DepDelay", "UniqueCarrier"]
    df = pd.read_csv("../data/{}.csv".format(year), usecols=columns)
    df = _set_date_index(df)
    df = _set_delayed(df)
    df = _group_by_date(df)
    return df


def _get_pre_processed_data(years) -> pd.DataFrame:
    return pd.concat(map(_get_pre_processed_year, years), ignore_index=False)

def _fit2(training_years):
    df = _get_pre_processed_data(training_years)
    df = _init_cleaned(df)
    df = _trending(df)
    pass

def _fit(training_years):
    df = _get_pre_processed_data(training_years)
    # df = _init_prediction(df)
    # df = _init_cleaned(df)
    # df = _smooth_delayed(df)
    # df = _trending(df)
    # df = _hollidays(df)
    # df = _sum_of_periodic(df)
    # df = _first_periodic(df)
    # df = _hollidays(df)
    # df = _within_week_periodic(df)
    # _plot_within_year(df)
    # _plot_within_month(df)
    # _plot_within_week(df)
    # df = _second_periodic(df)
    # df = _year_periodicity(df)
    # df = _month_periodicity(df)
    # df = _week_periodicity(df)

    coeffs = _train(df)
    return coeffs


def _predict(test_years, coeffs):
    df = _get_pre_processed_data(test_years)
    df = df[df.index < '2008-09-01']  # remove 2008 crisis
    df = df[~df["Delayed"].isna()]

    prediction = _build_least_squares_matrix(df) @ coeffs

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(df.index, df["Delayed"], color='r', label='Delayed', ax=ax)
    sns.lineplot(df.index, prediction, color='b', label='Prediction', ax=ax)
    ax.set_xlim([df.index[0], df.index[-1]])
    for date in df.index:
        if date.is_year_start:
            ax.axvline(x=date, color='k', linestyle='--')
    print("Final prediction RMSE: {}".format(math.sqrt(mean_squared_error(df["Delayed"], prediction))))
    fig.savefig(os.path.join(_PLOTS_FOLDER, "final_prediction.png"))
    plt.close()


def main():
    ALL_DATA_RANGE = list(range(1987, 2009))
    training_years_count = -3
    training_years = ALL_DATA_RANGE[:training_years_count]
    test_years = ALL_DATA_RANGE[training_years_count:]
    _fit2(training_years)
    #coeffs = _fit(training_years)
    #_predict(test_years, coeffs)


if __name__ == "__main__":
    main()

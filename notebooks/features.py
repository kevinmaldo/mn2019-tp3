import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import seaborn as sns
import os

import airline
import dataset

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

sns.set()

sns.set()

_PLOTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "informe", "plots")
if not os.path.exists(_PLOTS_FOLDER):
    os.makedirs(_PLOTS_FOLDER)

def _trending(df):
    grouped = df.groupby("Date")["Cleaned"].mean().to_frame()
    grouped = grouped.reset_index()

    day_timestamps = grouped["Date"].astype(np.int64) // 10 ** 9
    sns.scatterplot(day_timestamps, grouped["Cleaned"])
    A = np.stack([day_timestamps, np.ones(grouped.shape[0])], axis=1)
    coefs = np.linalg.lstsq(A, grouped["Cleaned"], rcond=None)[0]
    linear_prediction = A @ np.array(coefs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.scatterplot(grouped["Date"], grouped["Cleaned"], ax=ax1).set_title("Linear trending")
    sns.lineplot(grouped["Date"], linear_prediction, ax=ax1, color='r')
    sns.scatterplot(grouped["Date"], grouped["Cleaned"] - linear_prediction, ax=ax2)\
        .set_title("Without linear trending")

    ax1.set_xlim([grouped["Date"].min(), grouped["Date"].max()])
    ax2.set_xlim([grouped["Date"].min(), grouped["Date"].max()])
    ax1.set(ylabel="Mean delayed time (min)")
    ax2.set(ylabel="Mean delayed time (min)")
    fig.savefig(os.path.join(_PLOTS_FOLDER, "linear_trending.png"))
    plt.close()

    total_A = np.stack([df["Date"].astype(np.int64) // 10 ** 9, np.ones(df.shape[0])], axis=1)
    df["Cleaned"] -= total_A @ coefs
    return df


def _init_cleaned(df):
    df["Cleaned"] = df["Delayed"]
    return df


def _autocorrelation(df):
    df = df.copy()
    df["DaysNumber"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, 'D')
    df = df.groupby("DaysNumber")["Cleaned"].mean()
    fig = plt.figure()
    ax = autocorrelation_plot(df)
    ax.set_xlim([0, 366])
    fig.savefig(os.path.join(_PLOTS_FOLDER, "autocorrelation.png"))
    return df


def _plot_within_year(df):
    df = df.groupby("Date")["Cleaned"].mean().to_frame().reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.lineplot(df["Date"].dt.month, df["Cleaned"], ax=ax1).set_title("Grouped by month of year")
    sns.boxplot(df["Date"].dt.month, df["Cleaned"], ax=ax2).set_title("Boxplot")
    fig.savefig(os.path.join(_PLOTS_FOLDER, "within_year.png"))


def _plot_within_month(df):
    df = df.groupby("Date")["Cleaned"].mean().to_frame().reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.lineplot(df["Date"].dt.day, df["Cleaned"], ax=ax1).set_title("Grouped by day of month")
    sns.boxplot(df["Date"].dt.day, df["Cleaned"], ax=ax2).set_title("Boxplot")
    fig.savefig(os.path.join(_PLOTS_FOLDER, "within_month.png"))


def _plot_within_week(df):
    df = df.groupby("Date")["Cleaned"].mean().to_frame().reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    sns.lineplot(df["Date"].dt.dayofweek, df["Cleaned"], ax=ax1).set_title("Grouped by day of week")
    sns.boxplot(df["Date"].dt.dayofweek, df["Cleaned"], ax=ax2).set_title("Boxplot")
    fig.savefig(os.path.join(_PLOTS_FOLDER, "within_week.png"))


def _plot_carrier_score(df):
    scores = pd.read_csv(os.path.join("..", "scores", "scores_carrier_2008.csv"))
    scores.columns = ["Carrier", "Score", "Count"]
    scores = scores.sort_values(by="Score", ascending=False).iloc[:10]
    fig = plt.figure()
    sns.barplot(y="Carrier", x="Score", data=scores)
    fig.savefig(os.path.join(_PLOTS_FOLDER, "carrier_scores_2008.png"))


def _plot_airport_score(filename, ax):
    scores = pd.read_csv(os.path.join("..", "scores", filename))
    scores.columns = ["Airport", "Score", "Count"]
    scores = scores.sort_values(by="Score", ascending=False).iloc[:10]
    sns.barplot(y="Airport", x="Score", data=scores, ax=ax)
    ax.set_xlim([10, 31])


def _plot_airport_both_scores(df):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    _plot_airport_score("scores_origin_2008.csv", ax1)
    _plot_airport_score("scores_dest_2008.csv", ax2)
    fig.savefig(os.path.join(_PLOTS_FOLDER, "airport_scores_2008.png"))


def experiments(years):
    df = dataset.get_pre_processed_data(years)
    df = _init_cleaned(df)
    df = _trending(df)
    # _autocorrelation(df)
    _plot_within_year(df)
    _plot_within_month(df)
    _plot_within_week(df)
    _plot_carrier_score(df)
    _plot_airport_both_scores(df)
    return df


def main():
    ALL_DATA_RANGE = list(range(2002, 2009))
    training_years_count = -3
    training_years = ALL_DATA_RANGE[:training_years_count]
    experiments(training_years)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import matplotlib
matplotlib.use('TkAgg')

import dataset
import airline
import numpy as np
import math
from sklearn.metrics import mean_squared_error, accuracy_score, balanced_accuracy_score, precision_score, recall_score
import scores
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

_PLOTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "informe", "plots")

def beep():
    duration = 0.3  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


sample_size = -1


class Model:

    def __init__(self, data_range):
        self._coeffs = None
        self.coeff_labels = []
        self.classifier = None

    def _build_lstq_matrix(self, df):

        features = []
        self.coeff_labels = []

        def add_feature(name, values):
            self.coeff_labels.append(name)
            features.append(values)

        # add_feature('ones', np.ones(df.shape[0]))

        s = 365.0  # observations per annum
        days_number = (df['Date'] - pd.Timestamp('1986-01-01')).dt.days
        add_feature('day', days_number)

        # freq_years = [
        #     0, 1, 2, 3, 4, 5, 6,  # some peaks per year
        #     10, 11, 12, 13, 14, 15,  # around one peak per month
        #     21, 22, 23, 24, 25, 26,  # around two peaks per month
        #     50, 51, 52, 53, 54, 55,  # around one peak per week
        #     102, 103, 104, 105, 106, 107,  # around two peaks per week
        #     363, 364, 365, 366, 367  # around one peak per day
        # ]

        freq_years = [1, 2, 3, 4, 6, 12]
        for j in freq_years:
            factor = 2 * np.pi * j / s
            add_feature(f'sin year/{j}', np.sin(days_number * factor))
            add_feature(f'cos year/{j}', np.cos(days_number * factor))

        # add_feature(f'sin day', np.sin(days_number % 7 * 2 * np.pi / 7))
        # add_feature(f'cos day', np.cos(days_number % 7 * 2 * np.pi / 7))

        for day in range(7):
            add_feature(f'day {day}', days_number % 7 == day)

        # holidays = [185, 245, 287, 315, 332, 359]
        # for holiday in holidays:
        #     add_feature(f'holiday {holiday}', days_number == holiday)

        add_feature('End of year',
                    1 / (1 + np.minimum(df['Date'].dt.dayofyear, abs(df['Date'].dt.dayofyear - 365)) ** 2))

        add_feature("OriginScore", df["OriginScore"])
        add_feature("DestScore", df["DestScore"])
        add_feature("CarrierScore", df["CarrierScore"])
        return np.stack(features, axis=1)

    def fit(self, df):
        A = self._build_lstq_matrix(df)
        # self._coeffs = np.linalg.lstsq(A, df["Delayed"], rcond=None)[0]
        sample = np.random.randint(0, len(A), size=sample_size)
        if sample_size > 0:
            self.classifier = airline.LeastSquaresClassifier()
            self._coeffs = self.classifier.calculate(A[sample], df["Delayed"].to_numpy()[sample])
        else:
            self._coeffs = airline.lstsq(A, df["Delayed"], rcond=None)[0]
        return self

    def show_coeffs(self):
        for label, coeff in zip(self.coeff_labels, self._coeffs):
            print(label, f'{label}: {coeff:0.5f}')

    def predict(self, df):
        return self._build_lstq_matrix(df) @ self._coeffs


def _is_delayed(delayed_times, t=15):
    return (delayed_times >= t).astype(np.float64)


ALL_DATA_RANGE = list(range(2002, 2009))


def nrmse(xs, ys):
    return rmse(xs, ys) / (np.max(xs) - np.min(xs))


def rmse(xs, ys):
    return math.sqrt(np.mean((xs-ys)**2))


def accuracy_threshold_plot(prediction, test_df):
    mins = list(range(1, 30))
    bass = [
        balanced_accuracy_score(
            _is_delayed(test_df.Delayed, t),
            _is_delayed(prediction, t)
        )
        for t in mins
    ]
    plt.plot(mins, bass)
    plt.xlabel("threshold (minutes)")
    plt.ylabel("balanced precision")
    plt.savefig(os.path.join(_PLOTS_FOLDER, "accuracy_by_thresshold.png"))


def show_scores(prediction, test_df):
    print(f"Delay RMSE: {rmse(test_df.Delayed, prediction)}")
    print(f"Delay NRMSE: {nrmse(test_df.Delayed, prediction)}")
    test_is_delayed = _is_delayed(test_df.Delayed)
    pred_is_delayed = _is_delayed(prediction)
    print(f"RMSE: {rmse(test_is_delayed, pred_is_delayed)}")
    print(f"Accuracy: {accuracy_score(test_is_delayed, pred_is_delayed)}")
    print(f"Precision: {precision_score(test_is_delayed, pred_is_delayed)}")
    print(f"Recall: {recall_score(test_is_delayed, pred_is_delayed)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(test_is_delayed, pred_is_delayed)}")

def _report_metrics(training_years_count):
    print(f"Running with training years count: {training_years_count}")
    model = Model(ALL_DATA_RANGE)

    training_years = ALL_DATA_RANGE[:training_years_count]
    test_years = ALL_DATA_RANGE[training_years_count:]

    train_df = dataset.get_pre_processed_data(training_years).reset_index()

    model.fit(train_df)
    #model.show_coeffs()

    train_df["Prediction"] = model.predict(train_df)

    test_df = dataset.get_pre_processed_data(test_years)
    test_df = test_df[test_df["Date"] < '2008-09-01']  # remove the 2008 recesion

    prediction = model.predict(test_df)

    show_scores(prediction, test_df)
    beep()

    accuracy_threshold_plot(prediction, test_df)

    test_df["Prediction"] = prediction

    final_df = pd.concat([train_df, test_df], sort=False)
    final_df["Prediction"] = model.predict(final_df)
    final_grouped = final_df.set_index("Date").resample("W").mean()
    #fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    #sns.lineplot(final_grouped.index, final_grouped["Delayed"], color='r', label="data", ax=ax)
    #sns.lineplot(final_grouped.index, final_grouped["Prediction"], color='b', label="prediction", ax=ax)
    #ax.set(ylabel="Delayed time (min)")
    #ax.axvline(x=test_df["Date"].min(), linestyle="--")
    #fig.savefig(os.path.join(_PLOTS_FOLDER, "example_fit_and_prediction_{}.png".format(training_years_count)))
    #plt.close()


def main():
    for training_years_count in range(1, len(ALL_DATA_RANGE)):
        _report_metrics(training_years_count)

if __name__ == "__main__":
    sample_size = 50_000
    main()

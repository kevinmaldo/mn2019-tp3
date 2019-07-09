import pickle
from typing import Callable
import numpy as np
import pandas as pd
import os
import scores

def _set_date_index(df):
    df["Date"] = df["Year"].map("{:04d}".format) + "-" + \
                 df["Month"].map("{:02d}".format) + "-" + \
                 df["DayofMonth"].map("{:02d}".format)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop(columns=['Year', 'Month', 'DayofMonth'])
    return df


def _set_delayed(df):
    df["Delayed"] = np.minimum(30, np.maximum(0, np.maximum(df["ArrDelay"], df["DepDelay"])))
    df = df[~df["Delayed"].isna()]
    df = df.drop(columns=["ArrDelay", "DepDelay"])
    return df

def _set_scores(df, year):
    carrier_scores = scores.get_carrier_scores(year)
    origin_scores = scores.get_origin_scores(year)
    dest_scores = scores.get_dest_scores(year)
    df["CarrierScore"] = df["UniqueCarrier"].map(carrier_scores.__getitem__)
    df["OriginScore"] = df["Origin"].map(origin_scores.__getitem__)
    df["DestScore"] = df["Dest"].map(dest_scores.__getitem__)
    df = df.drop(columns=["UniqueCarrier", "Origin", "Dest"])
    return df

def _group_by_date(df):
    df = df.groupby(["Date", "UniqueCarrier", "Origin", "Dest"])["Delayed"].mean().to_frame()
    df = df.reset_index()
    return df


def cached_df(prefix: str):
    def decorator(f: Callable[[int], pd.DataFrame]):
        def wraped(year: int):
            filename = os.path.join(
                os.path.dirname(__file__), '..', 'data', prefix + str(year) + '.csv'
            )
            if os.path.isfile(filename):
                df = pd.read_csv(filename)
                df["Date"] = pd.to_datetime(df["Date"])
                return df
            else:
                df = f(year)
                df.to_csv(filename)
                return df
        return wraped
    return decorator


@cached_df('processed')
def _get_pre_processed_year(year) -> pd.DataFrame:
    columns = ["Year", "Month", "DayofMonth", "ArrDelay", "DepDelay",
               "UniqueCarrier", "Origin", "Dest"]
    df = pd.read_csv("../data/{}.csv".format(year), usecols=columns)
    df = _set_date_index(df)
    df = _set_delayed(df)
    df = _group_by_date(df)
    df = _set_scores(df, year)
    return df


def get_pre_processed_data(years) -> pd.DataFrame:
    return pd.concat(map(_get_pre_processed_year, years), ignore_index=False)

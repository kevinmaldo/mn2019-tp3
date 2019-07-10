import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

_AIRLINES = ["AA", "AS", "B6", "DL", "HA", "OO", "UA", "WN"]

def _get_flies_count():
    ALL_DATA_RANGE = list(range(2002, 2009))
    columns = ["Year", "Month", "DayofMonth", "UniqueCarrier"]
    dfs = []
    for year in ALL_DATA_RANGE:
        dfs.append(pd.read_csv(os.path.join("..", "data", "{}.csv".format(year)), usecols=columns))
    df = pd.concat(dfs, ignore_index=False)
    df["Date"] = df["Year"].map("{:04d}".format) + "-" + \
                 df["Month"].map("{:02d}".format) + "-" + \
                 df["DayofMonth"].map("{:02d}".format)
    df["Date"] = pd.to_datetime(df["Date"])
    flies_df = df.drop(columns=['Year', 'Month', 'DayofMonth'])
    flies_df = flies_df.groupby(["Date", "UniqueCarrier"]).size().to_frame()
    flies_df = flies_df.reset_index()
    flies_df = flies_df.rename(columns={0: "Count", "UniqueCarrier": "Carrier"})
    flies_df = flies_df[flies_df["Carrier"].isin(_AIRLINES)]
    return flies_df

def _get_stock_prices():
    dfs = []
    for airline in _AIRLINES:
        df = pd.read_csv(os.path.join("..", "carrier_stock_prices", airline + ".csv"))
        df["Carrier"] = airline
        df["Date"] = pd.to_datetime(df["Date"])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=False)
    df = df.sort_values(by=["Date"])
    return df

def _make_bar_plot(flies_df, prices_df):
    count_by_carrier = pd.pivot_table(flies_df,
                                      index="Date", columns="Carrier", values="Count",
                                      aggfunc=np.sum, fill_value=0).resample("W").sum()
    prices_by_carrier = pd.pivot_table(prices_df,
                                      index="Date", columns="Carrier", values="Close",
                                      aggfunc=np.mean, fill_value=0).resample("W").mean()
    fig, axs = plt.subplots(2, 1, sharex=True)

    accums = [np.zeros(count_by_carrier.shape[0]), np.zeros(prices_by_carrier.shape[0])]
    colors = list(sns.xkcd_rgb.values())[:100]
    for plot_idx, pivot_table in enumerate([count_by_carrier, prices_by_carrier]):
        for airline_idx, airline in enumerate(_AIRLINES):
            airline_serie = pivot_table[airline]
            axs[plot_idx].bar(pivot_table.index, airline_serie,
                        bottom=accums[plot_idx], color=colors[airline_idx], width=1.0)
            accums[plot_idx] += airline_serie.values
    plt.show()
    plt.close()

def _make_correlations_plot(count_pivot_table, prices_pivot_table):
    count_sum = count_pivot_table.sum(axis=1)
    prices_sum = prices_pivot_table.sum(axis=1)
    for airline_idx, airline in enumerate(_AIRLINES):
        count_ratio = count_pivot_table[airline] / count_sum
        price_ratio = prices_pivot_table[airline] / prices_sum
        fig, ax = plt.subplots(1, 1)
        sns.lineplot(count_sum.index, count_ratio, ax=ax)
        sns.lineplot(prices_sum.index, price_ratio, ax=ax)
        plt.show()
        plt.close()

def main():
    flies_df = _get_flies_count()
    prices_df = _get_stock_prices()
    count_by_carrier = pd.pivot_table(flies_df,
                                      index="Date", columns="Carrier", values="Count",
                                      aggfunc=np.sum, fill_value=0).resample("W").sum()
    prices_by_carrier = pd.pivot_table(prices_df,
                                      index="Date", columns="Carrier", values="Close",
                                      aggfunc=np.mean, fill_value=0).resample("W").mean()
    _make_correlations_plot(count_by_carrier, prices_by_carrier)
    return
    _make_bar_plot(flies_df, prices_df)
    return
    for airline in ["AA", "AS", "B6", "DL", "HA", "OO", "UA", "WN"]:

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        sns.lineplot(x="Date", y="Close", data=prices_df[prices_df["Carrier"] == airline], ax=ax1)
        sns.lineplot(x="Date", y="Count", data=flies_df[flies_df["Carrier"] == airline], ax=ax2)
        plt.show()

if __name__ == "__main__":
    main()
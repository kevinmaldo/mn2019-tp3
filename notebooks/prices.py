import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import matplotlib.ticker as ticker

_AIRLINES = ["AA", "AS", "B6", "DL", "HA", "OO", "UA", "WN"]
_PLOTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "informe", "plots")

def _get_flights_count():
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
    #df = df[df["Date"] < '2008-09-01']
    flights_df = df.drop(columns=['Year', 'Month', 'DayofMonth'])
    flights_df = flights_df.groupby(["Date", "UniqueCarrier"]).size().to_frame()
    flights_df = flights_df.reset_index()
    #flights_df["Date"] = pd.to_datetime(flights_df["Date"])
    flights_df = flights_df.rename(columns={0: "Count", "UniqueCarrier": "Carrier"})
    flights_df = flights_df[flights_df["Carrier"].isin(_AIRLINES)]
    return flights_df

def _get_stock_prices():
    dfs = []
    for airline in _AIRLINES:
        df = pd.read_csv(os.path.join("..", "carrier_stock_prices", airline + ".csv"))
        df["Carrier"] = airline
        df["Date"] = pd.to_datetime(df["Date"])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=False)
    #df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Date"])
    return df

def _make_bar_plot(flights_df, prices_df):
    count_by_carrier = pd.pivot_table(flights_df,
                                      index="Date", columns="Carrier", values="Count",
                                      aggfunc=np.sum, fill_value=0).resample("W").sum()
    prices_by_carrier = pd.pivot_table(prices_df,
                                      index="Date", columns="Carrier", values="Close",
                                      aggfunc=np.mean, fill_value=0).resample("W").mean()
    fig, axs = plt.subplots(2, 1, sharex=True)

    accums = [np.zeros(count_by_carrier.shape[0]), np.zeros(prices_by_carrier.shape[0])]
    colors = sns.color_palette("hls", len(_AIRLINES))
    for plot_idx, pivot_table in enumerate([count_by_carrier, prices_by_carrier]):
        for airline_idx, airline in enumerate(_AIRLINES):
            airline_serie = pivot_table[airline]
            g = sns.barplot(pivot_table.index, airline_serie,
                        bottom=accums[plot_idx], color=colors[airline_idx], ax=axs[plot_idx], label=airline)
            #g.set(xticks=pivot_table["Date"][::60])
            accums[plot_idx] += airline_serie.values
    axs[0].set(ylabel="Flights count")
    axs[1].set(ylabel="Share price")
    axs[0].legend()
    axs[1].legend()
    fig.savefig(os.path.join(_PLOTS_FOLDER, "flights_and_stock_prices.png"))
    plt.close()

def _make_correlations_plot(count_pivot_table, prices_pivot_table):
    count_sum = count_pivot_table.sum(axis=1)
    prices_sum = prices_pivot_table.sum(axis=1)
    for airline_idx, airline in enumerate(_AIRLINES):
        count_ratio = count_pivot_table[airline] / count_sum
        price_ratio = prices_pivot_table[airline] / prices_sum
        idxs = (count_ratio > 0.0) & (price_ratio > 0.0)
        fig, ax = plt.subplots(1, 1, sharex=True)
        sns.lineplot(count_sum.loc[idxs].index, count_ratio[idxs], ax=ax, label="Flies ratio").set_title("Airline: " + airline)
        sns.lineplot(prices_sum.loc[idxs].index, price_ratio[idxs], ax=ax, label="Stock price ratio")
        pearson = scipy.stats.pearsonr(count_ratio[idxs], price_ratio[idxs])[0]
        print("Pearsion for airline {}: {}".format(airline, pearson))
        plt.close()

def main():
    flights_df = _get_flights_count()
    prices_df = _get_stock_prices()
    count_by_carrier = pd.pivot_table(flights_df,
                                      index="Date", columns="Carrier", values="Count",
                                      aggfunc=np.sum, fill_value=0).resample("W").sum()
    prices_by_carrier = pd.pivot_table(prices_df,
                                      index="Date", columns="Carrier", values="Close",
                                      aggfunc=np.mean, fill_value=0).resample("W").mean()
    _make_correlations_plot(count_by_carrier, prices_by_carrier)
    #_make_bar_plot(flights_df, prices_df)
    return
    for airline in ["AA", "AS", "B6", "DL", "HA", "OO", "UA", "WN"]:

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        sns.lineplot(x="Date", y="Close", data=prices_df[prices_df["Carrier"] == airline], ax=ax1)
        sns.lineplot(x="Date", y="Count", data=flights_df[flights_df["Carrier"] == airline], ax=ax2)
        plt.show()

if __name__ == "__main__":
    main()
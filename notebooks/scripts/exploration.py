"""
All columns:
    'Year', 'Month', 'DayofMonth', 'DayOfWeek',
    'DepTime', 'CRSDepTime',
    'ArrTime', 'CRSArrTime',
    'UniqueCarrier', 'FlightNum', 'TailNum',
    'ActualElapsedTime', 'CRSElapsedTime', 'AirTime',
    'ArrDelay', 'DepDelay',
    'Origin', 'Dest', 'Distance',
    'TaxiIn', 'TaxiOut',
    'Cancelled', 'CancellationCode', 'Diverted',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'
"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR: Path = Path(__file__).parents[2] / 'data'
PLOT_DIR: Path = Path(__file__).parents[2] / 'plots'


def get_year(year: int) -> pd.DataFrame:
    # columns = ["Year", "Month", "DayofMonth", "ArrDelay", "DepDelay", "UniqueCarrier"]
    columns = None
    df = pd.read_csv(DATA_DIR / f'{year}.csv', usecols=columns, encoding='cp1252')
    return set_date_index(df)


def set_date_index(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df["Year"].map("{:04d}".format) + "-" + \
                 df["Month"].map("{:02d}".format) + "-" + \
                 df["DayofMonth"].map("{:02d}".format)
    df["Date"] = pd.to_datetime(df["Date"])
    # df = df.drop(columns=['Year', 'Month', 'DayofMonth'])
    return df


def departure_vs_arrive():

    (PLOT_DIR / 'Arr_vs_Dep_Delays').mkdir(exist_ok=True)

    for year in range(1987, 2009):
        df = get_year(year)
        sample = df[df['DepDelay'] > 0].sample(1000)
        xm = sample[['ArrDelay', 'DepDelay']].max().to_numpy().max()
        sample.plot('ArrDelay', 'DepDelay', marker="o", ls='')
        plt.title(f'{year}')
        plt.plot([0, xm], [0, xm])
        plt.savefig(PLOT_DIR / 'Arr_vs_Dep_Delays' / f'{year}.png')
        plt.close()
        print(f'year = {year}')
        # plt.show()


if __name__ == '__main__':
    departure_vs_arrive()

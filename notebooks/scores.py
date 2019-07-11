import csv
import os


def _load_file(score_type, year):
    filename = os.path.join(
        os.path.dirname(__file__), '..', 'scores', 'scores_{}_{}.csv'.format(score_type, year)
    )
    ret = {}
    with open(filename, newline='') as csvfile:
        scores_reader = csv.reader(csvfile, delimiter=',')
        for row in scores_reader:
            ret[row[0]] = float(row[1])
    return ret


def get_carrier_scores(year):
    return _load_file('carrier', year)


def get_origin_scores(year):
    return _load_file('origin', year)


def get_dest_scores(year):
    return _load_file('dest', year)

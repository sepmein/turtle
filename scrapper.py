"""
    Scrap quandl for the bitcoin data
"""

# imports
from datetime import date, datetime, timedelta

import pandas as pd
import quandl

from config import api_key, days_before, feature_labels
from preprocess import gen_days_back, fit_transform, interpolate

quandl.ApiConfig.api_key = api_key

all_labels = feature_labels


def scrap_all():
    """
        scrap_all function
    """

    DIFF = quandl.get("BCHAIN/DIFF")
    TRFEE = quandl.get('BCHAIN/TRFEE')
    MKTCP = quandl.get('BCHAIN/MKTCP')
    TOTBC = quandl.get('BCHAIN/TOTBC')
    MWNUS = quandl.get('BCHAIN/MWNUS')
    BCDDY = quandl.get('BCHAIN/BCDDY')
    BCDDM = quandl.get('BCHAIN/BCDDM')
    BCDDE = quandl.get('BCHAIN/BCDDE')
    TVTVR = quandl.get('BCHAIN/TVTVR')
    NETDF = quandl.get('BCHAIN/NETDF')
    MIOPM = quandl.get('BCHAIN/MIOPM')
    MWNTD = quandl.get('BCHAIN/MWNTD')
    MWTRV = quandl.get('BCHAIN/MWTRV')
    AVBLS = quandl.get('BCHAIN/AVBLS')
    BLCHS = quandl.get('BCHAIN/BLCHS')
    ATRCT = quandl.get('BCHAIN/ATRCT')
    MIREV = quandl.get('BCHAIN/MIREV')
    HRATE = quandl.get('BCHAIN/HRATE')
    CPTRA = quandl.get('BCHAIN/CPTRA')
    CPTRV = quandl.get('BCHAIN/CPTRV')
    TRVOU = quandl.get('BCHAIN/TRVOU')
    TOUTV = quandl.get('BCHAIN/TOUTV')
    ETRVU = quandl.get('BCHAIN/ETRVU')
    ETRAV = quandl.get('BCHAIN/ETRAV')
    NTRBL = quandl.get('BCHAIN/NTRBL')
    NADDU = quandl.get('BCHAIN/NADDU')
    NTREP = quandl.get('BCHAIN/NTREP')
    NTRAT = quandl.get('BCHAIN/NTRAT')
    NTRAN = quandl.get('BCHAIN/NTRAN')
    # target value
    MKPRU = quandl.get('BCHAIN/MKPRU')
    # manipulating data
    # concacenating
    DATA = pd.concat(
        [
            DIFF, TRFEE, MKTCP, TOTBC, MWNUS, BCDDY, BCDDM, BCDDE, TVTVR,
            NETDF, MIOPM, MWNTD, MWTRV, AVBLS, BLCHS, ATRCT, MIREV, HRATE,
            CPTRA, CPTRV, TRVOU, TOUTV, ETRVU, ETRAV, NTRBL, NADDU, NTREP,
            NTRAT, NTRAN, MKPRU
        ],
        axis=1)

    DATA.columns = feature_labels
    return DATA


# scrap a specific date
def scrap(year, month, day):
    """
    To scrap bitcoin data from quandl and preprocess it using predefined preprocess module
    :param date: a date string
    :return: a data frame object to be feed to model
    """
    # check out the date inputted is 'today' or not
    # FIXME quandl api timezone and china timezone not compatible, the latest data will be NaN ocassionally (8/24?)
    if year is None:
        target_date = datetime.today()
    else:
        target_date = date(year, month, day)

    # days back
    end_date = target_date - timedelta(days=1)
    days_interval = timedelta(days=days_before - 1)
    start_date = end_date - days_interval
    # get the date of today if need
    result = []
    for label in all_labels:
        temp = quandl.get('BCHAIN/' + label, start_date=start_date, end_date=end_date)
        result.append(temp)

    data_frame = pd.concat(result, axis=1)
    data_frame.columns = all_labels

    # interpolation
    interpolated = interpolate(data_frame)

    # interpolated.to_csv('data.csv')
    # use data preprocess api to generate data to feed model
    gen_df, labels = gen_days_back(data=interpolated,
                                   labels=feature_labels,
                                   days=days_before)

    # perform data transformation
    gen_df = fit_transform(gen_df)
    return gen_df

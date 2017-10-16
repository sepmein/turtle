"""
    Scrap quandl for the bitcoin data
"""

# imports
from datetime import datetime, timedelta
from config import api_key, days_before, feature_labels
import pandas as pd
import quandl
from preprocess import gen_days_back, fit_transform

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
def scrap(date='today'):
    # check out the date inputted is 'today' or not
    if date is 'today':
        date = datetime.today()
    else:
        date = date

    # days back
    days_interval = timedelta(days=days_before - 1)
    date_before = date - days_interval
    # get the date of today if need
    result = []
    for label in all_labels:
        temp = quandl.get('BCHAIN/' + label, start_date=date_before, end_date=date)
        result.append(temp)

    data_frame = pd.concat(result, axis=1)
    data_frame.columns = all_labels
    gen_df = gen_days_back(data=data_frame,
                           labels=feature_labels,
                           days=days_before)
    gen_df = fit_transform(gen_df)
    return gen_df

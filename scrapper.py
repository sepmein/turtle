"""
    Scrap quandl for the bitcoin data
"""

from datetime import datetime

import pandas as pd
# imports
import quandl

# api key
quandl.ApiConfig.api_key = "6ywQ69kRqt26zAsHkFDP"

# labels
LABELS = [
    'DIFF', 'TRFEE', 'MKTCP', 'TOTBC',
    'MWNUS', 'BCDDY', 'BCDDM', 'BCDDE',
    'TVTVR', 'NETDF', 'MIOPM', 'MWNTD',
    'MWTRV', 'AVBLS', 'BLCHS', 'ATRCT',
    'MIREV', 'HRATE', 'CPTRA', 'CPTRV',
    'TRVOU', 'TOUTV', 'ETRVU', 'ETRAV',
    'NTRBL', 'NADDU', 'NTREP', 'NTRAT',
    'NTRAN', 'MKPRU'
]


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
    ## target value
    MKPRU = quandl.get('BCHAIN/MKPRU')
    ## manipulating data
    # concacenating
    DATA = pd.concat(
        [
            DIFF, TRFEE, MKTCP, TOTBC, MWNUS, BCDDY, BCDDM, BCDDE, TVTVR,
            NETDF, MIOPM, MWNTD, MWTRV, AVBLS, BLCHS, ATRCT, MIREV, HRATE,
            CPTRA, CPTRV, TRVOU, TOUTV, ETRVU, ETRAV, NTRBL, NADDU, NTREP,
            NTRAT, NTRAN, MKPRU
        ],
        axis=1)

    DATA.columns = LABELS
    return DATA


# scrap a specific date
def scrap(date='today'):
    # check out the date inputted is 'today' or not
    if date is 'today':
        date = datetime.today()
    else:
        date = date

    # get the date of today if need
    result = []
    result.append(quandl.get(k) for k in LABELS)
    # define the labels to be scraped

    # scrap the data 50 days_before back

    # concacenating data

    # return data

''''''
Scrap quandl for the bitcoin data
''''''
import quandl
import numpy as np
import tensorflow as tf
import pandas as pd
quandl.ApiConfig.api_key = "6ywQ69kRqt26zAsHkFDP"
## data
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
        DIFF, TRFEE, MKTCP, TOTBC, MWNUS, BCDDY, BCDDM, BCDDE, TVTVR, NETDF,
        MIOPM, MWNTD, MWTRV, AVBLS, BLCHS, ATRCT, MIREV, HRATE, CPTRA, CPTRV,
        TRVOU, TOUTV, ETRVU, ETRAV, NTRBL, NADDU, NTREP, NTRAT, NTRAN, MKPRU
    ],
    axis=1)

## analysing the data

## define feature column
f_DIFF = tf.contrib.layers.real_valued_column('DIFF', dimension=10)
f_TRFEE = tf.contrib.layers.real_valued_column('TRFEE', dimension=10)
f_MKTCP = tf.contrib.layers.real_valued_column('MKTCP', dimension=10)
f_TOTBC = tf.contrib.layers.real_valued_column('TOTBC', dimension=10)
f_MWNUS = tf.contrib.layers.real_valued_column('MWNUS', dimension=10)
f_BCDDY = tf.contrib.layers.real_valued_column('BCDDY', dimension=10)
f_BCDDM = tf.contrib.layers.real_valued_column('BCDDM', dimension=10)
f_BCDDE = tf.contrib.layers.real_valued_column('BCDDE', dimension=10)
f_TVTVR = tf.contrib.layers.real_valued_column('TVTVR', dimension=10)
f_NETDF = tf.contrib.layers.real_valued_column('NETDF', dimension=10)
f_MIOPM = tf.contrib.layers.real_valued_column('MIOPM', dimension=10)
f_MWNTD = tf.contrib.layers.real_valued_column('MWNTD', dimension=10)
f_MWTRV = tf.contrib.layers.real_valued_column('MWTRV', dimension=10)
f_AVBLS = tf.contrib.layers.real_valued_column('AVBLS', dimension=10)
f_BLCHS = tf.contrib.layers.real_valued_column('BLCHS', dimension=10)
f_ATRCT = tf.contrib.layers.real_valued_column('ATRCT', dimension=10)
f_MIREV = tf.contrib.layers.real_valued_column('MIREV', dimension=10)
f_HRATE = tf.contrib.layers.real_valued_column('HRATE', dimension=10)
f_CPTRA = tf.contrib.layers.real_valued_column('CPTRA', dimension=10)
f_CPTRV = tf.contrib.layers.real_valued_column('CPTRV', dimension=10)
f_TRVOU = tf.contrib.layers.real_valued_column('TRVOU', dimension=10)
f_TOUTV = tf.contrib.layers.real_valued_column('TOUTV', dimension=10)
f_ETRVU = tf.contrib.layers.real_valued_column('ETRVU', dimension=10)
f_ETRAV = tf.contrib.layers.real_valued_column('ETRAV', dimension=10)
f_NTRBL = tf.contrib.layers.real_valued_column('NTRBL', dimension=10)
f_NADDU = tf.contrib.layers.real_valued_column('NADDU', dimension=10)
f_NTREP = tf.contrib.layers.real_valued_column('NTREP', dimension=10)
f_NTRAT = tf.contrib.layers.real_valued_column('NTRAT', dimension=10)
f_NTRAN = tf.contrib.layers.real_valued_column('NTRAN', dimension=10)

features = [
    f_DIFF, f_TRFEE, f_MKTCP, f_TOTBC, f_MWNUS, f_BCDDY, f_BCDDM, f_BCDDE,
    f_TVTVR, f_NETDF, f_MIOPM, f_MWNTD, f_MWTRV, f_AVBLS, f_BLCHS, f_ATRCT,
    f_MIREV, f_HRATE, f_CPTRA, f_CPTRV, f_TRVOU, f_TOUTV, f_ETRVU, f_ETRAV,
    f_NTRBL, f_NADDU, f_NTREP, f_NTRAT, f_NTRAN
]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

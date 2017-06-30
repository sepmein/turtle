"""
Read data from csv
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

rawdata = pd.read_csv('cleaned_data.csv')
data = rawdata.iloc[600:2500]

# build input layers
f_DIFF = tf.contrib.layers.real_valued_column('DIFF')
f_TRFEE = tf.contrib.layers.real_valued_column('TRFEE')
f_MKTCP = tf.contrib.layers.real_valued_column('MKTCP')
f_TOTBC = tf.contrib.layers.real_valued_column('TOTBC')
f_MWNUS = tf.contrib.layers.real_valued_column('MWNUS')
f_MWNTD = tf.contrib.layers.real_valued_column('MWNTD')
f_MWTRV = tf.contrib.layers.real_valued_column('MWTRV')
f_AVBLS = tf.contrib.layers.real_valued_column('AVBLS')
f_BLCHS = tf.contrib.layers.real_valued_column('BLCHS')
f_ATRCT = tf.contrib.layers.real_valued_column('ATRCT')
f_MIREV = tf.contrib.layers.real_valued_column('MIREV')
f_HRATE = tf.contrib.layers.real_valued_column('HRATE')
f_CPTRA = tf.contrib.layers.real_valued_column('CPTRA')
f_CPTRV = tf.contrib.layers.real_valued_column('CPTRV')
f_TRVOU = tf.contrib.layers.real_valued_column('TRVOU')
f_TOUTV = tf.contrib.layers.real_valued_column('TOUTV')
f_ETRVU = tf.contrib.layers.real_valued_column('ETRVU')
f_ETRAV = tf.contrib.layers.real_valued_column('ETRAV')
f_NTRBL = tf.contrib.layers.real_valued_column('NTRBL')
f_NADDU = tf.contrib.layers.real_valued_column('NADDU')
f_NTREP = tf.contrib.layers.real_valued_column('NTREP')
f_NTRAT = tf.contrib.layers.real_valued_column('NTRAT')
f_NTRAN = tf.contrib.layers.real_valued_column('NTRAN')

## extracted_data = tf.contrib.learn.extract_pandas_data(data.iloc[:,1:])
feature_labels = ['DIFF', 'TRFEE'
, 'MKTCP', 'TOTBC', 'MWNUS', 'MWNTD', 'MWTRV', 'AVBLS', 'BLCHS', 'ATRCT', 'MIREV',
                  'HRATE', 'CPTRA', 'CPTRV', 'TRVOU', 'TOUTV', 'ETRVU', 'ETRAV', 'NTRBL', 'NADDU', 'NTREP', 'NTRAT', 'NTRAN']
target_label = ['MKPRU']
# building input function


def input_fn_train():
    # returns x, y
    features = {k: tf.constant(data[k].values, shape=[data[k].size, 1]) for k in feature_labels}
    target = tf.constant(data[target_label].values)
    return features, target
# building eval function


def input_fn_eval():
    pass
# building models

estimator = tf.contrib.learn.DNNRegressor(
    feature_columns=[f_DIFF, f_TRFEE,
        f_MKTCP, f_TOTBC, f_MWNUS, f_MWNTD, f_MWTRV, f_AVBLS, f_BLCHS, f_ATRCT, f_MIREV,
                     f_HRATE, f_CPTRA, f_CPTRV, f_TRVOU, f_TOUTV, f_ETRVU, f_ETRAV, f_NTRBL, f_NADDU, f_NTREP, f_NTRAT, f_NTRAN],
    hidden_units=[200,25],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.001,
        l1_regularization_strength=0.001
    ))

# train model
estimator.fit(input_fn=input_fn_train)
estimator.evaluate(input_fun=input_fn_eval)

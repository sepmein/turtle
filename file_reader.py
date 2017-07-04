"""
Read data from csv
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

## read data from pre-downloaded csv file
RAW_DATA = pd.read_csv('cleaned_data.csv')
## extracted_data = tf.contrib.learn.extract_pandas_data(data.iloc[:,1:])
## define raw labels, from which function will generate feature labels
RAW_LABELS = [
    'DIFF', 'TRFEE']
#, 'MKTCP', 'TOTBC', 'MWNUS', 'MWNTD', 'MWTRV', 'AVBLS',
#    'BLCHS', 'ATRCT', 'MIREV', 'HRATE', 'CPTRA', 'CPTRV', 'TRVOU', 'TOUTV',
#    'ETRVU', 'ETRAV', 'NTRBL', 'NADDU', 'NTREP', 'NTRAT', 'NTRAN'
#]
## Define how much rows should be skipped
## Because at the initial year of bitcoin, there weren't any $-BTC data.
## So it should be skipped
STARTS_AT = 400
## convert date string to pandas datetime format
RAW_DATA['Date'] = pd.to_datetime(RAW_DATA['Date'])


def gen_days_back(data, labels, days, starts_at):
    """generate data by days back
    """
    gen_labels = []
    gen_data = []
    for label in labels:
        for j in range(days):
            gen_labels.append(label + '_' + str(j + 1))
    for k in range(starts_at, data.shape[0]):
        days_back_data = data[k - days:k]
        selected_day_back_data = days_back_data.loc[:, 'DIFF':'TRFEE']
        selected_day_back_data_np = selected_day_back_data.values
        reshaped = np.reshape(selected_day_back_data_np.T,
                              (1, len(labels) * days))
        gen_data.append(reshaped)
    stacked = np.vstack(row for row in gen_data)
    dataframe = pd.DataFrame(data=stacked, columns=gen_labels)
    return dataframe, gen_labels

## Building the data by calling the gen_days_back function
GEN_DATA, GEN_FEATURE_LABELS = gen_days_back(RAW_DATA, RAW_LABELS, 1, 400)

# build input layers
FEATURES = []
for label in GEN_FEATURE_LABELS:
    FEATURES.append(tf.contrib.layers.real_valued_column(label))

TARGET_LABEL = ['MKPRU']

# building input function


def input_fn_train():
    """
        Build Input function
        Using tf.constant to build feature and target value
    """
    # returns x, y
    features_tf = {
        k: tf.constant(GEN_DATA[k].values)
        for k in GEN_FEATURE_LABELS
    }
    target = tf.constant(
        RAW_DATA[STARTS_AT:][TARGET_LABEL].values,
        #preprocessing.normalize(RAW_DATA[STARTS_AT:][TARGET_LABEL].values),
        shape=[RAW_DATA[STARTS_AT:][TARGET_LABEL].size, 1])
    return features_tf, target


# building eval function


def input_fn_eval():
    """
        input_fn_eval, a function to build eval function
    """
    pass


# Building models
# tf dnn regressor is used
# TODO: consider RNN model?
ESTIMATOR = tf.contrib.learn.DNNRegressor(
    feature_columns=FEATURES,
    hidden_units=[256, 128, 64]
    #    ,optimizer=tf.train.ProximalAdagradOptimizer(
    #        learning_rate=0.001, l1_regularization_strength=0.001)
)

# train model
ESTIMATOR.fit(input_fn=input_fn_train)
ESTIMATOR.evaluate(input_fun=input_fn_eval)

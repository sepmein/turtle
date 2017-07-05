"""
Read data from csv
"""
# TODO: inverse-scale data, get the params of the scaler
# TODO: split data in to three parts: training, validation and test
# TODO: improve logging and tensorboard

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# Setting logging verbosity
tf.logging.set_verbosity(tf.logging.INFO)

# read data from csv file
RAW_DATA_TRAIN = pd.read_csv('raw_data_train.csv')
RAW_DATA_CROSS_VALIDATION = pd.read_csv('raw_data_cross_validation.csv')
RAW_DATA_TEST = pd.read_csv('raw_data_test.csv')
# extracted_data = tf.contrib.learn.extract_pandas_data(data.iloc[:,1:])

# Define how much rows should be skipped
# Because at the initial year of bitcoin, there weren't any $-BTC data.
# So it should be skipped
STARTS_AT = 50

# Convert date string to pandas datetime format
RAW_DATA_TRAIN['Date'] = pd.to_datetime(RAW_DATA_TRAIN['Date'])
RAW_DATA_CROSS_VALIDATION['Date'] = pd.to_datetime(
    RAW_DATA_CROSS_VALIDATION['Date'])
RAW_DATA_TEST['Date'] = pd.to_datetime(RAW_DATA_TEST['Date'])

# Define target label
TARGET_LABEL = ['MKPRU']

# Define raw labels, from which function will generate feature labels
RAW_LABELS = [
    'DIFF', 'TRFEE', 'MKTCP', 'TOTBC', 'MWNUS', 'MWNTD', 'MWTRV', 'AVBLS',
    'BLCHS', 'ATRCT', 'MIREV', 'HRATE', 'CPTRA', 'CPTRV', 'TRVOU', 'TOUTV',
    'ETRVU', 'ETRAV', 'NTRBL', 'NADDU', 'NTREP', 'NTRAT', 'NTRAN'
]


def gen_days_back(data, labels, days, starts_at):
    """
        generate data by days back
    """
    gen_labels = []
    gen_data = []
    for label in labels:
        for j in range(days):
            gen_labels.append(label + '_' + str(j + 1))
    for k in range(starts_at, data.shape[0]):
        days_back_data = data[k - days:k]
        selected_day_back_data = days_back_data.loc[:, 'DIFF':'NTRAN']
        selected_day_back_data_np = selected_day_back_data.values
        reshaped = np.reshape(selected_day_back_data_np.T,
                              (1, len(labels) * days))
        gen_data.append(reshaped)
    stacked = np.vstack(row for row in gen_data)
    dataframe = pd.DataFrame(data=stacked, columns=gen_labels)
    return dataframe, gen_labels


# Building the data by calling the gen_days_back function
GEN_DATA_TRAIN, GEN_FEATURE_LABELS = gen_days_back(RAW_DATA_TRAIN, RAW_LABELS,
                                                   10, STARTS_AT)
GEN_DATA_CROSS_VALIDATION, GEN_FEATURE_LABELS_CV = gen_days_back(
    RAW_DATA_CROSS_VALIDATION, RAW_LABELS, 10, STARTS_AT)


# Generate target data
def gen_target_data(data, label, starts_at):
    """
        generate raw target data
    """
    result = data[starts_at:][label].values
    return result


TARGET_DATA_TRAIN = gen_target_data(
    data=RAW_DATA_TRAIN, label=TARGET_LABEL, starts_at=STARTS_AT)
TARGET_DATA_CROSS_VALIDATION = gen_target_data(
    data=RAW_DATA_CROSS_VALIDATION, label=TARGET_LABEL, starts_at=STARTS_AT)

# build input layers
FEATURES = []
for gen_feature_label in GEN_FEATURE_LABELS:
    FEATURES.append(tf.contrib.layers.real_valued_column(gen_feature_label))

# Data scaler
SCALER = preprocessing.StandardScaler()
FEATURE_SCALER = SCALER.fit(GEN_DATA_TRAIN)
# SCALED_GEN_DATA = FEATURE_SCALER.fit_transform(GEN_DATA_TRAIN)
TARGET_SCALER = SCALER.fit(TARGET_DATA_TRAIN)

# SCALED_TARGET_DATA = TARGET_SCALER.fit_transform(TARGET_DATA_TRAIN)

# Get numpy array from pandas dataframe
# SCALED_GEN_DATA_NP_ARRAY = SCALED_GEN_DATA.values
# SCALED_TARGET_DATA_NP_ARRAY = SCALED_TARGET_DATA.values

# building input function
def input_fn(feature, target):
    """
        Build Input function
        Using tf.constant to build feature and target value
    """
    # returns x, y
    features_tf = {
        k: tf.constant(FEATURE_SCALER.fit_transform(feature[k]))
        for k in GEN_FEATURE_LABELS
    }
    target_tf = tf.constant(TARGET_SCALER.fit_transform(target))
    # shape=[RAW_DATA_TRAIN[STARTS_AT:][TARGET_LABEL].size, 1])
    return features_tf, target_tf


def input_fn_train():
    """
        Build Input function
        Using tf.constant to build feature and target value
    """
    # returns x, y
    features_tf = {
        k: tf.constant(FEATURE_SCALER.fit_transform(GEN_DATA_TRAIN[k]))
        for k in GEN_FEATURE_LABELS
    }
    target_tf = tf.constant(TARGET_SCALER.fit_transform(TARGET_DATA_TRAIN))
    # shape=[RAW_DATA_TRAIN[STARTS_AT:][TARGET_LABEL].size, 1])
    return features_tf, target_tf


# building eval function
def input_fn_eval():
    """
        input_fn_eval, a function to build eval function
    """
    features_tf = {
        k: tf.constant(
            FEATURE_SCALER.fit_transform(GEN_DATA_CROSS_VALIDATION[k]))
        for k in GEN_FEATURE_LABELS
    }
    target_tf = tf.constant(
        TARGET_SCALER.fit_transform(TARGET_DATA_CROSS_VALIDATION))
    # shape=[RAW_DATA_TRAIN[STARTS_AT:][TARGET_LABEL].size, 1])
    return features_tf, target_tf


# Added a layer of validation monitor
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=input_fn_eval, every_n_steps=100, eval_steps=1)

# Building models
# tf dnn regressor is used
# TODO: consider RNN model?
ESTIMATOR = tf.contrib.learn.DNNRegressor(
    feature_columns=FEATURES,
    hidden_units=[256, 128, 64],
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30),
    model_dir='~/Desktop/turtle_model_17_07_06',
    dropout=0.01
    #optimizer=tf.train.ProximalAdagradOptimizer(
    #    learning_rate=0.05, l1_regularization_strength=0.1)
    )

# Fit train model
ESTIMATOR.fit(
    input_fn=input_fn_train, monitors=[validation_monitor], steps=100000)

# Cross validate data
ESTIMATOR.evaluate(input_fn=input_fn_eval, steps=1)

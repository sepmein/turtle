"""
    Fully Connectted DNN regressor
"""
# TODO: inverse-scale data, get the params of the scaler
# TODO: split data in to three parts: training, validation and test
# TODO: improve logging and tensorboard
# TODO: universal input fn
# TODO: predict function
# TODO: define fn to import trained model
# FIXME: Generation data error, should not generate data after random selection

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# Setting logging verbosity
tf.logging.set_verbosity(tf.logging.INFO)

# Define target label
TARGET_LABEL = ['MKPRU']

# Define raw labels, from which function will generate feature labels
RAW_LABELS = [
    'DIFF', 'TRFEE', 'MKTCP', 'TOTBC', 'MWNUS', 'MWNTD', 'MWTRV', 'AVBLS',
    'BLCHS', 'ATRCT', 'MIREV', 'HRATE', 'CPTRA', 'CPTRV', 'TRVOU', 'TOUTV',
    'ETRVU', 'ETRAV', 'NTRBL', 'NADDU', 'NTREP', 'NTRAT', 'NTRAN'
]


def gen_feature_labels(labels, days):
    """
        generate data by days back
        using "days" back data to predict BTC_USD value
    """
    gen_labels = []
    for label in labels:
        for j in range(days):
            gen_labels.append(label + '_' + str(j + 1))
    return gen_labels

# Generate feature labels
GEN_FEATURE_LABELS = gen_feature_labels(RAW_LABELS, 50)

# read data from csv file
gen_feature_data_training = pd.read_csv(
        'gen_feature_data_training.csv').loc[:, 'DIFF_1':]
gen_feature_data_cv = pd.read_csv(
        'gen_feature_data_cv.csv').loc[:, 'DIFF_1':]
gen_feature_data_test = pd.read_csv(
        'gen_feature_data_test.csv').loc[:, 'DIFF_1':]
gen_target_data_training = pd.read_csv(
        'gen_target_data_training.csv').loc[:, 'MKPRU']
gen_target_data_cv = pd.read_csv(
        'gen_target_data_cv.csv').loc[:, 'MKPRU']
gen_target_data_test = pd.read_csv(
        'gen_target_data_test.csv').loc[:, 'MKPRU']

# build tensorflow input layers
FEATURES = []
for gen_feature_label in GEN_FEATURE_LABELS:
    FEATURES.append(tf.contrib.layers.real_valued_column(gen_feature_label))

# Data scaler
SCALER = preprocessing.StandardScaler()
FEATURE_SCALER = SCALER.fit(gen_feature_data_training)
TARGET_SCALER = SCALER.fit(gen_target_data_training)


def input_fn(feature, target):
    """
        Build Input function
        Using tf.constant to build feature and target value
    """
    features_tf = {
        k: tf.constant(
                FEATURE_SCALER.fit_transform(feature[k]),
                shape=[feature[k].size, 1])
        for k in GEN_FEATURE_LABELS
    }
    target_tf = tf.constant(TARGET_SCALER.fit_transform(target))
    return features_tf, target_tf


def input_fn_train():
    """
        Build Input function
        Using tf.constant to build feature and target value
    """
    features_tf, target_tf = input_fn(
        feature=gen_feature_data_training,
        target=gen_target_data_training
    )
    return features_tf, target_tf


def input_fn_eval():
    """
        input_fn_eval, a function to build eval function
    """
    features_tf, target_tf = input_fn(
        feature=gen_feature_data_cv,
        target=gen_target_data_cv
    )
    return features_tf, target_tf


def input_fn_test():
    """
        input_fn_test, a function to build test function
    """
    features_tf, target_tf = input_fn(
        feature=gen_feature_data_test,
        target=gen_target_data_test
    )
    return features_tf, target_tf


def input_fn_predict():
    """
        Function for prediction
    """
    features_tf, target_tf = input_fn(
            feature=gen_feature_data_test,
            target=gen_target_data_test
            )
    return features_tf


# Added a layer of validation monitor
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=input_fn_eval,
    every_n_steps=100,
    eval_steps=1
)

# Building models
# tf dnn regressor is used
# TODO: consider RNN model?
ESTIMATOR = tf.contrib.learn.DNNRegressor(
    feature_columns=FEATURES,
    hidden_units=[512, 256, 128],
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60),
    model_dir="c:\\Users\\Spencer\\Desktop\\turtle-model-50",
    dropout=0.01
    # optimizer=tf.train.ProximalAdagradOptimizer(
    #    learning_rate=0.05, l1_regularization_strength=0.1)
)

# Fit train model
ESTIMATOR.fit(
    input_fn=input_fn_train,
    monitors=[validation_monitor],
    steps=100000
)

# Cross validate data
ESTIMATOR.evaluate(
    input_fn=input_fn_eval,
    steps=1
)

# Test and predict
ESTIMATOR.evaluate(
    input_fn=input_fn_test,
    steps=1
)

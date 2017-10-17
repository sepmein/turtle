"""
    Fully Connected DNN regressor
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
from sklearn import preprocessing

# Setting logging verbosity
tf.logging.set_verbosity(tf.logging.INFO)


# build tensorflow input layers
FEATURES = []
for gen_feature_label in GEN_FEATURE_LABELS:
    FEATURES.append(tf.contrib.layers.real_valued_column(gen_feature_label))

# Data scaler
SCALER = preprocessing.StandardScaler()
FEATURE_SCALER = SCALER.fit(gen_feature_data_training)
TARGET_SCALER = SCALER.fit(gen_target_data_training.values.reshape(-1, 1))


def input_fn(feature, target):
    """
        Build Input function
        Using tf.constant to build feature and target value
    """
    features_tf = {
        k: tf.constant(
            FEATURE_SCALER.fit_transform(feature[k].values.reshape(-1, 1)),
            shape=[feature[k].size, 1])
        for k in GEN_FEATURE_LABELS
    }
    target_tf = tf.constant(TARGET_SCALER.fit_transform(target.values.reshape(-1, 1)))
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
    every_n_steps=200,
    eval_steps=1
)

# Building models
# tf dnn regressor is used
# TODO: consider RNN model?
regressor = tf.estimator.DNNRegressor(
    feature_columns=FEATURES,
    hidden_units=[64, 64, 64, 32, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2],
    model_dir="D:\\OneDrive\\models\\171010",
    optimizer=tf.train.AdamOptimizer()    # optimizer=tf.train.ProximalAdagradOptimizer(
    #    learning_rate=0.05, l1_regularization_strength=0.1)
)

# Fit train model
regressor.train(
    input_fn=input_fn_train,
    steps=10000
)

# # Cross validate data
regressor.evaluate(
    input_fn=input_fn_eval,
    steps=1
)

#
# # Test and predict
# regressor.evaluate(
#     input_fn=input_fn_test,
#     steps=1
# )

# predictions_raw = list(regressor.predict(input_fn=input_fn_test))
#
# predictions = TARGET_SCALER.inverse_transform(predictions_raw)
# print(predictions)
#
# predictions_data_frame = pd.DataFrame(predictions)
# predictions.to_csv('predictions.csv')

def get_estimator():
    return regressor


def get_scaler():
    return FEATURE_SCALER, TARGET_SCALER

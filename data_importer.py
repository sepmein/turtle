import pandas as pd
import tensorflow as tf

# Define target label
target_label = ['MKPRU']

# Define raw labels, from which function will generate feature labels
feature_labels = ["DIFF", "TRFEE", "MKTCP", "TOTBC", "MWNUS",
                  "MWNTD", "MWTRV", "AVBLS", "BLCHS", "ATRCT",
                  "MIREV", "HRATE", "CPTRA", "CPTRV", "TRVOU",
                  "TOUTV", "ETRVU", "ETRAV", "NTRBL", "NADDU",
                  "NTREP", "NTRAT", "NTRAN"]

# how many days back
days = 50

feature_length = len(feature_labels) * days


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
GEN_FEATURE_LABELS = gen_feature_labels(feature_labels, days)

# read data from csv file
gen_feature_data_training = pd.read_csv(
    './data/generated/gen_feature_data_training.csv').loc[:, 'DIFF_1':]
gen_feature_data_cv = pd.read_csv(
    './data/generated/gen_feature_data_cv.csv').loc[:, 'DIFF_1':]
gen_feature_data_test = pd.read_csv(
    './data/generated/gen_feature_data_test.csv').loc[:, 'DIFF_1':]
gen_target_data_training = pd.read_csv(
    './data/generated/gen_target_data_training.csv').loc[:, 'MKPRU']
gen_target_data_cv = pd.read_csv(
    './data/generated/gen_target_data_cv.csv').loc[:, 'MKPRU']
gen_target_data_test = pd.read_csv(
    './data/generated/gen_target_data_test.csv').loc[:, 'MKPRU']

training_data_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=gen_feature_data_training,
    y=gen_target_data_training,
    batch_size=128,
    shuffle=True
)

cross_validation_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=gen_feature_data_cv,
    y=gen_target_data_cv,
    batch_size=128,
    shuffle=True
)

test_data_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=gen_feature_data_test,
    y=gen_target_data_test,
    batch_size=128,
    shuffle=True
)

from sklearn import preprocessing

SCALER = preprocessing.StandardScaler()
FEATURE_SCALER = SCALER.fit(gen_feature_data_training)
TARGET_SCALER = SCALER.fit(gen_target_data_training.values.reshape(-1, 1))

gen_feature_data_training = FEATURE_SCALER.fit_transform(gen_feature_data_training)
gen_target_data_training = gen_target_data_training.values.reshape(-1, 1)

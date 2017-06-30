"""
Read data from csv
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

rawdata = pd.read_csv('cleaned_data.csv')
## extracted_data = tf.contrib.learn.extract_pandas_data(data.iloc[:,1:])
raw_labels = [
    'DIFF', 'TRFEE', 'MKTCP', 'TOTBC', 'MWNUS', 'MWNTD', 'MWTRV', 'AVBLS',
    'BLCHS', 'ATRCT', 'MIREV', 'HRATE', 'CPTRA', 'CPTRV', 'TRVOU', 'TOUTV',
    'ETRVU', 'ETRAV', 'NTRBL', 'NADDU', 'NTREP', 'NTRAT', 'NTRAN'
]
STARTS_AT = 400
## convert date string to pandas datetime format
rawdata['Date'] = pd.to_datetime(rawdata['Date'])
def gen_days_back(data, days, starts_at):
    """generate data by days back
    """
    rows = data.shape[0]
    gen_labels = []
    gen_data = []
    for i in range(len(raw_labels)):
        for j in range(days):
           gen_labels.append(raw_labels[i] + '_' + str(j + 1)) 
    for i in range(starts_at, rawdata.shape[0]):
        days_back_data = rawdata[i-days: i]
        selected_day_back_data = days_back_data.loc[:, 'DIFF':'NTRAN']
        selected_day_back_data_np = selected_day_back_data.values
        reshaped = np.reshape(selected_day_back_data_np.T, (1,len(raw_labels) * days))
        gen_data.append(reshaped)
    stacked = np.vstack(row for row in gen_data)
    dataframe = pd.DataFrame(data=stacked,
                            columns=gen_labels)
    return dataframe, gen_labels

data, feature_labels = gen_days_back(rawdata, 10, STARTS_AT)

# build input layers
features = []
for i in range(len(feature_labels)):
    features.append(tf.contrib.layers.real_valued_column(feature_labels))

target_label = ['MKPRU']

# building input function


def input_fn_train():
    # returns x, y
    f = {
        k: tf.constant(
            preprocessing.normalize(data[k].values), shape=[data[k].size, 1])
        for k in feature_labels
    }
    target = tf.constant(
        preprocessing.normalize(rawdata[STARTS_AT:][target_label].values),
        shape=[rawdata[STARTS_AT:][target_label].size, 1])
    return f, target


# building eval function


def input_fn_eval():
    pass


# building models

estimator = tf.contrib.learn.DNNRegressor(
    feature_columns=features,
    hidden_units=[256, 128, 64]
#    ,optimizer=tf.train.ProximalAdagradOptimizer(
#        learning_rate=0.001, l1_regularization_strength=0.001)
)

# train model
estimator.fit(input_fn=input_fn_train)
estimator.evaluate(input_fun=input_fn_eval)

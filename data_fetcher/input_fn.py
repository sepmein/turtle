import pandas as pd
import tensorflow as tf

from data_fetcher.preprocess import transform

# read data from generated csv file
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

# input fns using tf.pandas input, not using currently
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

# scale feature data
gen_feature_data_training = transform(gen_feature_data_training)
gen_feature_data_cv = transform(gen_feature_data_cv)
gen_feature_data_test = transform(gen_feature_data_test)

# reshape target data to fit tensor shape
gen_target_data_training = gen_target_data_training.values.reshape(-1, 1)
gen_target_data_cv = gen_target_data_cv.values.reshape(-1, 1)
gen_target_data_test = gen_target_data_test.values.reshape(-1, 1)

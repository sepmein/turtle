import shutil
from os.path import isdir

import numpy as np
import pandas as pd
import tensorflow as tf
from oyou import Model
from twone import RNNContainer as Container

from turtle.config import feature_labels, target_label

if isdir('./log'):
    shutil.rmtree('./log')
if isdir('./model'):
    shutil.rmtree('./model')
####################################################################
# scrap data
####################################################################
# df = scrap_all()
# df.to_csv('raw.csv')
fetched_raw_df = pd.read_csv('raw.csv')
####################################################################
# process data using twone
####################################################################
batch_size = 1
time_steps = 100
container = Container(data_frame=fetched_raw_df,
                      training_set_split_ratio=0.8,
                      cross_validation_set_split_ratio=0.15,
                      test_set_split_ratio=0.05)
container.set_feature_tags(feature_tags=feature_labels) \
    .set_target_tags(target_tags=target_label, shift=-1) \
    .interpolate()

container.data[container.target_tags] = np.where(
    (container.data['MKPRU'] < container.data['MKPRU_target']),
    0, 1).reshape(-1, 1)

container.gen_batch(batch=batch_size,
                    time_steps=time_steps,
                    random_batch=False,
                    shuffle=False)
num_features = container.num_features
num_targets = container.num_targets

#####################################################################
# build tensorflow graph
#####################################################################
state_size = 500
num_classes = 2
features = tf.placeholder(dtype=tf.float32,
                          shape=[batch_size, time_steps, num_features],
                          name='features')
targets = tf.placeholder(dtype=tf.int32,
                         shape=[batch_size, time_steps, num_targets],
                         name='targets')
# one_hot_target_labels = tf.one_hot(indices=targets,
#                                    depth=num_classes)
target_labels_reshaped = tf.reshape(targets, shape=[batch_size, time_steps, num_targets])
cell = tf.contrib.rnn.GRUCell(state_size)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                             inputs=features,
                                             dtype=tf.float32)
with tf.name_scope('softmax'):
    w = tf.get_variable(name='softmax_w',
                        dtype=tf.float32,
                        shape=[state_size, num_classes])
    b = tf.get_variable(name='softmax_b',
                        dtype=tf.float32,
                        shape=[num_classes])
    outputs_reshaped_for_matmul = tf.reshape(tensor=rnn_outputs,
                                             shape=[-1, state_size])
    predictions = tf.matmul(outputs_reshaped_for_matmul, w) + b
    predictions_reshaped_for_softmax = tf.reshape(tensor=predictions,
                                                  shape=[batch_size, time_steps, num_classes])
    logits = tf.nn.softmax(predictions_reshaped_for_softmax)
    losses = tf.losses.sparse_softmax_cross_entropy(labels=target_labels_reshaped,
                                                    logits=logits)
#####################################################################
# continue with oyou
#####################################################################

# build model
model = Model(name='turtle')
model.features = features
model.targets = targets
model.prediction = predictions
model.losses = losses
# logs
model.create_log_group(name='training',
                       feed_tensors=[features, targets])
model.create_log_group(name='cv',
                       feed_tensors=[features, targets])
model.log_scalar(name='training_loss',
                 tensor=losses,
                 group='training')
model.log_scalar(name='cross_validation_loss',
                 tensor=losses,
                 group='cv')
model.log_histogram(name='softmax_w',
                    tensor=w,
                    group='training')
model.log_histogram(name='softmax_b',
                    tensor=b,
                    group='training')
# savings
model.define_saving_strategy(indicator_tensor=losses,
                             interval=50,
                             feed_dict=[features, targets])
# train
model.train(features=container.get_training_features,
            targets=container.get_training_targets,
            training_steps=1000000,
            training_features=container.get_training_features,
            training_targets=container.get_training_targets,
            cv_features=container.get_cv_features,
            cv_targets=container.get_cv_targets,
            saving_features=container.get_cv_features,
            saving_targets=container.get_cv_targets
            )

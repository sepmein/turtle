import tensorflow as tf
from twone import RNNContainer as Container

from config import feature_labels, target_label
from data_fetcher import scrap

####################################################################
# scrap data
####################################################################
fetched_raw_df = scrap()

####################################################################
# process data using twone
####################################################################
container = Container(data_frame=fetched_raw_df)
# todo: change interpolate api to the most recent version
container.interpolate() \
    .normalize() \
    .set_feature_tags(feature_tags=feature_labels) \
    .set_target_tags(target_tags=target_label)
container.gen_batch(
    batch=5,
    time_steps=20
)
num_features = container._num_features
num_targets = container._num_targets

#####################################################################
# build tensorflow graph
#####################################################################
features = tf.placeholder(dtype=tf.float32,
                          shape=[None, None, num_features],
                          name='features')
targets = tf.placeholder(dtype=tf.float32,
                         shape=[None, None, num_targets],
                         name='targets')
cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                             inputs=features)
#####################################################################
# continue with oyou
#####################################################################

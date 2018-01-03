import tensorflow as tf
from oyou import Model
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
batch_size = 5
time_steps = 20
container = Container(data_frame=fetched_raw_df)
# todo: change interpolate api to the most recent version
container.interpolate() \
    .normalize() \
    .set_feature_tags(feature_tags=feature_labels) \
    .set_target_tags(target_tags=target_label, shift=-1)
container.gen_batch(
    batch=batch_size,
    time_steps=time_steps
)
num_features = container._num_features
num_targets = container._num_targets

#####################################################################
# build tensorflow graph
#####################################################################
state_size = 10
num_classes = 2
features = tf.placeholder(dtype=tf.float32,
                          shape=[None, None, num_features],
                          name='features')
targets = tf.placeholder(dtype=tf.float32,
                         shape=[None, None, num_targets],
                         name='targets')
one_hot_target_labels = tf.one_hot(indices=targets,
                                   depth=2)
cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                             inputs=features)
with tf.name_scope('softmax'):
    w = tf.get_variable(name='softmax_w',
                        dtype=tf.float32,
                        shape=[state_size, num_classes])
    b = tf.get_variable(name='softmax_b',
                        dtype=tf.float32,
                        shape=[num_classes])
    outputs_reshaped_for_matmul = tf.reshape(tensor=rnn_outputs,
                                             shape=[-1, state_size])
    predictions = tf.matmul(w, outputs_reshaped_for_matmul) + b
    predictions_reshaped_for_softmax = tf.reshape(tensor=predictions,
                                                  shape=[batch_size, time_steps, num_classes])
    logits = tf.nn.softmax(predictions_reshaped_for_softmax)
    losses = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_target_labels,
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
# train
model.train()

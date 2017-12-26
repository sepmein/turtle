# hidden_size = 100
# batch_size = 64
# feature_size =
#
# rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
# initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
#
# input_data = tf.placeholder(shape=(batch_size, max_length, feature_size), dtype=tf.float32)
#
# outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data, initial_state=initial_state, dtype=tf.float32)
#
# loss = tf.reduce_mean(tf.abs(outputs - y))
#
# optimizer = tf.train.AdamOptimizer().minimize(loss)
#
# session = tf.Session()
# session.run(tf.global_variables_initializer())
#
# for i in range(100000):
#     l, _ = session.run(
#         [loss, optimizer],
#         feed_dict={
#             input_data: x
#         }
#     )
#     if i % 100 == 0:
#         print(l)
#
from twone import Container

from config import feature_labels, target_label
from data_fetcher import scrap

fetched_raw_df = scrap()

container = Container(data_frame=fetched_raw_df)
# todo: change interpolate api to the most recent version
container.interpolate(container.data)
container.set_feature_tags(feature_tags=feature_labels)
container.set_target_tags(target_tags=target_label)

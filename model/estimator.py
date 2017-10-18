import tensorflow as tf

from config import logdir
from data_fetcher.input_fn import gen_feature_data_cv

model_dir = logdir + '\\model830'


# with tf.Session() as session:
#     results = session.run(
#         h_3,
#         {
#             x: gen_feature_data_cv
#         }
#     )
#
#     print(results)
#
#     mar = tf.reduce_mean(tf.abs(h_3 - y))
#     print(session.run(mar, {
#         x: gen_feature_data_cv,
#         y: gen_target_data_cv
#     }))
#
#     print(session.run(mar, {
#         x: gen_feature_data_training,
#         y: gen_target_data_training
#     }))


def load_model():
    session = tf.Session()
    loader = tf.saved_model.loader.load(session, ['turtle'], model_dir)
    graph = tf.get_default_graph()
    predictions = graph.get_operation_by_name('forward_propagation/prediction')
    inputs = graph.get_collection('inputs')
    x = inputs[0]
    y = inputs[1]
    predicted_results = session.run(
        predictions,
        {x: gen_feature_data_cv}
    )

    # TODO get rid of this part
    w_1 = graph.get_tensor_by_name('w_1:0')
    b_1 = graph.get_tensor_by_name('b_1:0')
    h_1 = tf.matmul(x, w_1, transpose_b=True) + b_1
    a_1 = tf.nn.relu(h_1)
    w_2 = graph.get_tensor_by_name('w_2:0')
    b_2 = graph.get_tensor_by_name('b_2:0')
    h_2 = tf.matmul(a_1, w_2, transpose_b=True) + b_2
    a_2 = tf.nn.relu(h_2)
    w_3 = graph.get_tensor_by_name('w_3:0')
    b_3 = graph.get_tensor_by_name('b_3:0')
    predictions = tf.matmul(a_2, w_3, transpose_b=True) + b_3

    return session, predictions, x, y


def evaluate(feature, labels):
    session, predictions, x, y = load_model()
    mean_absolute_error = tf.reduce_mean(tf.abs(predictions - y))
    mae = session.run(mean_absolute_error, {
        x: feature,
        y: labels
    })
    session.close()
    return mae


def predict(feature):
    session, predictions, x, _ = load_model()
    results = session.run(predictions, {
        x: feature
    })
    session.close()
    return results

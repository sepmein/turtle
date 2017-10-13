import tensorflow as tf

from config import num_feature_labels, lambd, training_steps, logdir
# import data
from data_importer import gen_target_data_training, \
    gen_feature_data_training, gen_target_data_cv, \
    gen_feature_data_cv

# define model layers
weight_dict = [
    num_feature_labels, 128, 64, 1
]
num_layers = len(weight_dict) - 1

# Initialize weights using xavier initializer
with tf.name_scope('parameters_initialize'):
    weights, biases = ([], [])

    for i in range(len(weight_dict) - 1):
        w = tf.get_variable(
            name='w_' + str(i + 1),
            shape=[weight_dict[i + 1], weight_dict[i]],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )
        b = tf.zeros(
            name='b_' + str(i + 1),
            shape=[weight_dict[i + 1]],
            dtype=tf.float32
        )
        weights.append(w)
        biases.append(b)

# training data
with tf.name_scope('data_placeholder'):
    x = tf.placeholder(
        dtype=tf.float32,
        name='training_features',
        shape=[None, num_feature_labels]
    )
    y = tf.placeholder(
        dtype=tf.float32,
        name='training_targets',
        shape=[None, 1]
    )

with tf.name_scope('forward_propagation'):
    # Forward propagation
    # hypothesis layer 1
    h_1 = tf.matmul(x, weights[0], transpose_b=True) + biases[0]
    # activation layer 1
    a_1 = tf.nn.relu(
        features=h_1,
        name='activation_layer_1'
    )
    # hypothesis layer 2
    h_2 = tf.matmul(a_1, weights[1], transpose_b=True) + biases[1]
    # activation layer 2
    a_2 = tf.nn.relu(
        features=h_2,
        name='activation_layer_2'
    )
    # hypothesis layer 3
    h_3 = tf.matmul(a_2, weights[2], transpose_b=True) + biases[2]

    # L2 norm of w's
    l2_regularization = lambd * (
        tf.nn.l2_loss(weights[0]) +
        tf.nn.l2_loss(weights[1]) +
        tf.nn.l2_loss(weights[2])
    )

    # Loss function
    loss = tf.losses.mean_squared_error(
        labels=h_3,
        predictions=y
    ) + l2_regularization
    # tf.reduce_sum(tf.square(tf.abs(a_2 - y))) / (2 * m) + l2_regularization

    mean_relative_error = tf.metrics.mean_relative_error(
        labels=h_3,
        predictions=y,
        normalizer=y
    )

    mean_absolute_error = tf.metrics.mean_absolute_error(
        labels=h_3,
        predictions=y
    )

with tf.name_scope('summary'):
    # define summaries
    tf.summary.scalar(
        name='loss',
        tensor=loss
    )
    tf.summary.scalar(
        name='L2_norm',
        tensor=l2_regularization
    )
    # tf.summary.scalar(
    #     name='mean_relative_error_cv',
    #     tensor=mean_absolute_error
    # )
    # tf.summary.scalar(
    #     name='mean_relative_error_cv',
    #     tensor=mean_relative_error
    # )
    tf.summary.histogram(
        name='w_1',
        values=weights[0]
    )
    tf.summary.histogram(
        name='w_2',
        values=weights[1]
    )
    tf.summary.histogram(
        name='w_3',
        values=weights[2]
    )
    tf.summary.histogram(
        name='b_1',
        values=biases[0]
    )
    tf.summary.histogram(
        name='b_2',
        values=biases[1]
    )
    tf.summary.histogram(
        name='b_3',
        values=biases[2]
    )

    # merge all summary as a tensor op
    summaries = tf.summary.merge_all()

with tf.name_scope('models'):
    optimizer = tf.train.AdamOptimizer()
    optimization = optimizer.minimize(loss)

with tf.Session() as session:
    # initialize tf.variable
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    # add session.graph to summary writer
    # summary writer to disk
    summary_writer = tf.summary.FileWriter(
        logdir=logdir,
        graph=session.graph
    )

    # train model
    for _ in range(training_steps):
        summaries_results, optimization_results = session.run(
            [summaries, optimization],
            {
                x: gen_feature_data_training,
                y: gen_target_data_training
            }
        )

        summary_writer.add_summary(
            summary=summaries_results,
            global_step=_
        )

        if _ % 100 == 0:
            l = session.run(
                loss,
                {
                    x: gen_feature_data_training,
                    y: gen_target_data_training
                }
            )
            l_cv, (mre, inc), (mae, inc_2) = session.run(
                [loss, mean_relative_error, mean_absolute_error],
                {
                    x: gen_feature_data_cv,
                    y: gen_target_data_cv
                })
            norm = session.run(
                l2_regularization
            )
            print("Epoch:", '%04d' % (_ + 1), "cost=", "{:.9f}".format(l), "cv=", "{:.9f}".format(l_cv), "mre", mre,
                  "mae", mae)
            print("L2 regularization: ", norm)

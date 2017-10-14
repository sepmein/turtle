import tensorflow as tf

from config import num_feature_labels, lambd, training_steps, logdir, learning_rate
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

    # mean_relative_error = tf.metrics.mean_relative_error(
    #     labels=h_3,
    #     predictions=y,
    #     normalizer=y
    # )

    mean_absolute_error = tf.abs(tf.reduce_mean(h_3 - y))

    mean_relative_error = tf.abs(tf.reduce_mean((h_3 - y) / y))

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
    tf.summary.scalar(
        name='mean_absolute_error',
        tensor=mean_absolute_error
    )
    tf.summary.scalar(
        name='mean_relative_error',
        tensor=mean_relative_error
    )
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
        name='a_1',
        values=a_1
    )
    tf.summary.histogram(
        name='h_1',
        values=h_1
    )
    tf.summary.histogram(
        name='a_2',
        values=a_2
    )
    tf.summary.histogram(
        name='h_2',
        values=h_2
    )
    tf.summary.histogram(
        name='h_3',
        values=h_3
    )

    # merge all summary as a tensor op
    summaries = tf.summary.merge_all()

with tf.name_scope('models'):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate
    )
    optimization = optimizer.minimize(loss)

with tf.Session() as session:
    # initialize tf.variable
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    # add session.graph to summary writer
    # summary writer to disk
    train_summary_writer = tf.summary.FileWriter(
        logdir=logdir + '/train',
        graph=session.graph
    )

    cv_summary_writer = tf.summary.FileWriter(
        logdir=logdir + '/cv',
        graph=session.graph
    )
    # train model
    for _ in range(training_steps):
        optimization_results = session.run(
            optimization,
            {
                x: gen_feature_data_training,
                y: gen_target_data_training
            }
        )

        if _ % 10 == 0:
            summaries_results_train, l, mre_train, mae_train = session.run(
                [summaries, loss, mean_relative_error, mean_absolute_error],
                {
                    x: gen_feature_data_training,
                    y: gen_target_data_training
                }
            )
            summaries_results_cv, l_cv, mre_cv, mae_cv = session.run(
                [summaries, loss, mean_relative_error, mean_absolute_error],
                {
                    x: gen_feature_data_cv,
                    y: gen_target_data_cv
                })
            norm = session.run(
                l2_regularization
            )
            print("Epoch:", '%04d' % (_ + 1), "cost=", "{:.9f}".format(l), "cv=", "{:.9f}".format(l_cv))
            print(
                "mre_train", mre_train,
                "mae_train", mae_train,
                "mre_cv", mre_cv,
                "mae_cv", mae_cv
            )
            print("L2 regularization: ", norm)
            train_summary_writer.add_summary(
                summary=summaries_results_train,
                global_step=_
            )

            cv_summary_writer.add_summary(
                summary=summaries_results_cv,
                global_step=_
            )

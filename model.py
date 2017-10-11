import tensorflow as tf

# import data
from data_importer import gen_target_data_training, gen_feature_data_training, gen_target_data_cv, gen_feature_data_cv

# fake input_vector_shape
input_vector_shape = [1869, 1150]

# define model layers
weight_dict = [
    input_vector_shape[1], 128, 1
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
        shape=[None, 1150]
    )
    y = tf.placeholder(
        dtype=tf.float32,
        name='training_targets',
        shape=[None, 1]
    )

with tf.name_scope('forward_propagation'):
    # Forward propagation
    # hypothesis
    h_1 = tf.matmul(x, weights[0], transpose_b=True) + biases[0]
    # activation layer 1
    a_1 = tf.nn.relu(
        features=h_1,
        name='activation_layer_1'
    )

    h_2 = tf.matmul(a_1, weights[1], transpose_b=True) + biases[1]
    # activation layer 1
    a_2 = tf.nn.relu(
        features=h_2,
        name='activation_layer_2'
    )

    # loss function
    lambd = 0.1
    normalization = lambd * (
        tf.reduce_sum(
            tf.nn.l2_normalize(
                x=weights[0],
                dim=[0, 1]
            )
        ) +
        tf.reduce_sum(
            tf.nn.l2_normalize(
                x=weights[1],
                dim=[0, 1]
            )
        )

    )
    # Loss function
    m = x.get_shape().as_list()[0]
    loss = tf.losses.mean_squared_error(
        labels=a_2,
        predictions=y
    ) + normalization
    # tf.reduce_sum(tf.square(tf.abs(a_2 - y))) / (2 * m) + normalization

with tf.name_scope('models'):
    optimizer = tf.train.AdamOptimizer()
    optimization = optimizer.minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for _ in range(100000):
        session.run(
            optimization,
            {
                x: gen_feature_data_training,
                y: gen_target_data_training
            }
        )
        if _ % 100 == 0:
            l = session.run(
                loss,
                {
                    x: gen_feature_data_training,
                    y: gen_target_data_training
                }
            )
            l_cv = session.run(
                loss,
                {
                    x: gen_feature_data_cv,
                    y: gen_target_data_cv
                })
            print("Epoch:", '%04d' % (_ + 1), "cost=", "{:.9f}".format(l), "cv=", "{:.9f}".format(l_cv))
# Model


# debugger
# session = tf.Session()
# session.run(tf.global_variables_initializer())
# session.run([weights, biases])

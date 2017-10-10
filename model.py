import tensorflow as tf

# import data
from data_importer import training_data_input_fn

# fake input_vector_shape
input_vector_shape = 1000

# define model layers
weight_dict = [
    1024, 64, 32, 16, 8
]
num_layers = len(weight_dict) - 1

# Initialize weights using xavier initializer

with tf.name_scope('parameters_initialize'):
    weights, biases = ([], [])

    for i in range(len(weight_dict) - 1):
        w = tf.get_variable(
            name='w_' + str(i + 1),
            shape=[weight_dict[i + 1], weight_dict[i]],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.zeros(
            name='b_' + str(i + 1),
            shape=[weight_dict[i + 1]]
        )
        weights.append(w)
        biases.append(b)

# training data
with tf.name_scope('data_placeholder'):
    x_train = tf.placeholder(
        dtype=tf.float64,
        name='training_features'
    )
    y_train = tf.placeholder(
        dtype=tf.float64,
        name='training_targets'
    )

with tf.name_scope('forward_propagation'):
    # Forward propagation
    # hypothesis
    h_1 = tf.matmul(w[1], x_train) + b[1]
    # activation layer 1
    a_1 = tf.nn.relu(
        features=h_1,
        name='activation_layer_1'
    )
    # loss function
    loss = tf.losses.mean_squared_error(
        labels=x_train,
        predictions=y_train
    )
    # Loss function


# Model


# debugger
# session = tf.Session()
# session.run(tf.global_variables_initializer())
# session.run([weights, biases])

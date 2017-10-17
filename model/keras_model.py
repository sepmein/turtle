from keras import regularizers
from keras.layers import Dense, Activation
from keras.models import Sequential

from config import num_feature_labels
from data_fetcher.input_fn import gen_feature_data_training, gen_target_data_training, gen_feature_data_cv, \
    gen_target_data_cv

model = Sequential()

model.add(Dense(units=128, input_dim=num_feature_labels, kernel_regularizer=regularizers.l2(0.8)))
model.add(Activation('relu'))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.8)))
model.add(Activation('relu'))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.8)))
model.add(Activation('relu'))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.8)))
model.add(Activation('relu'))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.8)))
model.add(Activation('relu'))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.8)))
model.add(Activation('relu'))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.8)))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(
    loss='mean_squared_error', optimizer='adam'
)

model.fit(
    gen_feature_data_training,
    gen_target_data_training,
    epochs=100000,
    validation_data=(gen_feature_data_cv, gen_target_data_cv),
    shuffle=True
)

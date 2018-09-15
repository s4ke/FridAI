from boston.config import FEATURE_DIMENSIONALITY
from keras.engine import Model
from keras.layers import Input, Dense, Dropout, Reshape, Conv2D, Flatten

from boston import config


def get_model() -> Model:
    # here starts your task!
    # implement an ANN that solves the boston house prices task.
    # solving means we want to approximate an unknown function which,
    # given some features like house size, location etc. predicts it's price
    # prices are normalized to range [0;1]

    # there are some useful imports already, check the imports from config and the
    # different layers from keras.layers!
    input = Input(shape=(config.FEATURE_DIMENSIONALITY,))

    hidden = Dense(units=256, activation='sigmoid')(input)
    hidden = Reshape(target_shape=(16, 16, 1))(hidden)
    hidden = Conv2D(filters=16, kernel_size=(4, 4))(hidden)
    hidden = Flatten()(hidden)

    output = Dense(units=1, activation='linear')(hidden)

    # create a new model by specifying input/output layer(s)
    model = Model(inputs=[input], outputs=[output])
    # I already chose optimizer and loss function, you won't need to teak them (but you can of course!)
    model.compile(optimizer='sgd', loss='mse')
    return model

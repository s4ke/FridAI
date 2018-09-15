from cancer.config import NUM_CLASSES, FEATURE_DIMENSIONALITY
from keras.engine import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD

from cancer import config


def get_model() -> Model:
    # here starts your task!
    # implement an ANN that solves the cancer task
    # (solving means it is better than guessing, which is correct 50% of the time)
    # delete the following (single) line and replace it with your model
    input = Input(shape=(config.NUM_FEATURES,))

    # hidden = Dense(units=512, activation='sigmoid')(input)
    hidden = Dense(units=8, activation='sigmoid')(input)
    hidden = Dense(units=2, activation='sigmoid')(hidden)
    # hidden = Dense(units=16)(hidden)

    output = Dense(units=config.NUM_CLASSES, activation='softmax')(hidden)

    # create a new model by specifying input/output layer(s)
    model = Model(inputs=[input], outputs=[output])
    # I already chose optimizer and loss function, you won't need to tweak them (but you can of course!)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

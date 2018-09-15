from datetime import datetime
from os import path

# seed the random generator, for reproducible results
from numpy import random
random.seed(1337)

from cancer.config import NUM_CLASSES
from keras import utils, callbacks
from cancer.models import get_model
from sklearn import datasets

features, labels = datasets.load_breast_cancer(return_X_y=True)
labels = utils.to_categorical(labels, NUM_CLASSES)
features /= features.max(axis=0)

# this is your task, implement the method get_model() in the models.py file
model = get_model()

run_name = 'cancer-{:%d-%b_%H-%M-%S}'.format(datetime.now())
dir_path = path.dirname(path.realpath(__file__))
log_dir = path.join(dir_path, 'logs', run_name)
print('logging to "{}"'.format(log_dir))
tb_callback = callbacks.TensorBoard(log_dir=log_dir)
model.fit(features, labels, batch_size=256, epochs=2000, validation_split=.2, callbacks=[tb_callback])

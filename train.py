
from DataGenerator import import_data
from nn import create_model


import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = import_data()

    data = np.expand_dims(data, axis=-1)
    print(data.shape)
    model = create_model()
    batch_size = 16
    epochs = 100
    callbacks = [TensorBoard(), ModelCheckpoint('model.h5')]

    # model.load_weights('model.h5')

    model.fit(data, data,
              batch_size=batch_size, epochs=epochs,
              callbacks=callbacks, verbose=1)

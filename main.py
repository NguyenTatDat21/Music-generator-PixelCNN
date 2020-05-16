import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint

from DataGenerator import import_data
from nn import create_model

data = import_data()
data = np.expand_dims(data, axis=-1)
print(data.shape)
model = create_model()
batch_size = 32
epochs = 100
callbacks = [TensorBoard(), ModelCheckpoint('model.h5')]

model.fit(data, data,
          batch_size=batch_size, epochs=epochs,
          callbacks=callbacks)

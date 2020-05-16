
from midi_generator import play_img

from nn import create_model


import numpy as np
model = create_model()
model.load_weights("model.h5")

pixels = np.zeros((1, 1000, 48, 1))
_, rows, cols, channels = pixels.shape


# Iterate the pixels because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        # Feed the whole array and retrieving the pixel value probabilities for the next pixel.
        ps = None
        if row >= 48:
            input = pixels[:, row-48:row]
            ps = model.predict_on_batch(input)[:, 48, col, 0]
        else:
            input = pixels[:, :48]
            ps = model.predict_on_batch(input)[:, row, col, 0]

        # Use the probabilities to pick a pixel value.
        # Lastly, we normalize the value.
        pixels[0, row, col, 0] = np.random.choice(2, p=[1-ps[0], ps[0]])




play_img(pixels[0,:,:,0])
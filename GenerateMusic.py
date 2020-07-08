
from midi_generator import play_img

from nn import create_model

import matplotlib.pyplot as plt
import numpy as np
model = create_model()
model.load_weights("model.h5")

pixels = np.zeros((1, 1000, 96, 1))
_, rows, cols, channels = pixels.shape

# Iterate the pixels because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        # Feed the whole array and retrieving the pixel value probabilities for the next pixel.
        ps = None
        if row >= 48:
            ps = model.predict(pixels[:, row-47:row+1])[0, 47, col, 0]
        else:
            ps = model.predict(pixels[:, :48])[0, row, col, 0]

        # Use the probabilities to pick a pixel value.
        # Lastly, we normalize the value.

        pixels[0, row, col, 0] = np.random.choice(2, p=[1.0-ps, ps])

    print(row)

    if row % 10 == 0:
        print(round(row / rows * 100), '%')

plt.matshow(pixels[0,:,:,0])
plt.show()




play_img(pixels[0,:,:,0])
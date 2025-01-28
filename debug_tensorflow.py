import tensorflow as tf

print("Available GPUs:", tf.config.experimental.list_physical_devices('GPU'))
print("Keras version:", tf.keras.__version__)

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse')

import numpy as np
X = np.random.rand(100, 10)
y = np.random.rand(100, 1)

# Train model
model.fit(X, y, epochs=3)

import tensorflow as tf
import numpy as np
# Define the XOR input and output data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
# Define the DNN architecture
model = tf.keras.Sequential([
tf.keras.layers.Dense(8, input_dim=2, activation='relu'),
tf.keras.layers.Dense(8, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])
# Train the DNN
model.fit(x_data, y_data, epochs=1000, verbose=0)
# Test the trained DNN
predictions = model.predict(x_data)
rounded_predictions = np.round(predictions)
print("Predictions:", rounded_predictions)

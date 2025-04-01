# verification_script.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import flask
import PIL

print("=== Versiones ===")
print(f"TensorFlow: {tf.__version__}")
print(f"Flask: {flask.__version__}")
print(f"Pillow (PIL): {PIL.__version__}")

# Verifica si TensorFlow detecta GPU (opcional)
print("\n=== GPU Disponible ===")
print(tf.config.list_physical_devices('GPU'))

# Prueba mínima de TensorFlow
print("\n=== Prueba de Modelo ===")
model = tf.keras.Sequential([layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mse')
print("✔ Modelo compilado correctamente")

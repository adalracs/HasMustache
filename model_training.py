import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Configuración de generadores de datos
def create_generator(data_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='multi_output',  # Para multi-etiquetas
        classes=['mustache_male', 'mustache_female', 'no_mustache_male', 'no_mustache_female'],
        shuffle=True
    )

# Definir modelo (misma arquitectura que en app.py)
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Entrenar modelo
def train_model():
    train_generator = create_generator('data/train')
    val_generator = create_generator('data/validation')
    
    model = build_model()
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )
    model.save('models/modelo_inicial.h5')

# Reentrenar con nuevos datos
def retrain_model():
    train_generator = create_generator('new_data')  # Usa datos de feedback
    model = tf.keras.models.load_model('models/modelo_inicial.h5')
    model.fit(train_generator, epochs=5)
    model.save('models/modelo_actualizado.h5')

if __name__ == '__main__':
    train_model()  # Ejecutar solo la primera vez
    # retrain_model()  # Descomentar para reentrenar luego

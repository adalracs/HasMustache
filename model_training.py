import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
import os

# Configuración optimizada para CPU
def create_generator(data_dir, augment=False):
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,  # Reducido para ahorrar recursos
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    
    return datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),  # Reducción de tamaño para menos cómputo
        batch_size=16,           # Batch más pequeño para ahorrar memoria
        class_mode='multi_output',
        classes=['mustache_male', 'mustache_female', 'no_mustache_male', 'no_mustache_female'],
        shuffle=True
    )

# Modelo optimizado para CPU
def build_model():
    model = Sequential([
        Input(shape=(128, 128, 3)),  # Definición explícita para evitar warnings
        Conv2D(16, (3,3), activation='relu'),  # Menos filtros
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),  # Capa adicional pero con más reducción
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),  # Menos neuronas
        Dense(2, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)  # Tasa de aprendizaje definida explícitamente
    model.compile(optimizer=optimizer, 
                 loss='binary_crossentropy', 
                 metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])  # Métrica adicional útil
    
    return model

# Entrenamiento optimizado
def train_model():
    train_generator = create_generator('data/train', augment=True)
    val_generator = create_generator('data/validation')
    
    model = build_model()
    
    # Callbacks para mejor control
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // val_generator.batch_size),
        epochs=15,  # Más épocas pero con early stopping
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('models/modelo_inicial.h5')
    return history

# Reentrenamiento optimizado
def retrain_model():
    train_generator = create_generator('new_data', augment=True)
    
    model = tf.keras.models.load_model('models/modelo_inicial.h5')
    
    # Fine-tuning con learning rate más bajo
    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        epochs=5,
        verbose=1
    )
    
    model.save('models/modelo_actualizado.h5')
    return history

if __name__ == '__main__':
    train_history = train_model()
    # retrain_history = retrain_model()  # Descomentar cuando haya nuevos datos

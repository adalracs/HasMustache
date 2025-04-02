#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2025 Andrés Riascos (adalracs@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Project: IA for Mustache Detection (CNN Training)
Author: Andrés Riascos (adalracs@gmail.com)
Location: Cali, Colombia
Date: April 1, 2025
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Configuración de directorios (consistente con app.py)
DATA_PATHS = {
    'train': 'data/train',  # Subdirectorios: mustache_male/, no_mustache_female/, etc.
    'validation': 'data/validation'  # Misma estructura que train/
}

def create_data_generator(data_dir, augment=False, batch_size=16):
    """Crea generador de imágenes con o sin aumentación de datos"""
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',  # Para clasificación multi-etiqueta
        classes=['mustache_male', 'mustache_female', 'no_mustache_male', 'no_mustache_female'],
        shuffle=True
    )

def build_model():
    """Construye el modelo CNN ligero"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3, 3), activation='relu'),  # Capa convolucional adicional
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Regularización Dropout
        Dense(64, activation='relu'),  # Capa densa adicional
        Dense(4, activation='softmax')# 4 clases: mustache_male, mustache_female, etc.
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def evaluate_model(model, generator):
    """Genera reporte de métricas"""
    y_true = generator.classes
    y_pred = model.predict(generator).argmax(axis=1)
    
    print("\n=== Métricas de Evaluación ===")
    print(classification_report(y_true, y_pred, target_names=generator.class_indices.keys()))
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))

def train():
    """Pipeline completo de entrenamiento"""
    train_generator = create_data_generator(DATA_PATHS['train'], augment=True)
    val_generator = create_data_generator(DATA_PATHS['validation'])

    model = build_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(
            'models/modelo_inicial.keras',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

    print("Iniciando entrenamiento...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    # Guardar modelo y métricas
    model.save('models/modelo_inicial.keras')
    evaluate_model(model, val_generator)

if __name__ == '__main__':
    train()
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

Project: IA for Mustache Detection (Model Fine-Tuning)
Author: Andrés Riascos (adalracs@gmail.com)
Location: Cali, Colombia
Date: April 1, 2025
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

# Directorios (consistentes con app.py y model_training.py)
NEW_DATA_DIR = 'new_data'  # Subdirectorios: mustache_male/, no_mustache_female/, etc.
MODEL_PATH = 'models/modelo_inicial.h5'
SAVE_PATH = 'models/modelo_actualizado.h5'

def create_generator():
    """Generador para datos nuevos (con aumentación mínima)"""
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,  # Aumento de datos muy ligero
        horizontal_flip=True
    )
    return datagen.flow_from_directory(
        NEW_DATA_DIR,
        target_size=(128, 128),
        batch_size=4,  # Batch pequeño para fine-tuning
        class_mode='categorical',
        shuffle=True
    )

def retrain():
    """Pipeline de reentrenamiento con nuevos datos"""
    # 1. Cargar modelo existente y datos nuevos
    model = tf.keras.models.load_model(MODEL_PATH)
    train_generator = create_generator()

    # 2. Recompilar con learning rate bajo
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # LR 10x menor que en entrenamiento inicial
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Fine-tuning (solo 3 épocas)
    print("\nReentrenando con nuevos datos...")
    model.fit(
        train_generator,
        epochs=3,
        verbose=1
    )

    # 4. Guardar y evaluar
    model.save(SAVE_PATH)
    print("\nModelo actualizado guardado en:", SAVE_PATH)
    
    # Evaluación con los mismos datos de reentrenamiento
    y_true = train_generator.classes
    y_pred = model.predict(train_generator).argmax(axis=1)
    
    print("\n📊 Métricas después del fine-tuning:")
    print(classification_report(y_true, y_pred, target_names=train_generator.class_indices.keys()))
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    retrain()
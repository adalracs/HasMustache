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

Project: IA for Mustache Detection (Flask API)
Author: Andrés Riascos (adalracs@gmail.com)
Location: Cali, Colombia
Date: April 1, 2025
"""

from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# Configuración inicial
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Estructura de directorios (ajustada a tu esquema)
DATA_PATHS = {
    'train': {
        'mustache_male': 'data/train/mustache_male',
        'mustache_female': 'data/train/mustache_female',
        'no_mustache_male': 'data/train/no_mustache_male',
        'no_mustache_female': 'data/train/no_mustache_female'
    },
    'new_data': {
        'mustache_male': 'new_data/mustache_male',
        'mustache_female': 'data/mustache_female',
        'no_mustache_male': 'new_data/no_mustache_male',
        'no_mustache_female': 'new_data/no_mustache_female'
    }
}

# Crear directorios necesarios
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
for dir_type in DATA_PATHS.values():
    for dir_path in dir_type.values():
        os.makedirs(dir_path, exist_ok=True)

# Cargar modelo
model = tf.keras.models.load_model('models/modelo_inicial.h5')

@app.route('/')
def home():
    """Renderiza la página principal con el formulario de subida"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para procesar imágenes y predecir bigotes"""
    if 'file' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    # Guardar y procesar imagen
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    has_mustache = bool(prediction[0][0] > 0.5)
    is_male = bool(prediction[0][1] > 0.5)

    return jsonify({
        'has_mustache': has_mustache,
        'is_male': is_male,
        'image_url': f'/static/uploads/{filename}'
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint para recibir correcciones humanas"""
    data = request.json
    filename = os.path.basename(data['image_url'])
    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Clasificar en directorio correspondiente
    mustache_key = 'mustache' if data['correct_mustache'] else 'no_mustache'
    gender_key = 'male' if data['correct_gender'] else 'female'
    dest_dir = DATA_PATHS['new_data'][f"{mustache_key}_{gender_key}"]
    
    os.rename(src_path, os.path.join(dest_dir, filename))
    return jsonify({'status': 'Feedback procesado correctamente'})

@app.route('/metrics')
def metrics():
    """Endpoint para mostrar métricas del modelo"""
    val_generator = create_generator(DATA_PATHS['train'])
    y_true = val_generator.classes
    y_pred = model.predict(val_generator).argmax(axis=1)
    
    return jsonify({
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    })

def create_generator(data_dirs):
    """Crea generador de datos para evaluación"""
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        data_dirs,
        target_size=(128, 128),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
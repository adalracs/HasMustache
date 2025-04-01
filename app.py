from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Cargar modelo (asegúrate de que existe 'models/modelo_inicial.h5')
model = tf.keras.models.load_model('models/modelo_inicial.h5')

# Crear directorios si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('new_data/mustache_male', exist_ok=True)
os.makedirs('new_data/mustache_female', exist_ok=True)
os.makedirs('new_data/no_mustache_male', exist_ok=True)
os.makedirs('new_data/no_mustache_female', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Preprocesar imagen
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predecir
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
    data = request.json
    filename = os.path.basename(data['image_url'])
    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Mover imagen a la carpeta correcta según retroalimentación
    mustache_key = 'mustache' if data['correct_mustache'] else 'no_mustache'
    gender_key = 'male' if data['correct_gender'] else 'female'
    dest_dir = f"new_data/{mustache_key}_{gender_key}"
    os.makedirs(dest_dir, exist_ok=True)
    os.rename(src_path, os.path.join(dest_dir, filename))
    
    return jsonify({'status': 'Feedback received'})

if __name__ == '__main__':
    app.run(debug=True)

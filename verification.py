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

Project: IA for Mustache Detection (Verification Script)
Author: Andrés Riascos (adalracs@gmail.com)
Location: Cali, Colombia
Date: April 1, 2025
"""

import tensorflow as tf
from tensorflow.keras import layers
import flask
import PIL
import platform
import cpuinfo

def print_separator(title):
    """Función auxiliar para imprimir secciones"""
    print(f"\n=== {title} {'=' * (50 - len(title))}")

def check_gpu():
    """Verifica hardware de aceleración disponible"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU detectada:")
        for gpu in gpus:
            print(f"  - Nombre: {gpu.name}, Tipo: {gpu.device_type}")
        return True
    else:
        print("No se detectó GPU. Se usará CPU.")
        return False

def check_cpu():
    """Obtiene información detallada de la CPU"""
    info = cpuinfo.get_cpu_info()
    print(f" CPU: {info['brand_raw']}")
    print(f"  - Núcleos: {info['count']}")
    print(f"  - Arquitectura: {info['arch']}")

def test_minimal_model():
    """Prueba mínima de funcionamiento de TensorFlow"""
    try:
        model = tf.keras.Sequential([layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer='sgd', loss='mse')
        print("✔ Prueba de modelo exitosa (CPU/GPU funcionales)")
    except Exception as e:
        print(f"❌ Error en prueba de modelo: {str(e)}")

def main():
    print_separator("Verificación de Entorno para IA Didáctica")
    
    # 1. Versiones de software
    print("📦 Versiones críticas:")
    print(f"  - TensorFlow: {tf.__version__}")
    print(f"  - Flask: {flask.__version__}")
    print(f"  - Pillow (PIL): {PIL.__version__}")
    print(f"  - Python: {platform.python_version()}")
    
    # 2. Hardware
    print_separator("Hardware Disponible")
    has_gpu = check_gpu()
    check_cpu()
    
    # 3. Pruebas funcionales
    print_separator("Pruebas Funcionales")
    test_minimal_model()
    
    # Mensaje final
    print_separator("Resultado")
    print("¡Entorno verificado con éxito!") if has_gpu else \
    print("Entorno verificado (modo CPU). Para mejor rendimiento, considere usar GPU.")

if __name__ == '__main__':
    main()
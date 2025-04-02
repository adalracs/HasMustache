# 🎓 Academic Project: Mustache Detection with AI

[![Apache License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python](https://img.shields.io/badge/Python-3.11%2B-yellowgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)

## Description
**An educational CNN project** designed to teach convolutional neural networks using low-resource hardware. Perfect for students who:
- Want to start with AI using minimal resources (only 1GB RAM needed!)
- Lack access to expensive GPUs
- Seek to understand the full pipeline: from data preprocessing to Flask deployment

> "Teaching AI shouldn't require expensive hardware" — **Andrés Riascos**

## Technologies
- **Backend**: Python 3.11, TensorFlow 2.16 (CPU version)
- **Interface**: Flask
- **Image Processing**: OpenCV, Pillow
- **License**: Apache 2.0

## Quick Start (3 Steps)

### 1. Minimum Requirements
```bash
# Hardware:
- x86-64 CPU (up to 10 years old)
- 1GB RAM
- 500MB disk space

# Software:
- OS: Linux (Debian/Ubuntu) or Windows 10+
- Python 3.11+

Installation
# Clone repository
git clone https://github.com/adalracs/HasMustache.git
cd mustache-detection-ai

# Install dependencies (no GPU needed)
pip install -r requirements.txt --no-cache-dir

Run the Project
# Verify your environment
python verification.py

# Train initial model
python model_training.py

# Launch web API
python app.py

Includes 40 pre-labeled images in:
data/
├── train/
│   ├── mustache_male/
│   ├── no_mustache_female/
│   └── ...
└── validation/

Why This Project?

    Accessible Education: Designed for low-end PCs

    Hands-on Learning: Covers full AI pipeline

    Open for Contributions: Your improvements help students worldwide!

How to Contribute

We welcome students, educators, and enthusiasts to:

    Report bugs via Issues

    Suggest dataset/code improvements

    Translate documentation

    Enhance the web interface
	

Contribution Guide:
1. Fork the project
2. Create a branch: git checkout -b my-improvement
3. Submit a Pull Request
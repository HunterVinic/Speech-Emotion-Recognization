# 🎙️ Speech Emotion Recognition System

A Speech Emotion Recognition (SER) system built using Deep Learning (LSTM & CNN) models.

This project includes:

- 🌐 Flask Web Application (Upload audio & predict emotion)
- 🖥️ Tkinter Desktop GUI
- 🎤 Live Audio Recording Version
- 🐳 Docker Support
- 🧠 Multiple trained .h5 deep learning models

---

## 🚀 Features

- Upload audio file and predict emotion
- Live audio recording emotion detection
- Multiple model architectures (LSTM & CNN)
- Docker container support
- Web + Desktop interfaces
- MFCC feature extraction using Librosa

---

## 🧠 Emotion Labels

### Emotion Codes

01 = Neutral  
02 = Angry  
03 = Happy  
04 = Sad  
05 = Frustrated  

### Intensity Labels

a = Normal  
b = Strong  

---

## 🛠 Tech Stack

- Python 3.9
- Flask
- TensorFlow / Keras
- Librosa
- NumPy
- Tkinter
- Docker

---

## 📂 Project Structure

.
├── main_flask.py
├── main.py
├── templates/
│   └── index.html
├── uploads/
├── model93.h5
├── speech_emotion_model.h5
├── Model_B.h5
├── Model_C.h5
├── Model_D.h5
├── my_model.h5
├── requirements.txt
├── Dockerfile
├── docker-compose.yml

---

## 🔬 Model Architecture

### Model A – LSTM
- LSTM(123)
- Dense(64)
- Dropout
- Dense(32)
- Dropout
- Dense(Softmax)

### Model B / C / D – CNN Variants
- Conv1D layers
- MaxPooling
- Dropout
- Dense Softmax output

All models use MFCC audio features (40 coefficients).

---

## 🎧 How It Works

1. Audio file is uploaded or recorded
2. MFCC features are extracted using Librosa
3. Features are reshaped to match model input
4. Pre-trained .h5 model loads weights
5. Model predicts emotion class
6. Result is displayed

---

## 💻 Running the Flask Web App (Local)

### Install Dependencies

pip install -r requirements.txt

### Run the App

python main_flask.py

Open in browser:

http://localhost:8000

---

## 🖥️ Running the Tkinter Desktop App

python main.py

Then:
1. Open an audio file
2. (Optional) Play audio
3. Click Predict

---

## 🐳 Running with Docker

### Build Image

docker build -t speech-emotion .

### Run Container

docker run -p 8000:8000 speech-emotion

Open:

http://localhost:8000

---

## 🐳 Using Docker Compose

docker-compose up --build

---

## 📦 Requirements

flask
numpy
keras
librosa
tensorflow

---

## 📌 Notes

- Make sure all .h5 model files are in the project directory.
- The Flask app saves uploaded files in the /uploads directory.
- Models expect MFCC features with shape (40, 1).
- TensorFlow must match the Keras version used during training.

---

## ⚠️ Important

If running inside Docker, make sure:
- Model paths are relative (not absolute)
- Required .h5 files are copied into the container
- Ports are correctly exposed

---

Copyright (c) 2026 Sheshehang Limbu (HunterVinic)

All rights reserved.

This project and its source code may not be copied, modified,
distributed, or used without explicit permission from the author.

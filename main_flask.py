import os
from flask import Flask, render_template, request
import numpy as np
from keras.models import Sequential
from keras.layers import *
import librosa

app = Flask(__name__)

emotions_used = ['1', '2', '3', '4', '5']


def extract_mfcc(audio_path):
    ''' Extracts MFCC features and outputs the average of each dimension '''
    y, sr = librosa.load(audio_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs


def model_A():
    model = Sequential()
    model.add(LSTM(123, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def predict_emotion(audio_path):
    mfcc = extract_mfcc(audio_path)
    model = model_A()
    model.load_weights("/Users/ak/Desktop/SpeechEmotion/model93.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    predicted_probabilities = model.predict(mfcc)
    predicted_class = np.argmax(predicted_probabilities)
    return emotions_used[predicted_class]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'audio_file' not in request.files:
            return render_template('index.html')

        file = request.files['audio_file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html')

        # Create the 'uploads' directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Save the uploaded file
        audio_path = 'uploads/audio.wav'
        file.save(audio_path)

        # Perform emotion prediction
        emotion = predict_emotion(audio_path)

        return render_template('index.html', emotion=emotion)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


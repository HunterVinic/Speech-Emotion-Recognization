import tkinter as tk
import sounddevice as sd
import soundfile as sf
import numpy as np
from keras.models import load_model
import librosa

# Load the pre-trained model
model = load_model("speech_emotion_model.h5")

# Function for extracting features from audio
def extract_features(audio):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)

    # Pad or truncate the features to match the desired shape
    max_seq_length = 100  # Maximum sequence length

    if mfccs.shape[1] < max_seq_length:
        # Pad the features with zeros
        mfccs = np.pad(mfccs, ((0, 0), (0, max_seq_length - mfccs.shape[1])), mode='constant')
    else:
        # Truncate the features
        mfccs = mfccs[:, :max_seq_length]

    # Reshape the features to match the desired shape
    mfccs = np.expand_dims(mfccs, axis=-1)

    return mfccs



def recognize_emotion(audio):
    # Extract features from the audio
    features = extract_features(audio)
    # Reshape the features to match the model's input shape
    reshaped_features = np.expand_dims(features, axis=0)
    # Perform emotion recognition using the loaded model
    predictions = model.predict(reshaped_features)
    # ...

# Function for recording audio
def record_audio():
    fs = 22050  # Sample rate
    duration = 5  # Duration of recording (in seconds)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait for recording to finish
    audio = recording.flatten()  # Flatten the recording to mono
    sf.write("recorded_audio.wav", audio, fs)  # Save the recording to a WAV file
    recognize_emotion(audio)  # Analyze the recorded audio

# Create the GUI
window = tk.Tk()
window.title("Live Speech Emotion Recognition")

# Create a button to record audio
record_button = tk.Button(window, text="Record Audio", command=record_audio)
record_button.pack(pady=20)

# Create a label to display the predicted emotion
result_label = tk.Label(window, text="")
result_label.pack()

# Start the GUI event loop
window.mainloop()

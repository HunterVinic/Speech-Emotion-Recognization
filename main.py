
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from playsound import playsound
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras
import librosa
import h5py

top = Tk()
topFrame = Frame(top)
top.title('Thai Emotion Recognitions using audio')
top.geometry('330x540')

canvas = Canvas(top, width=40, height=40, bd=0, bg='white')
canvas.grid(row=1, column=0)

def openAudio():
    ''' Opens the audio file'''
    File = askopenfilename(title='Open an Audio file')
    e.set(File)


def playAudio():
    ''' Play the audio file '''
    playsound(e.get())


e = StringVar()
submit_button = Button(top, text='Open an Audio file', command=openAudio)
submit_button.grid(row=1, column=0)

submit_button = Button(top, text='Play an Audio File', command=playAudio)
submit_button.grid(row=3, column=0)

emotions_used = ['1a', '1b', '2a', '2b', '3a', '3b', '4a', '4b', '5a', '5b']


def model_A():
    model = Sequential()
    model.add(LSTM(123, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def model_B():
    ''' CNN model definition and architecture
    The model returned here is referred to as model B
    '''
    model = Sequential()
    model.add(Conv1D(8, kernel_size=3, input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(16, kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=3))
    model.add(Activation('relu'))
    model.add(Conv1D(16, kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def model_C():
    ''' CNN model definition and architecture
    The model returned here is referred to as model C
    '''
    model = Sequential()
    model.add(Conv1D(8, 5, padding='same', input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(16, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(32, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(16, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def model_D():
    ''' CNN model definition and architecture
    The model returned here is referred to as model D
    '''
    model = Sequential()
    model.add(Conv1D(128, 5, padding='same',
                     input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def Predict_A():
    ''' Prediction using model A'''
    mfcc = extract_mfcc(e.get())
    model = model_A()
    model.load_weights("my_model.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    predicted_probabilities = model.predict(mfcc)
    predicted_class = np.argmax(predicted_probabilities)
    textvar = "The Emotion is : %s" % emotions_used[predicted_class]
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar + '\n')
    t1.update()


def Predict_B():
    ''' Prediction using model B'''
    mfcc = extract_mfcc(e.get())
    model = model_B()
    model.load_weights("Model_B.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict_classes(mfcc)
    textvar = "The object is : %s" % (emotions_used[int(cls_wav)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar + '\n')
    t1.update()


def Predict_C():
    ''' Prediction using model C'''
    mfcc = extract_mfcc(e.get())
    model = model_C()
    model.load_weights("Model_C.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict_classes(mfcc)
    textvar = "The object is : %s" % (emotions_used[int(cls_wav)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar + '\n')
    t1.update()


def Predict_D():
    ''' Prediction using model D'''
    mfcc = extract_mfcc(e.get())
    model = model_D()
    model.load_weights("Model_D.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict_classes(mfcc)
    textvar = "The object is : %s" % (emotions_used[int(cls_wav)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar + '\n')
    t1.update()


def extract_mfcc(wav_file_name):
    ''' Extracts mfcc features and outputs the average of each dimension'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs


submit_button = Button(top, text='Predict using LSTM Model', command=Predict_A, font=('Arial', 12), bg='#4CAF50', fg='Black', padx=15, pady=10, height=2, width=20)
submit_button.grid(row=25, column=0)

l1 = Label(top, text='How to use this model:', font=('Arial', 15), fg='#333333')
l1.grid(row=5)

l1 = Label(top, text='Step-1: Press "Open the Audio file"', font=('Arial', 12), fg='#333333')
l1.grid(row=6)

l1 = Label(top, text='Step-2: Choose the audio file', font=('Arial', 12), fg='#333333')
l1.grid(row=7)

l1 = Label(top, text='Step-3(Optional): Play the audio file', font=('Arial', 12), fg='#333333')
l1.grid(row=8)

l1 = Label(top, text='Step-4: Press the Predict Button', font=('Arial', 12), fg='#333333')
l1.grid(row=9)

l1 = Label(top, text='** Important Note about Results **', font=('Arial', 12), fg='Blue')
l1.grid(row=12)

l1 = Label(top, text='01 = neutral', font=('Arial', 12), fg='#333333')
l1.grid(row=13, column=0)
l1 = Label(top, text='02 = angry', font=('Arial', 12), fg='#333333')
l1.grid(row=14, column=0)
l1 = Label(top, text='03 = happy', font=('Arial', 12), fg='#333333')
l1.grid(row=15, column=0)
l1 = Label(top, text='04 = sad', font=('Arial', 12), fg='#333333')
l1.grid(row=16, column=0)
l1 = Label(top, text='05 = frustrated', font=('Arial', 12), fg='#333333')
l1.grid(row=17, column=0)

l1 = Label(top, text='--------------------------------------------', font=('Arial', 12), fg='silver')
l1.grid(row=19, column=0)

l1 = Label(top, text='** Intensity **', font=('Arial', 12), fg='red')
l1.grid(row=20, column=0)
l1 = Label(top, text='a = Normal', font=('Arial', 12), fg='#333333')
l1.grid(row=21, column=0)
l1 = Label(top, text='b = Strong', font=('Arial', 12), fg='#333333')
l1.grid(row=22, column=0)


t1 = Text(top, bd=0, width=30, height=4, font=('Arial', 19))
t1.grid(row=0, column=0)


top.mainloop()

if __name__ == '__main__':
    top.run(port=8000)



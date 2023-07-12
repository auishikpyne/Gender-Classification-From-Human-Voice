import pyaudio
import os
import wave
import glob 
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from tqdm import tqdm
from utils import load_data, split_data, create_model
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model = create_model()
    
model.load_weights("results/best_model.h5")

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


def gender_pred(file):
    features = extract_feature(file, mel=True).reshape(1, -1)
    
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
    
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"
    result = {"Predicted Gender" : gender,
              "Male probability" : f"{male_prob*100:.2f}%",
              "Female probability" : f"{female_prob*100:.2f}%"}
    
    return result


if __name__ == "__main__":
   
    # audio_files = glob.glob('/home/auishik/gender_classification/test_data/dataset/male_denoised/*.flac')
    file = '/home/auishik/age_prediction_from_voice/Age-Estimation-based-on-Human-Voice/tusher.wav'
    
        
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"
    
    
    
    print("Result:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
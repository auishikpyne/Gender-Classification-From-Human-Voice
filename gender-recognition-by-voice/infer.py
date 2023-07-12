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

tf_xla_flags = os.getenv("TF_XLA_FLAGS")  # Get the current value of TF_XLA_FLAGS
if tf_xla_flags:
    # Remove the --xla_gpu_disable flag from the string
    modified_tf_xla_flags = tf_xla_flags.replace("--xla_gpu_disable", "")
    os.environ["TF_XLA_FLAGS"] = modified_tf_xla_flags  # Update the value of TF_XLA_FLAGS


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000


SILENCE = 30

model = create_model()
    
model.load_weights("results/best_model.h5")

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r
    
    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

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

if __name__ == "__main__":
   
    # audio_files = glob.glob('/home/auishik/gender_classification/test_data/dataset/male_denoised/*.flac')
    audio_files = ['/home/auishik/age_prediction_from_voice/Age-Estimation-based-on-Human-Voice/tusher.wav']
    with open('prediction.txt', 'w') as f:
        f.write("File Path, Prediction\n")
        
        for file in tqdm(audio_files):
        
            features = extract_feature(file, mel=True).reshape(1, -1)
            # predict the gender!
            male_prob = model.predict(features)[0][0]
            female_prob = 1 - male_prob
            gender = "Male" if male_prob > female_prob else "Female"
            
            f.write(f"{file}, {gender}\n")
            
            print("Result:", gender)
            print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
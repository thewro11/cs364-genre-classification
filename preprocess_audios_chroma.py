import multiprocessing
from functools import partial
from tqdm import tqdm
import librosa

def load_chroma_features(audio_file,sr,duration,hop_length):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sr,mono=True,offset=0.0, duration=duration)

    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    return chroma

def preprocess_audios_chroma(audio_files,num_processes,sr=22050,duration=5.0,hop_length=512):
    pool = multiprocessing.Pool(num_processes)
    func = partial(load_chroma_features,sr=sr,duration=duration,hop_length=hop_length)
    
    features = []
    with tqdm(total=len(audio_files)) as pbar:
        for chroma in pool.imap(func, audio_files):
            features.append(chroma)
            pbar.update(1)
            
    pool.close()
    pool.join()
    return features
# Preprocessing Steps
# 1- Load a file
# 2- Pad the signal
# 3- Extract log spectrogram
# 4- Normalize spectrogram
# 5- Save normalized spectrogram

import os
import pickle
import librosa
import numpy as np


class DataLoader:
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, fpath):
        signal = librosa.load(fpath,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal


class Padder:
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_arr = np.pad(array,
                            (num_missing_items, 0),
                            mode=self.mode)
        
        return padded_arr

    def right_pad(self, array, num_missing_items):
        padded_arr = np.pad(array,
                            (0, num_missing_items),
                            mode=self.mode)
        
        return padded_arr
    

class LogSpectrogramExtractor:
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
        
    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxScaler:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min())/(array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, array, min, max):
        array = (array - self.min)/(self.max - self.min)
        array = array * (max - min) + min
        return array


class Saver:
    def __init__(self, feature_save_dir, min_max_val_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_val_save_dir = min_max_val_save_dir
        
    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        
    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path
    
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_val_save_dir, "min_max_val.pkl")
        self._save(min_max_values, save_path)
        
    @staticmethod
    def _save(data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)



class PreprocessingPipeline:
    def __init__(self):
        self._loader = None
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_val = {}
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate*loader.duration)

    def process(self, audio_file_dir):
        for root, _, files in os.walk(audio_file_dir):
            for file in files:
                fname = os.path.join(root, file)
                self._process_file(fname)
                print(f"Processed File {fname}")
        self.saver.save_min_max_values(self.min_max_val)
    
    def _is_padding_required(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        else:
            return False
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_val(self, save_path, min, max):
        self.min_max_val[save_path] = {
            "min": min,
            "max": max
        }
    
    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_required(signal):
            signal = self._apply_padding(signal)
            
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        
        self._store_min_max_val(save_path, feature.min(), feature.max())
        
        
        
if __name__ == "__main__":
    frame_size = 512
    hop_len = 256
    duration = 0.74
    sample_rate = 22050
    mono = True

    spectrogram_save_dir = "/home/sachin/Data/fsdd/spectrograms/"
    min_max_val_save_dir = "/home/sachin/Data/fsdd/"
    files_dir = "/home/sachin/Codes/fsdd/recordings/"
    
    loader = DataLoader(sample_rate, duration, mono)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(frame_size, hop_len)
    min_max_normalizer = MinMaxScaler(0, 1)
    saver = Saver(spectrogram_save_dir, min_max_val_save_dir)
    
    pipeline = PreprocessingPipeline()
    pipeline.loader = loader
    pipeline.padder = padder
    pipeline.extractor = log_spectrogram_extractor
    pipeline.normalizer = min_max_normalizer
    pipeline.saver = saver

    pipeline.process(files_dir)
    
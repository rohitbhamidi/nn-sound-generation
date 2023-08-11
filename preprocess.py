import librosa
import numpy as np
import os
import pickle

class Loader:
    '''Responsible for loading the audio file.'''

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, filepath):
        signal, _ = librosa.load(
            path=filepath,
            sr=self.sample_rate,
            duration=self.duration,
            mono=self.mono
        )
        return signal

class Padder:
    '''Responsible for applying padding.'''

    def __init__(self, mode='constant'):
        self.mode = mode

    def left_pad(self, array, num_missing_items):       # append the missing items to the beginning of the array
        padded_array = np.pad(
            array=array,
            pad_width=(num_missing_items, 0),           # inserts `num_missing_items` 0s at the beginning of the array
            mode=self.mode
        )
        return padded_array

    def right_pad(self, array, num_missing_items):      # append the missing items to the end of the array
        padded_array = np.pad(
            array=array,
            pad_width=(0, num_missing_items),           # inserts `num_missing_items` 0s at the end of the array
            mode=self.mode
        )
        return padded_array

class LogSpectrogramExtractor:
    '''Extracts Log Spectrograms in dB from a time series signal.'''

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(                            
            y=signal,
            n_fft=self.frame_size,
            hop_length=self.hop_length
        )[:-1]                                          # [1+(frame_size)/2, num_frames] -> [frame_size/2, num_frames]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

class MinMaxNormalizer:
    '''Applies MinMax normalization to an array.'''

    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def normalize(self, array):
        norm_array = (array - array.min())/(array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min)/(self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array

class Saver:
    '''Responsible for saving features, and the min max values.'''
    
    def __init__(self, save_feature_dir, min_max_values_save_dir):
        self.save_feature_dir = save_feature_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, norm_feature, filepath):
        save_path = self._generate_save_path(filepath)
        np.save(save_path, norm_feature)
        return save_path

    def _generate_save_path(self, filepath):
        file_name = os.path.split(filepath)[1]
        save_path = os.path.join(self.save_feature_dir, file_name+'.npy')
        return save_path
    
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, 'min_max_values.pkl')
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(min_max_values, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(min_max_values, f)


class PreprocessingPipeline:
    '''Processes audio file in a directory, applying the following steps to each file:
        1. Load a file
        2. Pad the signal
        3. Extract the spectrogram
        4. Normalize the spectrogram
        5. Save the spectrogram
    Storing the minmax values for all log spectrograms.
    '''
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_directory):
        for root, _, files in os.walk(audio_files_directory):
            for file in files:
                filepath = os.path.join(root, file)
                self._process_file(filepath)
                print(f'Processed file {filepath}')
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, filepath):
        signal = self.loader.load(filepath)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, filepath)
        self._store_min_max_values(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_values(self, save_path, min_value, max_value):
        self.min_max_values[save_path] = {
            'min': min_value,
            'max': max_value
        }

if __name__=='__main__':
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION =.74
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = './FSDD/spectrograms/'
    MINMAX_SAVE_DIR = './FSDD/'
    FILES_DIR = './FSDD/audio/recordings'

    # instantiate all objects
    loader = Loader(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        mono=MONO
    )

    padder = Padder()

    log_spectrogram_extractor = LogSpectrogramExtractor(
        frame_size=FRAME_SIZE,
        hop_length=HOP_LENGTH
    )

    min_max_normalizer = MinMaxNormalizer(
        max_value=1,
        min_value=0
    )

    saver = Saver(
        save_feature_dir=SPECTROGRAMS_SAVE_DIR,
        min_max_values_save_dir=MINMAX_SAVE_DIR
    )

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normalizer = min_max_normalizer
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(audio_files_directory=FILES_DIR)
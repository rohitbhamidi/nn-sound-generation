import os
import pickle
import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from autoencoder import VAE

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = './samples/original'
SAVE_DIR_GENERATED = './samples/generated'
MIN_MAX_VALUES_PATH = './FSDD/min_max_values.pkl'
SPECTROGRAMS_PATH = './FSDD/spectrograms/'

def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)                # [n_bins, n_frames], need an extra dimension num_channels
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]                      # -> [num_samples=3000, num_bins=256, num_frames=64, num_channels=1]
    return x_train, file_paths

def select_spectrograms(spectrograms, filepaths, min_max_values, num_spectrograms=2):
    sampled_indices = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indices]
    filepaths = [filepaths[index] for index in sampled_indices]
    sampled_min_max_values = [min_max_values[filepath] for filepath in filepaths]
    print(filepaths)
    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate = 22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i)+'.wav')
        sf.write(save_path, signal, sample_rate)

if __name__ == '__main__':
    vae = VAE.load('model')                                 # init model
    sound_generator = SoundGenerator(vae, HOP_LENGTH)       # init generator

    with open(MIN_MAX_VALUES_PATH, 'rb') as f:
        min_max_values = pickle.load(f)                     # load min-max values
    
    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)        # load spectrograms and filepaths
    sampled_specs, sampled_minmax = select_spectrograms(specs, file_paths, min_max_values, 5)

    signals, latent_reps = sound_generator.generate(sampled_specs, sampled_minmax)
    orignial_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_minmax)

    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(orignial_signals, SAVE_DIR_ORIGINAL)
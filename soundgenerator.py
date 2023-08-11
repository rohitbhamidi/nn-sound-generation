from preprocess import MinMaxNormalizer

import librosa

class SoundGenerator:
    '''Responsible for generating audio from spectrograms.'''

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormalizer(                # normalized spectrograms -> denormalized audio
            min_value=0,
            max_value=1
        )

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_reps = self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_reps
    
    def convert_spectrograms_to_audio(self, generated_spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(generated_spectrograms, min_max_values):
            log_spectrogram = spectrogram[:, :, 0]                                      # reshape log spectrogram
            denormed_log_spectrogram = self._min_max_normalizer.denormalize(            # apply denormalization
                norm_array=log_spectrogram,
                original_min=min_max_value['min'],
                original_max=min_max_value['max']
            )
            spec = librosa.db_to_amplitude(S_db=denormed_log_spectrogram)               # log spec -> spec
            signal = librosa.istft(                                                     # apply Griffin-Lim algorithm
                stft_matrix=spec,
                hop_length=self.hop_length
            )
            signals.append(signal)
        return signals
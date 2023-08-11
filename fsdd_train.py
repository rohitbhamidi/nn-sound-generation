from autoencoder import VAE
import os
import numpy as np

LEARNING_RATE = .0005
BATCH_SIZE = 64
NUM_EPOCHS = 5

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)                # [n_bins, n_frames], need an extra dimension num_channels
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]                      # -> [num_samples=3000, num_bins=256, num_frames=64, num_channels=1]
    return x_train


def train(x_train, learning_rate, batch_size, num_epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),                           # same shape as the FSDD data
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3,3,3,3,3),
        conv_strides=(2,2,2,2,(2,1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate=learning_rate)
    autoencoder.train(
        x_train=x_train,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    return autoencoder

if __name__=='__main__':
    x_train = load_fsdd('./FSDD/spectrograms/')
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
    autoencoder.save('./model')
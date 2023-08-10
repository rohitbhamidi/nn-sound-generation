from autoencoder import Autoencoder
from keras.datasets import mnist

LEARNING_RATE = .0001
BATCH_SIZE = 32
NUM_EPOCHS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()        # grayscale data

    x_train = x_train.astype('float32') / 255                       # normalize train data
    x_train = x_train.reshape(x_train.shape + (1,))                  # reshape train data, adding channel dim
    x_test = x_test.astype('float32') / 255                         # normalize test data
    x_test = x_test.reshape(x_test.shape + (1,))                     # reshape test data, adding channel dim

    return x_train, y_train, x_test, y_test

def train(x_train, learning_rate, batch_size, num_epochs):
    autoencoder = Autoencoder(
        input_shape=(28,28,1),                      # same shape as the MNIST data
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2
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
    x_train, _, _, _ = load_mnist()
    autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
    autoencoder.save('./model')
from keras import Model
from keras import backend as K
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import os
import pickle
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class Autoencoder:
    '''
    Autoencoder represents a Deep Convolutional autoencoder architecture with mirrored encoder and decoder components.
    '''

    def __init__(self, 
                 input_shape,       # [width, height, num_channels] -> spectrograms can be interpreted as grayscale images
                 conv_filters,      # no of filters for each layer, list or tuple
                 conv_kernels,      # no of kernels for each layer, list or tuple
                 conv_strides,      # no of strides for each layer, list or tuple
                 latent_space_dim   # dimensionality of the bottleneck, int
                 ):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None           # tensorflow model for the whole model

        self._num_conv_layers = len(self.conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(
            x=x_train,          # training data
            y=x_train,          # output of model should be the same as the training data
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True
        )

    def reconstruct(self, images):
        latent_reps = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_reps)
        return reconstructed_images, latent_reps

    def save(self, filepath='.'):
        self._create_nonexistant_filepath(filepath)
        self._save_parameters(filepath)
        self._save_weights(filepath)

    def _create_nonexistant_filepath(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def _save_parameters(self, filepath):
        parameters = [
            self.input_shape,       # [width, height, num_channels] -> spectrograms can be interpreted as grayscale images
            self.conv_filters,      # no of filters for each layer, list or tuple
            self.conv_kernels,      # no of kernels for each layer, list or tuple
            self.conv_strides,      # no of strides for each layer, list or tuple
            self.latent_space_dim   # dimensionality of the bottleneck, int
        ]
        savepath = os.path.join(filepath, 'parameters.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(parameters, f)

    def _save_weights(self, filepath):
        savepath = os.path.join(filepath, 'weights.h5')
        self.model.save_weights(savepath)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, filepath='.'):
        parameters_path = os.path.join(filepath, 'parameters.pkl')
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(filepath, 'weights.h5')
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')
    
    def _add_conv_layers(self, encoder_input):
        '''Creates all convolutional blocks in the encoder.'''
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)    # getting back a graph of layers with an added node
        return x
    
    def _add_conv_layer(self, layer_index, x):
        '''Adds a convolutional block to a graph of layers consisting of (2D Convolution + ReLU + batch normalization)'''
        layer_number = layer_index+1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],         # fetch the number of filters at `layer_index`
            kernel_size=self.conv_kernels[layer_index],     # fetch the kernel size at `layer_index`
            strides=self.conv_strides[layer_index],         # fetch the number of strides at `layer_index` 
            padding='same',
            name=f'encoder_conv_layer_{layer_number}'
        )

        x = conv_layer(x)                                              # drawing an arrow from x to the `conv_layer`
        x = ReLU(name=f'encoder_relu_{layer_number}')(x)               # activation function
        x = BatchNormalization(name=f'encoder_bn_{layer_number}')(x)   # batch normalization

        return x
    
    def _add_bottleneck(self, x):
        '''Flatten data and add bottleneck (dense layer).'''
        self._shape_before_bottleneck = K.int_shape(x)[1:]                 # [batch_size, width, height, num_channels] -> [width, height, num_channels]
        x = Flatten()(x)
        x = Dense(units=self.latent_space_dim, name='encoder_output')(x)
        return x
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name='decoder_input')
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)               # [width, height, num_channels] -> width*height*num_channels = num_neurons
        dense_layer = Dense(units=num_neurons, name='decoder_dense')(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(target_shape=self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        '''Add conv. transpose blocks.'''
        for layer_index in reversed(range(1, self._num_conv_layers)):       # loop through all the conv. layers in reverse order and stop at the first layer.
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers-layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding='same',
            name=f'decoder_conv_trans_layer_{layer_number}'
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f'decoder_relu_{layer_number}')(x)
        x = BatchNormalization(name=f'decoder_bn_{layer_number}')(x)
        return x
    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,                                                      # [width, height, num_channels=1] => spectrogram/grayscale
            kernel_size=self.conv_kernels[0],                               # data for the first conv layer
            strides=self.conv_strides[0],                                   # data for the first conv layer
            padding='same',
            name=f'decoder_conv_trans_layer_{self._num_conv_layers}'
        )
        x = conv_transpose_layer(x)
        output_layer = Activation(activation='sigmoid', name='sigmoid_layer')(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')

class VAE:
    '''
    VAE represents a Deep Convolutional VAE architecture with mirrored encoder and decoder components.
    '''

    def __init__(self, 
                 input_shape,       # [width, height, num_channels] -> spectrograms can be interpreted as grayscale images
                 conv_filters,      # no of filters for each layer, list or tuple
                 conv_kernels,      # no of kernels for each layer, list or tuple
                 conv_strides,      # no of strides for each layer, list or tuple
                 latent_space_dim   # dimensionality of the bottleneck, int
                 ):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self.model = None           # tensorflow model for the whole model

        self._num_conv_layers = len(self.conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=.0001):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer, 
            loss=self._calculate_combined_loss,
            # metrics=[self._calculate_reconstruction_loss, self._calculate_kl_loss]
        )

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(
            x=x_train,          # training data
            y=x_train,          # output of model should be the same as the training data
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True
        )

    def reconstruct(self, images):
        latent_reps = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_reps)
        return reconstructed_images, latent_reps

    def save(self, filepath='.'):
        self._create_nonexistant_filepath(filepath)
        self._save_parameters(filepath)
        self._save_weights(filepath)

    def _create_nonexistant_filepath(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def _save_parameters(self, filepath):
        parameters = [
            self.input_shape,               # [width, height, num_channels] -> spectrograms can be interpreted as grayscale images
            self.conv_filters,              # no of filters for each layer, list or tuple
            self.conv_kernels,              # no of kernels for each layer, list or tuple
            self.conv_strides,              # no of strides for each layer, list or tuple
            self.latent_space_dim,          # dimensionality of the bottleneck, int
        ]
        savepath = os.path.join(filepath, 'parameters.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(parameters, f)

    def _save_weights(self, filepath):
        savepath = os.path.join(filepath, 'weights.h5')
        self.model.save_weights(savepath)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, filepath='.'):
        parameters_path = os.path.join(filepath, 'parameters.pkl')
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(filepath, 'weights.h5')
        autoencoder.load_weights(weights_path)
        return autoencoder
    
    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss
    
    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1,2,3])
        return reconstruction_loss
    
    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        return kl_loss

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')
    
    def _add_conv_layers(self, encoder_input):
        '''Creates all convolutional blocks in the encoder.'''
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)    # getting back a graph of layers with an added node
        return x
    
    def _add_conv_layer(self, layer_index, x):
        '''Adds a convolutional block to a graph of layers consisting of (2D Convolution + ReLU + batch normalization)'''
        layer_number = layer_index+1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],         # fetch the number of filters at `layer_index`
            kernel_size=self.conv_kernels[layer_index],     # fetch the kernel size at `layer_index`
            strides=self.conv_strides[layer_index],         # fetch the number of strides at `layer_index` 
            padding='same',
            name=f'encoder_conv_layer_{layer_number}'
        )

        x = conv_layer(x)                                              # drawing an arrow from x to the `conv_layer`
        x = ReLU(name=f'encoder_relu_{layer_number}')(x)               # activation function
        x = BatchNormalization(name=f'encoder_bn_{layer_number}')(x)   # batch normalization

        return x
    
    def _add_bottleneck(self, x):
        '''Flatten data and add bottleneck with gaussian sampling (dense layer).'''
        self._shape_before_bottleneck = K.int_shape(x)[1:]                 # [batch_size, width, height, num_channels] -> [width, height, num_channels]
        x = Flatten()(x)
        self.mu = Dense(units=self.latent_space_dim, name='mu')(x)                                                  # Dense layer for mean
        self.log_variance = Dense(units=self.latent_space_dim, name='log_variance')(x)                              # Dense layer for variance
        
        def sample_point_from_gaussian(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
            sampled_point = mu + K.exp(log_variance/2)*epsilon
            return sampled_point

        x = Lambda(sample_point_from_gaussian, name='encoder_output')([self.mu, self.log_variance])                 # samples points from gaussian
        return x
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name='decoder_input')
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)               # [width, height, num_channels] -> width*height*num_channels = num_neurons
        dense_layer = Dense(units=num_neurons, name='decoder_dense')(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(target_shape=self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        '''Add conv. transpose blocks.'''
        for layer_index in reversed(range(1, self._num_conv_layers)):       # loop through all the conv. layers in reverse order and stop at the first layer.
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers-layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding='same',
            name=f'decoder_conv_trans_layer_{layer_number}'
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f'decoder_relu_{layer_number}')(x)
        x = BatchNormalization(name=f'decoder_bn_{layer_number}')(x)
        return x
    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,                                                      # [width, height, num_channels=1] => spectrogram/grayscale
            kernel_size=self.conv_kernels[0],                               # data for the first conv layer
            strides=self.conv_strides[0],                                   # data for the first conv layer
            padding='same',
            name=f'decoder_conv_trans_layer_{self._num_conv_layers}'
        )
        x = conv_transpose_layer(x)
        output_layer = Activation(activation='sigmoid', name='sigmoid_layer')(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')

if __name__=='__main__':
    autoencoder = VAE(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2
    )
    autoencoder.summary()

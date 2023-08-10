from keras import Model
from keras import backend as K
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense

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

        self._build()

    def summary(self):
        self.encoder.summary()

    def _build(self):
        self._build_encoder()
        # self._build_decoder()
        # self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
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
        x = Dense(self.latent_space_dim, name='encoder_output')(x)
        return x
    
if __name__=='__main__':
    autoencoder = Autoencoder(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2
    )
    autoencoder.summary()

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Reshape
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense
from tensorflow.keras.layers import Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
import pickle
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard


class AutoEncoder:
    """
    Deep Convolutional AutoEncoder with symmetric encoder and decoder
    """
    def __init__(self,
                 input_shape,
                 num_filters,
                 kernel_size,
                 strides,
                 latent_dim):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = strides
        self.latent_dim = latent_dim
        self._shape_before_bottleneck = None
        self._model_input = None

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_layers = len(num_filters)

        self._build()

    def summary(self):
        # self.encoder.summary()
        # self.decoder.summary()
        self.model.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        conv_layer = Conv2D(filters=self.num_filters[layer_index],
                            kernel_size=self.kernel_size[layer_index],
                            strides=self.stride[layer_index],
                            padding="same",
                            name=f"encoder_conv_layer_{layer_index+1}")
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_index+1}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_index+1}")(x)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        fc_layer = Dense(self.latent_dim, name="encoder_output")
        x = fc_layer(x)
        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        fc_layer = self._add_fc_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(fc_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_dim, name="decoder_input")

    def _add_fc_layer(self, decoder_input):
        nodes = np.prod(self._shape_before_bottleneck)
        fc_layer = Dense(nodes, name="decoder_dense")(decoder_input)
        return fc_layer

    def _add_reshape_layer(self, fc_layer):
        return Reshape(self._shape_before_bottleneck)(fc_layer)

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self._num_layers)):
            x = self._add_conv_transpose_layer(x, layer_index)
        return x

    def _add_conv_transpose_layer(self, x, layer_index):
        layer_num = self._num_layers-layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.num_filters[layer_index],
            kernel_size=self.kernel_size[layer_index],
            strides=self.stride[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.kernel_size[0],
            strides=self.stride[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def compile(self, lr=1e-3):
        optimizer = Adam(learning_rate=lr)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs, val_data):
        log_dir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       validation_data=(val_data, val_data),
                       callbacks=[tensorboard_callback],
                       shuffle=True)

    def save(self, path="."):
        self._create_folder(path)
        self._save_params(path)
        self._save_weights(path)

    @staticmethod
    def _create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _save_params(self, path):
        parameters = [self.input_shape,
                      self.num_filters,
                      self.kernel_size,
                      self.stride,
                      self.latent_dim]
        save_path = os.path.join(path, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, path):
        save_path = os.path.join(path, "weights.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, path="."):
        # load params
        param_path = os.path.join(path, "parameters.pkl")
        with open(param_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = AutoEncoder(*parameters)

        # load weights
        weights_path = os.path.join(path, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, path):
        self.model.load_weights(path)


if __name__ == "__main__":
    autoencoder = AutoEncoder(
        input_shape=(28,28,1),
        num_filters=(32, 64, 64, 64),
        kernel_size=(3,3,3,3),
        strides=(1,2,2,1),
        latent_dim=2
    )
    autoencoder.summary()

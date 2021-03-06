from model import VariationalAutoEncoder
from tensorflow.keras.datasets import mnist

LR = 1e-4
BS = 128
EPCH = 10

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize dataset
    x_train = x_train.astype("float32") / 255.
    # add channel dimension
    x_train = x_train.reshape(x_train.shape + (1,))

    x_test = x_test.astype("float32") / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, x_test, lr, batch_size, epochs):
    autoencoder = VariationalAutoEncoder(
        input_shape=(28, 28, 1),
        num_filters=(32, 64, 64, 64),
        kernel_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        latent_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(lr)
    autoencoder.train(x_train, batch_size, epochs, val_data=x_test)
    return autoencoder


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    autoencoder = train(x_train, x_test, LR, BS, EPCH)
    # save trained model
    model_name = "./model_large"
    autoencoder.save(model_name)
    # load saved model
    # autoencoder2 = AutoEncoder.load("/model")
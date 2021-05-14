from model import AutoEncoder
from tensorflow.keras.datasets import mnist

LR = 1e-4
BS = 64
EPS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize dataset
    x_train = x_train.astype("float32") / 255.
    # add channel dimension
    x_train = x_train.reshape(x_train.shape + (1,))

    x_test = x_test.astype("float32") / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, lr, batch_size, epochs):
    autoencoder = AutoEncoder(
        input_shape=(28, 28, 1),
        num_filters=(32, 64, 64, 64),
        kernel_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        latent_dim=128
    )
    # autoencoder.summary()
    autoencoder.compile(lr)
    autoencoder.train(x_train, batch_size, epochs, val_data=x_test)
    return autoencoder


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    autoencoder = train(x_train, LR, BS, EPS)
    # save trained model
    autoencoder.save("~/Data/model")
    # load saved model
    # autoencoder2 = AutoEncoder.load("~/Data/model")
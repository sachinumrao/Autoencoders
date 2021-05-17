import numpy as np
import matplotlib.pyplot as plt

from model import VariationalAutoEncoder
from train import load_mnist

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15,3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i+1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i+num_images+1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()
    
    
def plot_images_latent(latent_representations, sample_labels):
    plt.figure(figsize=(10,10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()
    

if __name__ == "__main__":
    
    # load trained model
    autoencoder = VariationalAutoEncoder.load("./model_large")
    
    # load mnist dataset
    x_trai, y_train, x_test, y_test = load_mnist()
    
    # sampple images from test set
    num_sample_img = 16
    sample_imgs, sample_labels = select_images(x_test, y_test, num_sample_img)
    
    # get reconstructed images from model
    recon_imgs, latent_reps = autoencoder.reconstruct(sample_imgs)
    
    # plot original and reconstructed images
    plot_reconstructed_images(sample_imgs, recon_imgs)
    
    # get data in 2-d space
    # plot_images_latent(latent_reps, sample_labels)
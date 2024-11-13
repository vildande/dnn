import struct
import numpy as np
import os

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        # Read header (first 8 bytes as two 4-byte integers in big-endian format.)
        # first number - magic. 
        #   In our dataset, should be 2049 for the labels
        # second number - number of labels.
        #   In our dataset, should be 60000 for training and 10000 for testing
        magic, num_labels = struct.unpack(">II", f.read(8))  
        assert magic == 2049, f"Magic number mismatch in labels file: {magic}"

        # Read the rest of the file as unsigned 8-bit integers, which represent our labels.
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

def load_images(filepath):
    with open(filepath, 'rb') as f:
        # Read header (first 16 bytes as four 4-byte integers in big-endian format.)
        # first number - magic.
        #   In our dataset, should be 2051 for the images
        # second number - number of images.
        #   In our dataset, should be 60000 for training and 10000 for testing
        # third number - number of rows.
        #   In our dataset, this is 28 (since each MNIST image is 28x28 pixels)
        # fourth number - number of columns.
        #   In our dataset, this is also 28 (since each MNIST image is 28x28 pixels)
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))  # Read header
        assert magic == 2051, f"Magic number mismatch in images file: {magic}"


        # Read the rest of the file as unsigned 8-bit integers, which represent all the pixel values.
        # And separate integers by groups of rows*cols, each group will represent an image in form of 1D array of pixels
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    return images


def save_data(data, filename, path='./'):
    np.save(os.path.join(path, filename + '.npy'), data)
    print(f"Data saved to {filename}.npy")

# check  if .npy file exists and load, otherwise load from .idx file and save
def load_or_save_data(load_func, filepath, save_filename, path='./'):
    npy_path = os.path.join(path, save_filename + '.npy')
    if os.path.exists(npy_path):
        print(f"Loading {save_filename} from {npy_path}")
        return np.load(npy_path)
    else:
        print(f"{save_filename}.npy not found. Loading from {filepath}")
        data = load_func(filepath)
        save_data(data, save_filename, path)
        return data



if __name__ == '__main__':
    path = './dataset/'

    train_images = load_images_normalized(path + 'train-images.idx3-ubyte')
    train_labels = load_labels(path + 'train-labels.idx1-ubyte')
    test_images = load_images_normalized(path + 't10k-images.idx3-ubyte')
    test_labels = load_labels(path + 't10k-labels.idx1-ubyte')



    import matplotlib.pyplot as plt

    # Display a few images and their labels
    def display_samples(images, labels, num_samples=3):
        for i in range(num_samples):
            plt.imshow(images[i].reshape(28, 28), cmap='gray')
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
            plt.show()

    display_samples(train_images, train_labels)

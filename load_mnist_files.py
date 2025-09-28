from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

(x_train, y_train), _ = mnist.load_data()
os.makedirs('mnist_images', exist_ok=True)

image_index = 8  # Index of the image to download
image = x_train[image_index]
label = y_train[image_index]
plt.imsave(f'mnist_images/digit_{label}_index_{image_index}.png', image, cmap='gray')
print(f"Image 'digit_{label}_index_{image_index}.png' saved successfully in 'mnist_images' directory.")
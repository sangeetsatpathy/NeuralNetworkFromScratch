from neural_net import MNIST_NeuralNet
import cv2
import numpy as np

filename = input("Enter the filename to load the weights for: ")

image_name = input("Enter the filename for the Image: ")


valid_activ = False
activation_function = "tanh"
while(not valid_activ):
    activation_function = input("Enter the activation function: ")
    if(activation_function == "sigmoid" or activation_function=="tanh" or activation_function=="relu"):
        valid_activ = True
    else:
        print("Activation function must be sigmoid, tanh, or relu. Try again.")

learning_rate = float(input("Enter the learning rate (numerical): "))

img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE) / 255.0
resized_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imshow("img", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

net = MNIST_NeuralNet(filename, activation_function=activation_function, lr=learning_rate)

pred, classification = net.image_inference(resized_img)

print(f"This image has been classified as a {classification}!")

# NeuralNetworkFromScratch

A fully connected neural network built entirely from scratch (no external neural network libraries) to classify handwritten digits (MNIST). I used `numpy` only for the random initialization of weights, used `ast` to efficiently read my exported weight files, and used `tensorflow` only to access the MNIST dataset pictures. This project displays my understanding of low-level neural network internals — forward propagation, backpropagation, activation functions, batch-normalization, weight updates — and my ability to make it work end-to-end.

Throughout the process, I was debugging errors (both compiler and conceptual errors) in the Neural Network, discovering necessary architectural elements along the way, allowing me to get a deep understanding of the mathematics behind neural nets. 

For example, to solve the vanishing gradient problem inherent in sigmoid activation functions, I learned to use batch normalization -- helping my pre-activation values stabilize. This included figuring out how to implement batch normalization for inference. In debugging why my weights were not effectively "learning" feature recognition, I realized that equal weight initialization was hindering the Neural Net's efficacy. In debugging why my test accuracies were decreasing across epochs, I decreased my learning rate from 0.1 to 0.001. Upon looking into why my prediction probabilities were lacking confidence, I realized that I was feeding last-layer logits into activation functions before softmaxing them, killing Neural Net confidence. With these learnings, along with many more, I was able to finally acquire test accuracies of 83% with only one hidden layer (of size 30)!

---

## 🚀 Project Overview

In this project, I implemented a neural network from first principles using only basic Python (no TensorFlow, PyTorch, Keras, etc.). The network is trained to classify digits from the MNIST dataset. Key goals:

- Demonstrate core machine learning and neural network mechanics manually  
- Build a pipeline for training, inference, and weight serialization  
- Experiment with activation functions (ReLU, Tanh), architectures, and hyperparameters  
- Provide clear interpretable code and results, not relying on “black box” frameworks  

This work signals to recruiters that you deeply understand how neural networks operate beneath the high-level APIs.

---

## 📂 Repository Structure

NeuralNetworkFromScratch/<br>
├── mnist_images/ # Raw MNIST image files (train/test)<br>
├── load_mnist_files.py # Script to download MNIST file images, one at a time.<br>
├── neural_net.py # Core implementation of the neural network (forward feed, backpropagation, training, and examining test accuracy over epochs.) <br>
├── run-inference.py # Script to use trained weights for prediction / evaluation <br>
├── relu-weights.txt # Example dumped weights from a run (ReLU activation) <br>
├── tanh-weights.txt # Example dumped weights from a run (Tanh activation) <br>


## How to Use
To run inference on a specific image, using pre-trained weights: run `python run-inference.py` (with all required dependencies installed). 
You will be prompted to enter the path for the image and the pre-trained weight files. Note that these weight files must be generated from my `neural_net.py` script, as it is in a specific format. You will also be prompted to enter an activation function and a learning rate (as seen below).
<img width="522" height="87" alt="Screenshot 2025-09-27 at 7 48 23 PM" src="https://github.com/user-attachments/assets/d1084703-6ce5-4e71-ac91-08900034ef86" />

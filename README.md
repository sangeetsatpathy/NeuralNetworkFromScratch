# NeuralNetworkFromScratch

A fully connected neural network built entirely from scratch (no external neural network libraries) to classify handwritten digits (MNIST). I used `numpy` only for the random initialization of weights, used `ast` to efficiently read my exported weight files, and used `tensorflow` only to access the MNIST dataset pictures. This project displays my understanding of low-level neural network internals — forward propagation, backpropagation, activation functions, batch-normalization, weight updates — and my ability to make it work end-to-end.

Throughout the process, I was debugging errors (both compiler and conceptual errors) in the Neural Network, discovering necessary architectural elements along the way, allowing me to get a deep understanding of the mathematics behind neural nets. 

For example, to solve the vanishing gradient problem inherent in sigmoid activation functions, I learned to use batch normalization -- helping my pre-activation values stabilize. This included figuring out how to implement batch normalization for inference. In debugging why my weights were not effectively "learning" feature recognition, I realized that equal weight initialization was hindering the Neural Net's efficacy. After setting up random weight initialization, I realized my pre-activation variances were exploding; so I set up Xavier initialization.
In debugging why my test accuracies were decreasing across epochs, I decreased my learning rate from 0.1 to 0.001. Upon looking into why my prediction probabilities were lacking confidence, I realized that I was feeding last-layer logits into activation functions before softmaxing them, killing Neural Net confidence. With these learnings, along with many more, I was able to finally acquire test accuracies of 83% with only one hidden layer (of size 30)!

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
To <b>run inference</b> on a specific image, using pre-trained weights: run `python run-inference.py` (with all required dependencies installed). 
You will be prompted to enter the path for the image and the pre-trained weight files. Note that these weight files must be generated from my `neural_net.py` script, as it is in a specific format. You will also be prompted to enter an activation function and a learning rate (as seen below).<br>
<img width="522" height="87" alt="Screenshot 2025-09-27 at 7 48 23 PM" src="https://github.com/user-attachments/assets/d1084703-6ce5-4e71-ac91-08900034ef86" />
<br> For reference, this was the image that was classified as a 0 above:
<img width="28" height="28" alt="img_a" src="https://github.com/user-attachments/assets/59f50669-84ca-4c9d-a442-9c9cf325f9a3" />

<br>
To <b>train the model</b>, run `python train-net.py`. This will prompt you for the number of epochs, activation function, and learning rate. Then, it will begin training each epoch. At the end of every epoch, it will print out the test accuracy of the model, and output a file with the weights (to be used for inference, if desired). Every batch within an epoch will print out a '|' character to let you know if the training is progressing. Training several epochs will take several hours, depending on your compute power.

<img width="522" height="87" alt="Screenshot 2025-09-27 at 7 58 07 PM" src="https://github.com/user-attachments/assets/d02edfcb-1867-4e78-84cf-ff84f5029c38" />


## More details on the internal workings:
This is the heart of the project. Key components:

- **Network Initialization**  
  - Random weight initialization (using np.random, Xavier initialization)  
  - Bias initialization  = 0
  - Configurable layer architecture (input layer, hidden layers, output layer)  

- **Forward Propagation**  
  - Linear combinations ( \( z = W x + b \) )  
  - Activation functions: ReLU, Tanh, Sigmoid (with corresponding derivatives)  
  - Softmax or other output activation for multi­class classification  

- **Loss Computation**  
  - Cross-entropy loss (categorical)  
  - Handling numerical stability (e.g. log-sum-exp)  

- **Backpropagation / Gradient Computation**  
  - Derivatives of loss → output layer  
  - Backpropagate through activation functions and linear layers  
  - Compute gradients w.r.t weights and biases  

- **Weight Updates / Training Loop**  
  - Gradient descent (or optionally mini-batch)  
  - Learning rate scheduling, epoch loops  
  - Tracking loss over epochs, possibly early stopping  

- **Utilities / Helpers**  
  - Shuffling training data  
  - Batch splitting  
  - Metrics (accuracy)  

**Skills shown**: Fundamental ML/NN internals, calculus in code, algorithmic thinking, careful numerical implementation.


## Test Results
Along my process of creating the Neural Network, I collected data. In a run, when I specify that I added something to the program, this change continues down for all the future runs.
Here is the context behind each run:
  * Run 1: Sigmoid, LR = 0.1
  * Run 2: Sigmoid, LR = 0.01
  * Run 3: Added gradient clipping to the program; Sigmoid, LR = 0.01
  * Run 4: Sigmoid, LR = 0.001
  * Run 5: Removed the activation function and batch normalization in last layer; Sigmoid, LR = 0.001
  * Run 6: ReLU, LR = 0.001
  * Run 7: TanH, LR = 0.001

This first graph compares Run 1 and Run 2. We see that decreasing the learning rate from 0.1 to 0.01 significantly increased test accuracy. We do see that both learning rates are overfitting (shown by the downward sloping lines). <br>
<img width="550" height="341" alt="(A) Test Accuracies Across Learning Rates" src="https://github.com/user-attachments/assets/177de977-44db-4073-9f87-3ce206a36ea0" />

Comparing Run 2 and Run 3, we see that implementing gradient clipping only served to reduce test accuracy; meaning that we didn't have any exploding gradients present.<br>
<img width="600" height="371" alt="Effect of Gradient Clipping on Accuracy (LR = 0 001, sigmoid)" src="https://github.com/user-attachments/assets/5d0f8154-0cc0-43a1-81dc-98a2d8bd7de0" />


After implementing gradient clipping, comparing LR = 0.01 and LR = 0.001(Run 3 vs. Run 4) shows us that LR = 0.001 fixes the overfitting problem, providing a higher test accuracy as well.<br>
<img width="600" height="371" alt="(B) Test Accuracy with Gradient Clipping" src="https://github.com/user-attachments/assets/f3636026-2eb9-4709-98f4-16b4dffdabdc" />

Measuring test accuracy after removing last-layer activations (comparing Run 4 and Run 5), we gain about 4% test accuracy; which is a lesser gain in test accuracy than expected from the magnitude of the problem. However, we see that this removal allows the model to learn better, even past epoch 2. <br>
<img width="661" height="371" alt="(C) Impact of Last-Layer Activation (LLA) on Accuracy (LR=0 001)" src="https://github.com/user-attachments/assets/83acd6fe-8fef-4020-b776-383da3ce17b8" />

After removing last-layer activations, I compared test accuracies across different activation functions. We see that ReLU performs fairly poorly across epochs, failing to learn more than 74% test accuracy. The sigmoid activation function's test accuracy increases across the 6 epochs. The TanH activation function provides the best test accuracy, at 83%.<br>
<img width="600" height="371" alt="(D) Test Accuracies across Activation Functions (LR=1e-3)" src="https://github.com/user-attachments/assets/6d4711d2-d901-44ec-bf02-d969917aa03a" />

However, test accuracy isn't always the only metric we want to measure. The confidence of a model's predictions also conveys information. Ideally, we want a model that has a high accuracy and a high confidence. When the model has low accuracy, however, we don't want it to have a high confidence. If a model always just "barely" gets a majority prediction value for the right answer, we don't trust it as much. So for each run, I calculated a <b>Confidence Score</b>: I took the last epoch's predictions on one image, summed the squared deviations from 0.10, and took the square root. <br>
From the graph below, we see that before fixing the Last-Layer Activations, the confidence was pretty low, and stayed relatively the same. It then increased a bit after fixing the Last-Layer Activation (for sigmoid). However, between the activation functions, ReLU and TanH seem to be much more confident than sigmoid. ReLU almost seems TOO confident...
<img width="600" height="371" alt="Confidence Scores over Different Runs" src="https://github.com/user-attachments/assets/b12ba8c6-48b2-433f-a3b4-4f52a4669c36" />

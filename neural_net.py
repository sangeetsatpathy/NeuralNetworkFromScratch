import math
import numpy # used for initializing random weights.
import tensorflow as tf # used to load the MNIST dataset training and testing images.
import sys
import ast # used to read in tensors from a text file and automatically convert it to floats.

def activation(x, func="sigmoid"):
    if func == "sigmoid":
        return 1.0/(1.0 + pow(math.e, -1 * x))
    if func == "tanh":
        a = pow(math.e, x)
        b = pow(math.e, -1 * x)
        return (a - b) / (a + b)
    if func == "relu":
        if x < 0:
            return 0
        return x

def softmax(list_nodes):
    exp_list = [pow(math.e, i.get_val()) for i in list_nodes]
    total_sum = sum(exp_list)
    return_val = [(k / total_sum) for k in exp_list]
    return return_val

class Node:
    def __init__(self, prev_layer, act_function="sigmoid"):
        # randomly instantiate weights
        self.weights = []
        self.weights_gradient = []
        self.val = 0.0
        self.bias = 0.0
        self.prev_layer = prev_layer
        self.prev_nodes = prev_layer.get_nodes()
        scaled_std_weights = math.sqrt(1/len(self.prev_nodes))
        for i in range(len(self.prev_nodes)):
            self.weights.append(numpy.random.normal(0, scaled_std_weights)) # sample the weights.
            self.weights_gradient.append(0.0)
        self.bias_gradient = 0.0

        self.avg_mean = 0.0
        self.avg_std = 0.0
        self.activation_function = act_function
    
    def process_net_input(self): # Pre-activation processing
        sum_vals = self.bias
        for n, neuron in enumerate(self.prev_nodes):
            sum_vals += neuron.get_val() * self.weights[n] # add up all the products
        self.val = sum_vals #set value to the net input, for now (till activation and batchnorm)
        return sum_vals

    def update_val(self, mean, stdev): # Batch Normalizes and Activates the pre-activations for each node.
        scaled_val = ((self.val - mean) / stdev)
        post_sigmoid = activation(scaled_val, self.activation_function) 
        self.val = post_sigmoid

    def update_val_inference(self):
        self.update_val(self.avg_mean, self.avg_std)

    def get_val(self):
        return self.val
    def get_weights(self):
        return [float(i) for i in self.weights]
    def set_weights(self, new_weights):
        self.weights = new_weights
    def set_bias(self, new_bias):
        self.bias = new_bias
    def get_bias(self):
        return float(self.bias)
    def get_prev_layer(self):
        return self.prev_layer
    def set_avg_mean(self, new_avg_mean):
        self.avg_mean = new_avg_mean
    def set_avg_std(self, new_avg_std):
        self.avg_std = new_avg_std
    def get_gradients(self):
        return self.weights_gradient, self.bias_gradient
    def get_avg_mean(self):
        return float(self.avg_mean)
    def get_avg_std(self):
        return float(self.avg_std)
    def set_gradients(self, weight_grad, bias_grad):
        for g in range(len(weight_grad)):
            self.weights_gradient[g] += weight_grad[g]
        self.bias_gradient += bias_grad

    def end_batch(self): #updates the weights and biases based on the gradient values.
        avg_weight_grad = sum(self.weights_gradient) / len(self.weights_gradient)
        avg_weight = sum(self.weights) / len(self.weights)
        
        for w in range(len(self.weights_gradient)):
            
            self.weights[w] -= (self.weights_gradient[w]) * MNIST_NeuralNet.LEARNING_RATE
            self.weights_gradient[w] = 0 #resetting for next batch
        
        self.bias -= self.bias_gradient * MNIST_NeuralNet.LEARNING_RATE
        self.bias_gradient = 0

class InputNode:
    def __init__(self):
        self.val = 0.0    
    def set_val(self, new_val):
        self.val = new_val
    def get_val(self):
        return self.val


class DenseLayer:
    def __init__(self, size, prev_layer, activation_function="sigmoid"):
        self.nodes = [Node(prev_layer, activation_function) for i in range(size)] # Create a new node and add it to the list
    def get_nodes(self):
        return self.nodes
   
    def process_net_input(self):
        net_inputs = []
        for n in self.nodes:
            net_input = n.process_net_input()
            net_inputs.append(net_input)
        return net_inputs

    def activate(self, mean, std):
        for n in self.nodes:
            n.update_val(mean, std)
    
    def run_inference(self):
        for n in self.nodes:
            inp = n.process_net_input()
            n.update_val_inference()

    def is_input_layer(self):
        return False
    

class InputLayer:
    def __init__(self, size):
        self.nodes = [InputNode() for i in range(size)]
    def get_nodes(self):
        return self.nodes
    def set_node_val(self, index, new_val):
        self.nodes[index].set_val(new_val)
    def is_input_layer(self):
        return True
    def print_input_nodes(self):
        for n in self.nodes:
            print(n.get_val(), end = ' ')

class MNIST_NeuralNet:
    ACTIVATION_FUNCTION = "tanh"
    LEARNING_RATE = 0.001
    def __init__(self, input_filename=None, activation_function = "tanh", lr=0.001):
        self.layers = [InputLayer(784)]
        self.layers.append(DenseLayer(30, self.layers[-1], MNIST_NeuralNet.ACTIVATION_FUNCTION))
        self.layers.append(DenseLayer(10, self.layers[-1], MNIST_NeuralNet.ACTIVATION_FUNCTION)) # output layer

        MNIST_NeuralNet.ACTIVATION_FUNCTION = activation_function
        MNIST_NeuralNet.LEARNING_RATE = lr

        if input_filename:
            self.process_file(input_filename)

    def process_file(self, filename):
        f = open(filename, "r")
        tensor_str = f.read()
        f.close()

        tensor = ast.literal_eval(tensor_str)
        for l in range(len(tensor)):
            current_layer = self.layers[l+1]
            for n in range(len(tensor[l])):
                current_node = current_layer.get_nodes()[n]
                current_node.set_weights(tensor[l][n][:-3])
                current_node.set_bias(tensor[l][n][-3])
                current_node.set_avg_mean(tensor[l][n][-2])
                current_node.set_avg_std(tensor[l][n][-1])
        


    def get_layers(self):
        return self.layers
    def read_prediction(self):
        output_nodes = self.layers[-1].get_nodes()
        probabilities = softmax(output_nodes)
        return probabilities
    
    def end_batch_net(self): # call end batch for each non-input node in the NeuralNet.
        for l in self.layers[1:]:
            for n in l.get_nodes():
                    n.end_batch()

    def backpropagate_img(self, correct_classification): # recurse the gradient backwards to each node, chain-ruling along each path.
        probabilities = self.read_prediction() 
        curr_max = 0
        prediction = -1
        for i in range(len(probabilities)):
            if probabilities[i] > curr_max:
                curr_max = probabilities[i]
                prediction = i
        for index, neur in enumerate(self.layers[-1].get_nodes()):
            if index == correct_classification: # if we are at the correct prediction node
                current_derivative = probabilities[index] - 1
            else:
                current_derivative = probabilities[index]
            self.recurse_bp(current_derivative, neur, len(self.layers))

    def recurse_bp(self, current_derivative, current_node, layer):
        # at every node, take derivative w.r.t. the weights and bias
        # for weights, just multiply by the output of the node it corresponds to.
        # for bias, the derivative is the same as current_derivative.
        if layer == 1:
            return
        previous_layer = current_node.get_prev_layer()
        previous_layer_nodes = previous_layer.get_nodes()
        weights = current_node.get_weights()
        if layer != len(self.layers):
            if MNIST_NeuralNet.ACTIVATION_FUNCTION == "sigmoid":
                current_derivative = ((1 - current_node.get_val()) * current_node.get_val()) * current_derivative # SIGMOID
            if MNIST_NeuralNet.ACTIVATION_FUNCTION == "tanh":
                current_derivative = (1 - pow(current_node.get_val(), 2)) * current_derivative #Tanh
            if MNIST_NeuralNet.ACTIVATION_FUNCTION == "relu":
                if current_node.get_val() == 0:
                    current_derivative = 0
                # otherwise, the current_derivative is just multiplied by 1; so it stays the same.
        weight_gradients = [current_derivative * neuron.get_val() for neuron in previous_layer_nodes]
        bias_gradient = current_derivative
        current_node.set_gradients(weight_gradients, bias_gradient) 
        
        # continue recursing down
        for n, node in enumerate(previous_layer_nodes):
            self.recurse_bp(current_derivative * weights[n], node, layer - 1) 
            # chain rule: derivative of function w.r.t. a neuron in the prev layer is the weight assigned to that neuron

    def copyNeuralNet(self):
        copy = MNIST_NeuralNet()
        copy_layers = copy.get_layers()
        for l in range(1, len(copy_layers)):
            curr_layer_nodes = self.layers[l].get_nodes()
            for n in range(len(curr_layer_nodes)):
                # copy the weights and biases of the current nodes into the copy's nodes
                copy_layers[l].get_nodes()[n].set_weights(curr_layer_nodes[n].get_weights())
                copy_layers[l].get_nodes()[n].set_bias(curr_layer_nodes[n].get_bias())
        return copy

    def feed_input(self, image):
        input_layer = self.layers[0]
        ctr = 0
        for i in image:
            for j in i:
                input_layer.set_node_val(ctr, j)# each pixel is an input
                ctr+=1

    def classify(self, predictions):
        max_val = -1
        classification = -1
        for i, value in enumerate(predictions):
            if value >= max_val:
                max_val = value
                classification = i
        return classification

    def image_inference(self, img):
        self.feed_input(img)
        for l in range(1, len(self.layers)):
            self.layers[l].run_inference()
        preds = self.read_prediction()
        output = self.classify(preds)
        return preds, output

    def export_nn(self, filename):
        # Encapsulates the weights of each of the nodes, and biases. Each layer is a 2D matrix.
        tensor = []
        for l in range(1, len(self.layers)):
            curr_layer = []
            for n in self.layers[l].get_nodes():
                node_params = n.get_weights()
                node_params.append(n.get_bias())
                node_params.append(n.get_avg_mean())
                node_params.append(n.get_avg_std())
                curr_layer.append(node_params)
            tensor.append(curr_layer)

        with open(filename, 'w') as f:
            f.write(str(tensor))

    def normalize_gradients(self):
        THRESHOLD = 5.0
        sum_gradients = 0
        # ad all of the weights together, square them, then square root
        for layer in self.layers[1:]:
            for node in layer.get_nodes():
                curr_weight_grads, curr_bias_grad = node.get_gradients()
                for i in curr_weight_grads:
                    sum_gradients += pow(i, 2)
                sum_gradients += pow(curr_bias_grad, 2)
        norm = math.sqrt(sum_gradients)
        if norm > THRESHOLD:
            scalar = THRESHOLD / norm
        else:
            scalar = 1.0

        for layer in self.layers[1:]:
            for node in layer.get_nodes():
                curr_weights_grads, curr_bias_grad = node.get_gradients()
                new_weights_grads = [g * scalar for g in curr_weights_grads]
                new_bias_grad = curr_bias_grad * scalar
                node.set_gradients(new_weights_grads, new_bias_grad)

def load_mnist():
    #total train size: 60,000
    BATCH_SIZE = 50
    num_runs = int(60000 / BATCH_SIZE)
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Split into batches.
    batches = []
    for i in range(num_runs):
        start_index = i * BATCH_SIZE
        end_index = (i+1) * BATCH_SIZE
        curr_batch = [(x_train[k] / 255.0, y_train[k]) for k in range(start_index, end_index)] #tuples paired up
        batches.append(curr_batch)
    return batches

def std(array, mean):
    num_elems = len(array)
    sum_residuals = 0
    for i in array:
        sum_residuals += pow(i - mean, 2.0)
    return math.sqrt(sum_residuals / (num_elems - 1))

def train_epoch(last_nn = None):
    if last_nn is None:
        nn = MNIST_NeuralNet()
    else:
        nn = last_nn
    batches = load_mnist()
    
    means_layers = [[ [] for n in k.get_nodes()] for k in nn.get_layers()] # add the mean of each batch into the corresponding layer
    stds_layers = [[ [] for n in k.get_nodes()] for k in nn.get_layers()] # add the std of each batch into the corresponding layer
    ctr = 1
    for b in batches: # b is each batch
        print("|", end="")
        sys.stdout.flush()
        nn_copies = [nn.copyNeuralNet() for k in range(len(b))]

        for i, img in enumerate(b):
            nn_copies[i].feed_input(img[0])
        
        for l in range(1, len(nn.get_layers())):
            for n in range(len(nn.get_layers()[l].get_nodes())):
                node_net_inputs = []
                for i, img in enumerate(b): # within each layer, feed each image forward.
                    net_input = nn_copies[i].get_layers()[l].get_nodes()[n].process_net_input()
                    node_net_inputs.append(net_input) # add each image's inputs for that node
                
                # calculate the mean and stdev of net inputs in one batch.
                net_inp_mean = sum(node_net_inputs) / len(node_net_inputs)
                net_inp_std = std(node_net_inputs, net_inp_mean) # SHOULD be 0.

                means_layers[l][n].append(net_inp_mean)
                stds_layers[l][n].append(net_inp_std)


                if l != len(nn.get_layers()) - 1: # only do the activation function if not on last layer.
                    for i, img in enumerate(b):
                        nn_copies[i].get_layers()[l].get_nodes()[n].update_val(net_inp_mean, net_inp_std) # update values of nodes w/ activation function

        # after each neural net has been fully evaluated, backpropagate each one individually.
        for i, img in enumerate(b):
            nn_copies[i].backpropagate_img(img[1]) # feed in the correct classification

        #Average all of the weight gradients 
        for l in range(1, len(nn.get_layers())):
            for n in range(len(nn.get_layers()[l].get_nodes())):
                #for each node, average across the different nets.
                sum_gradients_grad = [0 for i in nn.get_layers()[l].get_nodes()[n].get_weights()]
                sum_bias_grad = 0
                
                for net in nn_copies:
                    curr_node = net.get_layers()[l].get_nodes()[n]
                    curr_weight_grad, curr_bias_grad = curr_node.get_gradients()
                    sum_gradients_grad = [sum_gradients_grad[index] + curr_weight_grad[index] for index in range(len(curr_weight_grad))]
                    sum_bias_grad += curr_bias_grad
                
                avg_weights_grad = [weight / len(nn_copies) for weight in sum_gradients_grad]
                avg_bias_grad = sum_bias_grad / len(nn_copies)

                nn_copies[0].get_layers()[l].get_nodes()[n].set_gradients(avg_weights_grad, avg_bias_grad)
        nn_copies[0].normalize_gradients()
        nn_copies[0].end_batch_net()
        nn = nn_copies[0] # At the end, collapse copies and set nn equal to that Neural Net.
        nn_copies = []
        ctr += 1
    for l in range(1, len(nn.get_layers())):
        for n in range(len(nn.get_layers()[l].get_nodes())):
            node = nn.get_layers()[l].get_nodes()[n]
            node.set_avg_mean(means_layers[l][n][-1])
            node.set_avg_std(stds_layers[l][n][-1])
    return nn

def test_nn(neural_net):
    num_correct = 0
    num_tests = 0
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    for t in range(len(x_test)//5):
        preds, output = neural_net.image_inference(x_test[t] / 255.0)
        if output == y_test[t]:
            num_correct += 1
        num_tests += 1
    accuracy = num_correct / num_tests
    return accuracy

def main():
    num_epochs = int(input("Enter the number of epochs to train for: "))

    activation_function = "tanh"

    valid_activ = False
    while(not valid_activ):
        activation_function = input("Enter the activation function: ")
        if(activation_function == "sigmoid" or activation_function=="tanh" or activation_function=="relu"):
            valid_activ = True
        else:
            print("Activation function must be sigmoid, tanh, or relu. Try again.")

    learning_rate = float(input("Enter the learning rate (numerical): "))
    MNIST_NeuralNet.ACTIVATION_FUNCTION = activation_function
    MNIST_NeuralNet.LEARNING_RATE = learning_rate

    net = train_epoch()
    print("Finished training 0 epoch!")
    net.export_nn(f"epoch_0.txt")
    accuracy = test_nn(net)
    print(f"Epoch 0 accuracy: {accuracy}")
    for e in range(1, num_epochs - 1):
        net = train_epoch(net)
        print(f"Finished training {e} epoch!")
        net.export_nn(f"epoch_{e}.txt")
        accuracy = test_nn(net)
        print(f"Epoch {e} accuracy: {accuracy}")
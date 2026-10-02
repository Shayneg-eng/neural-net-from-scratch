import random
import math
from colorama import Fore, Style
import csv
import numpy as np


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def readCSV(file_path, line_number, answeOrData):
    with open(file_path, 'r') as csvfile:
        csvreader = list(csv.reader(csvfile))
        
        # Remove header if it exists
        if csvreader[0][0].isalpha():
            csvreader = csvreader[1:]
        
        # Adjust line number if it's out of range
        adjusted_line = (line_number - 1) % len(csvreader)
        
        row = csvreader[adjusted_line]
        
        if answeOrData == "label and data":
            return int(row[0]), [float(value) for value in row[1:]]
        elif answeOrData == "answer":
            return int(row[0])
        elif answeOrData == "data":
            return [float(value) for value in row[1:]]
        else:
            raise ValueError("Invalid return_type. Use 'answer' or 'data'.")
        
def load_all_data(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = list(csv.reader(csvfile))
        # Skip header if present
        if csvreader[0][0].isalpha():
            csvreader = csvreader[1:]
        for row in csvreader:
            label = int(row[0])
            pixels = [float(v) for v in row[1:]]
            data.append((label, pixels))
    return data
        
def calculateMSE(true, pred):
    
    if len(pred) != 10:
        print("ERROR: output isnt 10 values")
        return -1
    
    # creating the perfect output
    trueOutput = [0] * len(pred)
    trueOutput[true] = 1
    
    # Convert lists to numpy arrays
    trueOutput = np.array(trueOutput)
    pred = np.array(pred)
    
    # Check if the lists have the same length
    if len(trueOutput) != len(pred):
        raise ValueError("The two lists must have the same length.")
    
    # Calculate MSE
    mse = np.mean((trueOutput - pred) ** 2)
    
    return mse

def findHighestActivation(lastRowNetworkStructure):
    highestActivation = 0
    highestActivationIndex = 0
    index = 0
    outputActivations = []
    
    for neuron in lastRowNetworkStructure:
        activation = neuron.activation
        if activation > highestActivation:
            highestActivation = activation
            
            highestActivationIndex = index
        index += 1
        outputActivations.append(activation)
        
    return outputActivations, highestActivationIndex, highestActivation

class Neuron:
    def __init__(self, row, column):
        self.activation = 0
        self.bias = random.uniform(-1,1)
        self.row = row
        self.column = column
        
    def setActivation(self, activation: float):
        self.activation = sigmoid(activation)
        
    def setBias(self, bias: float):
        self.bias = bias
        
    def __repr__(self):
        return f"{Fore.RED}|{Style.RESET_ALL}Neuron:({self.row},{self.column}) activation={self.activation} bias={self.bias}{Fore.RED}|{Style.RESET_ALL}"
        
class Connection:
    def __init__(self, row, number):
        self.weight = random.uniform(-1,1)
        self.row = row
        self.number = number
        
    def setWeight(self, weight: float):
        self.weight = weight

    def __repr__(self):
        return f"{Fore.RED}|{Style.RESET_ALL}Connection:({self.row},{self.number}) weight={self.weight}{Fore.RED}|{Style.RESET_ALL}"
        
class Network:
    def __init__(self, rows):
        self.rows = rows
        self.networkStructure = []
        
    def create(self):
        creatingNeuronLayer = True # tracks if im making a neuron layer or a connection layer
        for row, column in enumerate(self.rows): # indexes through the rows to know how many neurons it should make
            if creatingNeuronLayer == True:
                layer = [Neuron(row, neuronIndex+1) for neuronIndex in range(column)] # column of neurons is plugged in and neurons are made with the row and column characteristics
                                              # +1 added for readability 

                self.networkStructure.append(layer)
                creatingNeuronLayer = False # setting creatingNeuronLayer to False to get ready for the next layer to be for connections
                previousNeuronLayerCount = column
            else:
                connectionNumber = 1 # Keeps track of what connection is being made. ie: how many connections down we are
                interLayer = [] # Inerlayer is the connection layer
                for connection in range(int(column/previousNeuronLayerCount)):
                    batch = [] # Seperated groups of neurons into "batches" to make it easy to index to the correct one
                    for connectionBatch in range(previousNeuronLayerCount):
                        batch.append(Connection(row, connectionNumber))
                        connectionNumber += 1
                    interLayer.append(batch)

                self.networkStructure.append(interLayer)
                creatingNeuronLayer = True
                interLayer = []
                
    def showAllValues(self):
        layer = 0
        while True:
            try:
                print(self.networkStructure[layer])
                print("--------------------------------------------------")
                layer += 1
            except:
                return
            
    def showOutputValues(self):
        for neuron in self.networkStructure[-1]:
            print(neuron.activation)

    def input(self, listOfInputValues):
        self.listOfInputValues = listOfInputValues
        
        if len(self.networkStructure[0]) == len(listOfInputValues):    
            for index, neuron in enumerate(self.networkStructure[0]):
                neuron.setActivation(listOfInputValues[index])
            
        else:
            print("ERROR: Intput value List incorrect size")
            return
        
    def forwardPropagation(self):
        layerIndex = 0
        connectionGroupNumberAndEndNeuronNumber = 0 # so the code knows how many connections to assign to what neurons
        # +1 and +2 are used to get one and two lines down the neural nework respectively
        while True:
            try:
                previousLayerNeurons = (self.networkStructure[layerIndex])
                coorispondingNeuronConnections = (self.networkStructure[layerIndex+1][connectionGroupNumberAndEndNeuronNumber])
                targetNeuron = self.networkStructure[layerIndex+2][connectionGroupNumberAndEndNeuronNumber]
                
                sum = 0 # going to be the value of the neuron in the next layer where all the connections lead
                for index in range(len(previousLayerNeurons)):
                    sum += previousLayerNeurons[index].activation * coorispondingNeuronConnections[index].weight

                targetNeuron.setActivation(sum + targetNeuron.bias)
        
                weAreAtTheEndOfTheTargetLayer = len(self.networkStructure[layerIndex+2]) == (connectionGroupNumberAndEndNeuronNumber+1)
                if weAreAtTheEndOfTheTargetLayer: # checking to see if I should index again to do the same operation on the nexnt neuron in the same row or if i should move to the next row
                    layerIndex += 2 #changing to the next neuron layer
                    connectionGroupNumberAndEndNeuronNumber = 0
                    
                else:
                    connectionGroupNumberAndEndNeuronNumber += 1
            except:
                return 1
            
    def backPropagation(self, learning_rate=0.01, batch_size=32):
        # Initialize gradient accumulators if not already present
        if not hasattr(self, 'weight_gradients'):
            self.weight_gradients = {}
            self.bias_gradients = {}
            self.batch_count = 0
        
        # Calculate output layer error
        output_layer = self.networkStructure[-1]
        target = [0] * len(output_layer)
        target[self.trueMnistlabel] = 1  # One-hot encoding of the true label
        
        # Store errors for each layer (will be calculated backwards)
        layer_errors = [None] * len(self.networkStructure)
        
        # Calculate error for output layer
        output_errors = []
        for i, neuron in enumerate(output_layer):
            # Error = (target - actual) * derivative of sigmoid
            error = (target[i] - neuron.activation) * neuron.activation * (1 - neuron.activation)
            output_errors.append(error)
        
        layer_errors[-1] = output_errors
        
        # Backpropagate error through the network (backwards)
        for layer_idx in range(len(self.networkStructure) - 3, -1, -2):  # Skip connection layers
            current_errors = []
            next_layer_idx = layer_idx + 2
            connection_layer_idx = layer_idx + 1
            
            # For each neuron in current layer
            for i, neuron in enumerate(self.networkStructure[layer_idx]):
                error_sum = 0
                
                # For each neuron in the next layer
                for j, next_neuron in enumerate(self.networkStructure[next_layer_idx]):
                    # Find the connection from this neuron to the next layer neuron
                    connection_batch = self.networkStructure[connection_layer_idx][j]
                    # Sum up the error contribution
                    error_sum += layer_errors[next_layer_idx][j] * connection_batch[i].weight
                
                # Calculate error for current neuron
                error = error_sum * neuron.activation * (1 - neuron.activation)
                current_errors.append(error)
            
            layer_errors[layer_idx] = current_errors
        
        # Accumulate gradients instead of updating immediately
        for layer_idx in range(0, len(self.networkStructure) - 2, 2):  # Process neuron layers
            next_layer_idx = layer_idx + 2
            connection_layer_idx = layer_idx + 1
            
            # Accumulate gradients for each connection
            for j, next_neuron in enumerate(self.networkStructure[next_layer_idx]):
                connection_batch = self.networkStructure[connection_layer_idx][j]
                
                # Create key for bias gradient storage
                bias_key = f"layer_{next_layer_idx}_neuron_{j}_bias"
                if bias_key not in self.bias_gradients:
                    self.bias_gradients[bias_key] = 0
                
                # Accumulate bias gradient
                self.bias_gradients[bias_key] += layer_errors[next_layer_idx][j]
                
                # Accumulate weight gradients for connections to this neuron
                for i, connection in enumerate(connection_batch):
                    # Create key for weight gradient storage
                    weight_key = f"layer_{layer_idx}_to_{next_layer_idx}_neuron_{i}_to_{j}"
                    if weight_key not in self.weight_gradients:
                        self.weight_gradients[weight_key] = 0
                    
                    # Gradient = error * input activation
                    gradient = layer_errors[next_layer_idx][j] * self.networkStructure[layer_idx][i].activation
                    self.weight_gradients[weight_key] += gradient
        
        # Increment batch counter
        self.batch_count += 1
        
        # Update weights and biases only when batch is complete
        if self.batch_count >= batch_size:
            # Apply accumulated gradients
            for layer_idx in range(0, len(self.networkStructure) - 2, 2):  # Process neuron layers
                next_layer_idx = layer_idx + 2
                connection_layer_idx = layer_idx + 1
                
                # Update weights and biases
                for j, next_neuron in enumerate(self.networkStructure[next_layer_idx]):
                    connection_batch = self.networkStructure[connection_layer_idx][j]
                    
                    # Update bias with averaged gradient
                    bias_key = f"layer_{next_layer_idx}_neuron_{j}_bias"
                    averaged_bias_gradient = self.bias_gradients[bias_key] / batch_size
                    next_neuron.setBias(next_neuron.bias + learning_rate * averaged_bias_gradient)
                    
                    # Update weights with averaged gradients
                    for i, connection in enumerate(connection_batch):
                        weight_key = f"layer_{layer_idx}_to_{next_layer_idx}_neuron_{i}_to_{j}"
                        averaged_weight_gradient = self.weight_gradients[weight_key] / batch_size
                        connection.setWeight(connection.weight + learning_rate * averaged_weight_gradient)
            
            # Reset gradient accumulators and batch counter
            self.weight_gradients.clear()
            self.bias_gradients.clear()
            self.batch_count = 0
        
        return layer_errors

    def getMSE(self, trueMnistlabel):
        self.trueMnistlabel = trueMnistlabel
        outputActivations, HighestActivationIndex, HighestActivation = findHighestActivation(self.networkStructure[-1])

        mse = calculateMSE(trueMnistlabel, outputActivations)
                    
        return mse
            
    def wasModelCorrect(self, trueMnistlabel, boolOrString = "bool"):
        outputActivations, HighestActivationIndex, HighestActivation = findHighestActivation(self.networkStructure[-1])
        
        if boolOrString == "bool":
            if trueMnistlabel == HighestActivationIndex:
                return True
            else:
                return False
            
        elif boolOrString == "string":
            if trueMnistlabel == HighestActivationIndex:
                print(f"[True]  actual:{trueMnistlabel} == predicted:{HighestActivationIndex}")
            else:
                print(f"[False] actual:{trueMnistlabel} != predicted:{HighestActivationIndex}")

    def __repr__(self):
        outputActivations, HighestActivationIndex, findHighestActivation = findHighestActivation(self.networkStructure[-1])
        outputActivations = [ '%.3f' % elem for elem in outputActivations ] # rounding all the elements in the list to 2 decimal places
        
        visualization = f"Number of rows: {len(self.rows)}\noutput Neurons: {outputActivations}\nData label: {HighestActivationIndex}\n\n"

        return visualization


# Load all data into memory (only once!)
training_data = load_all_data('MNIST dataset/mnist_train.csv')
#[784,1568,2,20,10]
#[3,6,2,6,3]
network1 = Network([784,1568,2,20,10]) #first has to be 784 and last has to be 10
network1.create()

learning_rate = 0.01
batch_size = 64

# Example: access the 300th sample
for i in range(10000):
    MNIST_label, MNIST_data = random.choice(training_data)
    network1.input(MNIST_data)
    network1.forwardPropagation()
    mse = network1.getMSE(MNIST_label)
    network1.backPropagation()
    if i % 200 == 0:
        print(mse)
        print(network1.networkStructure[-1]) 
import random
import math
from colorama import Fore, Style
import csv
import numpy as np


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def softmax(x):
    """Apply softmax normalization to a list of values"""
    exp_values = [math.exp(val - max(x)) for val in x]  # Subtract max for numerical stability
    sum_exp = sum(exp_values)
    return [val / sum_exp for val in exp_values]

def get_dynamic_margin(iteration):
    if iteration < 10000:
        return 0.5
    elif iteration < 20000:
        return 0.8
    else:
        return 1.2

def calculateCrossEntropyScore(true_label, raw_outputs):
    """Calculate cross-entropy score"""
    # Apply softmax to raw outputs
    softmax_preds = softmax(raw_outputs)
    # Calculate cross-entropy score
    return -math.log(softmax_preds[true_label] + 1e-15)  # Small epsilon to prevent log(0)

def calculateMarginScore(true_label, raw_outputs, margin=1.0):
    """Margin score - correct answer should be higher than others by margin"""
    correct_score = raw_outputs[true_label]
    score = 0
    
    for i, score_val in enumerate(raw_outputs):
        if i != true_label:
            # Score increases when wrong answer is too close to correct answer
            score += max(0, score_val - correct_score + margin)
    
    return score / (len(raw_outputs) - 1)  # Average over wrong answers

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
    
    # Get raw activations first
    for neuron in lastRowNetworkStructure:
        outputActivations.append(neuron.activation)
        
    # Apply softmax normalization
    normalized_activations = softmax(outputActivations)
    
    # Find highest after normalization
    for i, activation in enumerate(normalized_activations):
        if activation > highestActivation:
            highestActivation = activation
            highestActivationIndex = i
        
    return normalized_activations, highestActivationIndex, highestActivation

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
        activations = [f"{neuron.activation:.2f}" for neuron in self.networkStructure[-1]]
        print(" | ".join(activations))

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
            
    def backPropagation(self, learning_rate=0.01, batch_size=32, margin=1.0):
        # Initialize gradient accumulators if not already present
        if not hasattr(self, 'weight_gradients'):
            self.weight_gradients = {}
            self.bias_gradients = {}
            self.batch_count = 0
        
        # Calculate output layer error for margin score
        output_layer = self.networkStructure[-1]
        raw_outputs = [neuron.activation for neuron in output_layer]
        
        # Store errors for each layer
        layer_errors = [None] * len(self.networkStructure)
        
        # Calculate error for output layer (margin score derivative)
        output_errors = []
        correct_score = raw_outputs[self.trueMnistlabel]
        
        for i, neuron in enumerate(output_layer):
            if i == self.trueMnistlabel:
                # For correct class: derivative is -sum of violated margins
                error = 0
                for j in range(len(raw_outputs)):
                    if j != i and raw_outputs[j] - correct_score + margin > 0:
                        error -= 1  # Each violation contributes -1 to gradient
                error = error / (len(raw_outputs) - 1)  # Average
            else:
                # For wrong classes: derivative is 1 if margin is violated, 0 otherwise
                if raw_outputs[i] - correct_score + margin > 0:
                    error = 1 / (len(raw_outputs) - 1)  # Average
                else:
                    error = 0
            
            # Apply sigmoid derivative for the activation function
            error = error * neuron.activation * (1 - neuron.activation)
            output_errors.append(error)
        
        layer_errors[-1] = output_errors
        
        # Backpropagate error through the network
        for layer_idx in range(len(self.networkStructure) - 3, -1, -2):
            current_errors = []
            next_layer_idx = layer_idx + 2
            connection_layer_idx = layer_idx + 1
            
            for i, neuron in enumerate(self.networkStructure[layer_idx]):
                error_sum = 0
                
                for j, next_neuron in enumerate(self.networkStructure[next_layer_idx]):
                    connection_batch = self.networkStructure[connection_layer_idx][j]
                    error_sum += layer_errors[next_layer_idx][j] * connection_batch[i].weight
                
                error = error_sum * neuron.activation * (1 - neuron.activation)
                current_errors.append(error)
            
            layer_errors[layer_idx] = current_errors
        
        # Accumulate gradients
        for layer_idx in range(0, len(self.networkStructure) - 2, 2):
            next_layer_idx = layer_idx + 2
            connection_layer_idx = layer_idx + 1
            
            for j, next_neuron in enumerate(self.networkStructure[next_layer_idx]):
                connection_batch = self.networkStructure[connection_layer_idx][j]
                
                bias_key = f"layer_{next_layer_idx}_neuron_{j}_bias"
                if bias_key not in self.bias_gradients:
                    self.bias_gradients[bias_key] = 0
                
                self.bias_gradients[bias_key] += layer_errors[next_layer_idx][j]
                
                for i, connection in enumerate(connection_batch):
                    weight_key = f"layer_{layer_idx}_to_{next_layer_idx}_neuron_{i}_to_{j}"
                    if weight_key not in self.weight_gradients:
                        self.weight_gradients[weight_key] = 0
                    
                    gradient = layer_errors[next_layer_idx][j] * self.networkStructure[layer_idx][i].activation
                    self.weight_gradients[weight_key] += gradient
        
        self.batch_count += 1
        
        # Update weights when batch is complete
        if self.batch_count >= batch_size:
            for layer_idx in range(0, len(self.networkStructure) - 2, 2):
                next_layer_idx = layer_idx + 2
                connection_layer_idx = layer_idx + 1
                
                for j, next_neuron in enumerate(self.networkStructure[next_layer_idx]):
                    connection_batch = self.networkStructure[connection_layer_idx][j]
                    
                    bias_key = f"layer_{next_layer_idx}_neuron_{j}_bias"
                    averaged_bias_gradient = self.bias_gradients[bias_key] / batch_size
                    next_neuron.setBias(next_neuron.bias - learning_rate * averaged_bias_gradient)
                    
                    for i, connection in enumerate(connection_batch):
                        weight_key = f"layer_{layer_idx}_to_{next_layer_idx}_neuron_{i}_to_{j}"
                        averaged_weight_gradient = self.weight_gradients[weight_key] / batch_size
                        connection.setWeight(connection.weight - learning_rate * averaged_weight_gradient)
            
            self.weight_gradients.clear()
            self.bias_gradients.clear()
            self.batch_count = 0
        
        return layer_errors

    def getMSE(self, trueMnistlabel):
        self.trueMnistlabel = trueMnistlabel
        outputActivations, HighestActivationIndex, HighestActivation = findHighestActivation(self.networkStructure[-1])

        mse = calculateMSE(trueMnistlabel, outputActivations)
                    
        return mse
    
    def getCrossEntropyScore(self, trueMnistlabel):
        self.trueMnistlabel = trueMnistlabel
        raw_outputs = [neuron.activation for neuron in self.networkStructure[-1]]
        score = calculateCrossEntropyScore(trueMnistlabel, raw_outputs)
        return score

    def getMarginScore(self, trueMnistlabel, margin=1.0):
        self.trueMnistlabel = trueMnistlabel
        raw_outputs = [neuron.activation for neuron in self.networkStructure[-1]]
        score = calculateMarginScore(trueMnistlabel, raw_outputs, margin)
        return score
           
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

def train_network(training_data_path, testing_data_path, structure, learning_rate=0.15, batch_size=16, batchesPerEpoch=1000, epochs=10):
    # Load all data into memory (only once!)
    training_data = load_all_data(training_data_path)
    testing_data = load_all_data(testing_data_path)
    #[784,1568,2,20,10]
    #[3,6,2,6,3]
    network1 = Network(structure) # first has to be 784 and last has to be 10
    network1.create() # create the network

    epoch_scores = []
    
    for iteration in range(batchesPerEpoch * epochs):
        margin = get_dynamic_margin(iteration)
        MNIST_label, MNIST_data = random.choice(training_data)
        network1.input(MNIST_data)
        network1.forwardPropagation()
        score = network1.getMarginScore(MNIST_label, margin)
        network1.backPropagation(learning_rate, batch_size, margin)
        epoch_scores.append(score)

        # At the end of each epoch, calculate and print average training and validation score
        if (iteration + 1) % batchesPerEpoch == 0:
            avg_train_score = sum(epoch_scores) / len(epoch_scores)
            epoch_scores = []

            correct = 0
            incorrect = 0
            val_scores = []
            for val_sample in range(100):
                MNIST_label, MNIST_data = random.choice(testing_data)
                network1.input(MNIST_data)
                network1.forwardPropagation()
                val_score = network1.getMarginScore(MNIST_label, margin)
                val_scores.append(val_score)
                if network1.wasModelCorrect(MNIST_label, "bool"):
                    correct += 1
                else:
                    incorrect += 1
            accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
            avg_val_score = sum(val_scores) / len(val_scores) if val_scores else 0
            print(f"Epoch {iteration // batchesPerEpoch:2d} | Train Score: {avg_train_score:.4f} | Val Score: {avg_val_score:.4f} | Val Accuracy: {accuracy:.4f}")

    return network1

def test_network(network1, testing_data, margin=1.0):
    correct = 0
    incorrect = 0
    margin_score_average = 0
    numberOfTestingSamples = 1000
    for i in range(numberOfTestingSamples):
        MNIST_label, MNIST_data = random.choice(testing_data)
        network1.input(MNIST_data)
        network1.forwardPropagation()
        score = network1.getMarginScore(MNIST_label, margin)
        margin_score_average += score
        if network1.wasModelCorrect(MNIST_label, "bool"):
            correct += 1
        else:
            incorrect += 1
            
        network1.wasModelCorrect(MNIST_label, "string")
        network1.showOutputValues()

    print(f"Average Margin Score: {margin_score_average / numberOfTestingSamples}")
    print(f"Correct: {correct}, Incorrect: {incorrect}")
    print(f"Accuracy: {correct / numberOfTestingSamples * 100:.2f}%")

if __name__ == "__main__":
    network1 = train_network('MNIST dataset\mnist_train.csv', 'MNIST dataset\mnist_test.csv', [784, 12544, 16, 160, 10], learning_rate=0.15, batch_size=16, batchesPerEpoch=1000, epochs=30)
    test_network(network1, load_all_data('MNIST dataset\mnist_test.csv'))
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
        
        if answeOrData == "lable and data":
            return int(row[0]), [float(value) for value in row[1:]]
        elif answeOrData == "answer":
            return int(row[0])
        elif answeOrData == "data":
            return [float(value) for value in row[1:]]
        else:
            raise ValueError("Invalid return_type. Use 'answer' or 'data'.")
        
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
                    activation = sigmoid(sum+targetNeuron.bias)
            
                targetNeuron.setActivation(activation)
        
                weAreAtTheEndOfTheTargetLayer = len(self.networkStructure[layerIndex+2]) == (connectionGroupNumberAndEndNeuronNumber+1)
                if weAreAtTheEndOfTheTargetLayer: # checking to see if I should index again to do the same operation on the nexnt neuron in the same row or if i should move to the next row
                    layerIndex += 2 #changing to the next neuron layer
                    connectionGroupNumberAndEndNeuronNumber = 0
                    
                else:
                    connectionGroupNumberAndEndNeuronNumber += 1
            except:
                return 1
            
    def getErodeValuesForOneExample(self, MNIST_lable):
        
        outputActivations, HighestActivationIndex, HighestActivation = findHighestActivation(self.networkStructure[-1])
        
        optimalOutputNeuronValues = []
        for i in range(len(outputActivations)):
            if i != MNIST_lable:
                optimalOutputNeuronValues.append(0)
            else:
                optimalOutputNeuronValues.append(1) #creating the optimal output list EX: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] would be a 6
    
        necessaryOutputChanges = []
        for i in range(len(self.networkStructure[-1])):
            necessaryOutputChanges.append(optimalOutputNeuronValues[i] - outputActivations[i]) # figuring out the changes that need to be made to the neural network to make obtain the optimal changes
            
            
            
            
        #1 identify neccisary changes in the output neurons
        
        #2 Change biases in the output neurons respectivly to adjust for the desired output neurons
        
        #3 change connection wieghts respectivly to adjust for the desired output neurons
        
        #4 add up the requested values from the connection layer to figure out how to change the desired activation of the next layers neurons 
        
        #5 start back at step 1
        
        pass
        
        
        
        
        
    def getMSE(self, trueMnistLable):
        self.trueMnistLable = trueMnistLable
        outputActivations, HighestActivationIndex, HighestActivation = findHighestActivation(self.networkStructure[-1])

        mse = calculateMSE(trueMnistLable, outputActivations)
                    
        return mse

            
    def __repr__(self):
        outputActivations, HighestActivationIndex, HighestActivation = findHighestActivation(self.networkStructure[-1])
        outputActivations = [ '%.3f' % elem for elem in outputActivations ] # rounding all the elements in the list to 2 decimal places
        
        visualization = f"Number of rows: {len(self.rows)}\noutput Neurons: {outputActivations}\nData lable: {HighestActivationIndex}\n\n"
            
        return visualization


#[784,1568,2,20,10]
#[3,6,2,6,3]
network1 = Network([784,1568,2,20,10]) #first has to be 784 and last has to be 10
network1.create()

MNIST_lable, MNIST_data = readCSV('C:\Coding\Code\Local AI Code\AIFromScratch\MNIST dataset\mnist_train.csv', 300, 'lable and data')
network1.input(MNIST_data)
network1.forwardPropagation()
print(network1)
mse = network1.getMSE(MNIST_lable)
network1.getErodeValuesForOneExample(MNIST_lable)





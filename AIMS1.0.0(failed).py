import random
import math
from colorama import Fore, Style
import csv
import numpy as np
import copy



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
    return np.maximum(0, x)

def he_initialization(input_size):
    return np.random.randn(input_size) * np.sqrt(2.0 / input_size)

def average(lst): 
    return sum(lst) / len(lst) 

class CSVDataManager:
    def __init__(self, file_path):
        self.data = self.load_csv(file_path)
    
    def load_csv(self, file_path):
        with open(file_path, 'r') as csvfile:
            csvreader = list(csv.reader(csvfile))
            
            # Remove header if it exists
            if csvreader[0][0].isalpha():
                csvreader = csvreader[1:]
            
            return csvreader
    
    def readCSV(self, line_number, answeOrData):
        # Adjust line number if it's out of range
        adjusted_line = (line_number - 1) % len(self.data)
        
        row = self.data[adjusted_line]
        
        if answeOrData == "lable and data":
            return int(row[0]), [float(value) for value in row[1:]]
        elif answeOrData == "answer":
            return int(row[0])
        elif answeOrData == "data":
            return [float(value) for value in row[1:]]
        else:
            raise ValueError("Invalid return_type. Use 'answer' or 'data'.")
        
dataStorage = CSVDataManager('C:\Coding\Code\Local AI Code\AIFromScratch\MNIST dataset\mnist_train.csv') # stores all the data to make using it faster
        
def calculateMSE(true, pred):
    
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

def testNetwork(network):
    highestActivation = 0
    highestActivationIndex = 0
    index = 0
    outputActivations = []
    correctVsIncorrectGuesses = []
    for test in range(1000): # test the best network a certain number of times
        MNIST_lable, MNIST_data = dataStorage.readCSV(random.randint(1, 50000), 'lable and data')

        for neuron in network.networkStructure[-1]: # looking at the output neurons only
            activation = neuron.activation
            if activation > highestActivation:
                highestActivation = activation
                
                highestActivationIndex = index
            index += 1
            outputActivations.append(activation)
            
        if MNIST_lable == highestActivationIndex:
            correctVsIncorrectGuesses.append(1)
        else:
            correctVsIncorrectGuesses.append(0)
            
    correctPercentage = (sum(correctVsIncorrectGuesses)) / len(correctVsIncorrectGuesses) # finding the average correct guess percent
    
    
    return correctPercentage


def createPopulation(originalNetwork, populationSize=10):
    population = []
    for _ in range(populationSize):
        # Create a deep copy of the original network
        newNetwork = copy.deepcopy(originalNetwork)
        
        # Apply evolution to the new network
        newNetwork.evolution(originalNetwork)  # Using default values for mutationRate and mutationRange
        
        population.append(newNetwork)
    
    return population

class Neuron:
    def __init__(self, row, column):
        self.activation = 0
        self.bias = random.uniform(-1,1)
        self.row = row
        self.column = column
        
    def setActivation(self, activation: float):
        self.activation = relu(activation)
        
    def setBias(self, bias: float):
        self.bias = bias
        
    def __repr__(self):
        return f"{Fore.RED}|{Style.RESET_ALL}Neuron:({self.row},{self.column}) activation={self.activation} bias={self.bias}{Fore.RED}|{Style.RESET_ALL}"
        
class Connection:
    def __init__(self, row, number, inputSize):
        self.weight = np.random.randn() * np.sqrt(2.0 / inputSize)
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
                    activation = relu(sum+targetNeuron.bias)
            
                targetNeuron.setActivation(activation)
        
                weAreAtTheEndOfTheTargetLayer = len(self.networkStructure[layerIndex+2]) == (connectionGroupNumberAndEndNeuronNumber+1)
                if weAreAtTheEndOfTheTargetLayer: # checking to see if I should index again to do the same operation on the nexnt neuron in the same row or if i should move to the next row
                    layerIndex += 2 #changing to the next neuron layer
                    connectionGroupNumberAndEndNeuronNumber = 0
                    
                else:
                    connectionGroupNumberAndEndNeuronNumber += 1
            except:
                return 1
            
    def getMSE(self):
        listOfMses = []
        for trial in range(100):
            trueMnistLable = dataStorage.readCSV(random.randint(1, 50000), 'answer')
            
            highestActivation = 0
            highestActivationIndex = 0
            index = 0
            outputActivations = []
            
            for neuron in self.networkStructure[-1]:
                activation = neuron.activation
                if activation > highestActivation:
                    highestActivation = activation
                    
                    highestActivationIndex = index
                index += 1
                outputActivations.append(activation)
                
            mse = calculateMSE(trueMnistLable, outputActivations)
            listOfMses.append(mse)
        
        averageMse = average(listOfMses)
                    
        return averageMse
    
    def evolution(self, network, mutationRate=0.2, mutationRange=0.2):
        # Ensure mutationRate and mutationRange are not None
        mutationRate = 0.1 if mutationRate is None else mutationRate
        mutationRange = 0.1 if mutationRange is None else mutationRange

        # Type checking
        if not isinstance(mutationRate, (int, float)) or not isinstance(mutationRange, (int, float)):
            raise TypeError("mutationRate and mutationRange must be numbers")

        for layer in self.networkStructure:
            for item in layer:
                if isinstance(item, Neuron):
                    if random.random() < mutationRate:
                        item.bias += random.uniform(-mutationRange, mutationRange)
                elif isinstance(item, Connection):
                    if random.random() < mutationRate:
                        item.weight += random.uniform(-mutationRange, mutationRange)
        return network
        
        

            
    def __repr__(self):
        highestActivation = 0
        highestActivationIndex = 0
        index = 0
        outputActivations = []
        
        for neuron in self.networkStructure[-1]:
            activation = neuron.activation
            if activation > highestActivation:
                highestActivation = activation
                
                highestActivationIndex = index
            index += 1
            outputActivations.append(activation)
            
        outputActivations = [ '%.3f' % elem for elem in outputActivations ] # rounding all the elements in the list to 2 decimal places
        
        visualization = f"Number of rows: {len(self.rows)}\noutput Neurons: {outputActivations}\nData lable: {highestActivationIndex}\n\n"
            
        return visualization


#[784,1568,2,20,10]
#[3,6,2,6,3]
mainNetwork = Network([3,6,2,6,3])
mainNetwork.create()

MNIST_lable, MNIST_data = dataStorage.readCSV(300, 'lable and data')
mainNetwork.input(MNIST_data)
mainNetwork.forwardPropagation()
print(f"---------{testNetwork(mainNetwork)}")
#mainNetwork.showAllValues()
#mse = mainNetwork.getMSE(MNIST_lable)



epochNumber = 1
for i in range(1000):
    print(f"EPOCH: {i}")  # Changed epochNumber to i for clarity
    
    # Create a population of 10 modified networks
    population = createPopulation(mainNetwork, 30)
    
    correctPercentList = []
    for network in population:   
        correctPercentList.append(testNetwork(network))

    print(correctPercentList)
    indexOfBestNetwork = correctPercentList.index(max(correctPercentList))
    print(indexOfBestNetwork)
    mainNetwork = copy.deepcopy(population[indexOfBestNetwork])
    print(f"average mse: {mainNetwork.getMSE()}")
    
    print(f"new network % correct: {testNetwork(mainNetwork)}")






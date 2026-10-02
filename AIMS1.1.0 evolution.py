import random
import math
from colorama import Fore, Style
import csv
import numpy as np
import copy


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
        
    def initialize(self):
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
            
    def getMSE(self, trueMnistLable):
        self.trueMnistLable = trueMnistLable
        
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
                    
        return mse
    
    def getAccuracyPercentage(self, tests=100):
        correctVsIncorrectGuesses = []
        for test in range(tests): # test the best network a certain number of times
            highestActivation = 0
            highestActivationIndex = 0
            index = 0
            outputActivations = []
            MNIST_lable, MNIST_data = dataStorage.readCSV(random.randint(1, 50000), 'lable and data')
            
            self.input(MNIST_data)
            self.forwardPropagation()

            for neuron in self.networkStructure[-1]: # looking at the output neurons only
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
                
        accuracyPercentage = (sum(correctVsIncorrectGuesses)) / len(correctVsIncorrectGuesses) # finding the average correct guess percent
        
        return accuracyPercentage
        
            
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

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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

def duplicateNetwork(network, numDuplicates):
    duplicates = []
    for i in range(numDuplicates):
        newNetwork = copy.deepcopy(network)
        duplicates.append(newNetwork)
    return duplicates

def randomize(networkList):
    for network in networkList:
        network.initialize      
    return networkList

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

def randomCombination(parent1, parent2):
    children = []
    for _ in range(4):  # Create 4 children instead of 2
        child = copy.deepcopy(parent1)
        for layerIndex in range(len(child.networkStructure)):
            layer = child.networkStructure[layerIndex]
            if isinstance(layer[0], Neuron):
                for i in range(len(layer)):
                    if random.random() < 0.5:
                        child.networkStructure[layerIndex][i].bias = parent2.networkStructure[layerIndex][i].bias
            else:  # Connection layer
                for i in range(len(layer)):
                    for j in range(len(layer[i])):
                        if random.random() < 0.5:
                            child.networkStructure[layerIndex][i][j].weight = parent2.networkStructure[layerIndex][i][j].weight
        children.append(child)
    return children

def mutateNetwork(network, mutationRate=0.2, mutationRange=0.2):
    if not hasattr(network, 'networkStructure'):
        print(f"Error: Invalid network object. Type: {type(network)}, Value: {network}")
        return [network, network]  # Return the original network twice to maintain the count

    mutatedNetworks = []
    for _ in range(2):  # Create 2 mutated networks
        mutatedNetwork = copy.deepcopy(network)
        for layer in mutatedNetwork.networkStructure:
            if isinstance(layer[0], Neuron):
                for neuron in layer:
                    if random.random() < mutationRate:
                        neuron.bias += random.uniform(-mutationRange, mutationRange)
            else:  # Connection layer
                for connections in layer:
                    for connection in connections:
                        if random.random() < mutationRate:
                            connection.weight += random.uniform(-mutationRange, mutationRange)
        mutatedNetworks.append(mutatedNetwork)
    return mutatedNetworks

# Also, let's modify the reproduction function to print out some debug information:

def reproduction(population, crossoverPercent=60, mutationPercent=30, elitePercent=10):
    populationSize = len(population)
    crossoverSize = int(populationSize * (crossoverPercent / 100))
    mutationSize = int(populationSize * (mutationPercent / 100))
    eliteSize = populationSize - crossoverSize - mutationSize
    
    # Sort population by fitness (assuming higher is better)
    population.sort(key=lambda x: x.getAccuracyPercentage(100), reverse=True)
    
    elite = population[:eliteSize]
    crossover = population[eliteSize:eliteSize+crossoverSize]
    mutation = population[eliteSize+crossoverSize:]
    
    offspring = []
    for i in range(0, len(crossover), 2):
        if i + 1 < len(crossover):
            children = randomCombination(crossover[i], crossover[i+1])
            offspring.extend(children)
        else:
            offspring.extend([copy.deepcopy(crossover[i]), copy.deepcopy(crossover[i])])
    
    mutated = []
    for network in mutation:
        mutated.extend(mutateNetwork(network))
    
    return elite + offspring + mutated

def findBest(networks, numberOfbestNetworks):
    networkEvals = []
    indexesNeeded = []
    for network in networks:
       networkEvals.append(network.getAccuracyPercentage(100)) # get evals
       
    networkEvalsSorted = networkEvals.copy()
    networkEvalsSorted.sort()
    topHalf = networkEvalsSorted[-numberOfbestNetworks:] # get top 50% of evals
    
    for eval in topHalf:
        indexesNeeded.append(networkEvals.index(eval))
    #print(f"network evals: {networkEvals}   indexes needed: {indexesNeeded}")
    bestNetworks = []
    for index in indexesNeeded:
        bestNetworks.append(networks[index])
        
    averageEval = (sum(topHalf))/len(topHalf)
        
    return bestNetworks, averageEval

        



dataStorage = CSVDataManager('C:\Coding\Code\Local AI Code\AIFromScratch\MNIST dataset\mnist_train.csv') # stores all the data in the code instead of pinging the csv every time to make using it faster


#[784,1568,2,20,10]
#[3,6,2,6,3]
network1 = Network([784,50176,64,2048,32,320,10])
network1.initialize()

MNIST_lable, MNIST_data = dataStorage.readCSV(300, 'lable and data')
network1.input(MNIST_data)
network1.forwardPropagation()
mse = network1.getMSE(MNIST_lable)
accuracy = network1.getAccuracyPercentage()
#print(f"mse: {mse}  accuracy: {accuracy}")

initialPopulation = duplicateNetwork(network1, 100)
initialPopulation = randomize(initialPopulation)

population, averageEval = findBest(initialPopulation, 100)  # Keep all initial population

for generation in range(30):
    population = reproduction(population, 60, 30, 10)  # 60% crossover, 30% mutation, 10% elite
    _, averageEval = findBest(population, 100)  # Evaluate all networks
    print(f"Generation: {generation} Average accuracy: {averageEval}")





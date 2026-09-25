from .BaseInterface import BaseInterface, ACTIV
import numpy as np
import math

class Weight():
    def __init__(self, initial_value):

class Neuron():
    def __init_(self):
        pass


class NNInterface(BaseInterface):

    class Activation():
        def __init__(self, activation: ACTIV = ACTIV.kBinary):
            self.activation = activation

        def function(self, x) -> float:
            if self.activation == ACTIV.kBinary:
                return self.binary_sigmoid(x)
            elif self.activation == ACTIV.kBipolar:
                return self.bipolar_sigmoid(x)
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")

        def derivative(self, x) -> float:
            if self.activation == ACTIV.kBinary:
                return self.deriv_binary_sigmoid(x)
            elif self.activation == ACTIV.kBipolar:
                return self.deriv_bipolar_sigmoid(x)
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")

        def bipolar_sigmoid(self, x) -> float:
            return 2 / (1 + math.exp(-x)) - 1

        def deriv_bipolar_sigmoid(self, x) -> float:
            return 0.5 * (1 - (self.bipolar_sigmoid(x) ** 2))

        def binary_sigmoid(self, x) -> float:
            return 1 / (1 + math.exp(-x))

        def deriv_binary_sigmoid(self, x) -> float:
            return self.binary_sigmoid(x) * (1 - self.binary_sigmoid(x))


    def __init__(self):
        super().__init__()
        self._errorThresh: float = 0.05
        self.activation: NNInterface.Activation

        self.input:     np.array
        self.nodes:     np.array
        self.weights:   list[np.array]

        self.outputs: np.array

    def initWeights(self):
        pass

    def zeroWeights(self):
        pass



class NeuralNetwork(NNInterface):
    def __init__(self, inputs: int = 2, hiddenNodes: int = 4, hiddenLayers: int = 2,
                 learningRate: float = 0.2, momentum: float = 0.0,
                 mode: ACTIV = ACTIV.kBinary):
        super().__init__()

        self.input = np.zeros(inputs) # input vector

        self.nodes = np.ones((hiddenLayers, hiddenNodes)) # matrix of nodes (organized as row = layer, column = node)

        self.weights = []
        for layerIndex in range(hiddenLayers-1):
            if layerIndex == 0:
                self.weights = [np.zeros((inputs, hiddenNodes))]
            else:
                self.weights.append(np.zeros((hiddenNodes + 1, hiddenNodes)))

        self.activation = NNInterface.Activation(activation=mode)

    def fwdProp(self) -> None:
        for index, layer in enumerate(self.weights):
            if index == 0:
                self.nodes[index, :] = np.dot(np.append(self.input, 1).T, layer)
            else:
                self.nodes[index, :] = np.dot(np.append(self.nodes[index - 1, :], 1), layer)

        weightedSum = np.dot(self.nodes[-1, :], self.weights[-1])

        self.output = self.activation.function(weightedSum)
        if self.outputs is not None:
            self.outputs = np.append(self.outputs, self.output)
        else:
            self.outputs = np.array([self.output])

    def halfTotalSquareError(self):
        error = self.outputs - self.expectedOutputs # vector subtraction
        return 0.5 * np.dot(error.T, error)

    def backProp(self):
        pass

    def trainModel(self):
        self.loadTrainingSet()
        self.expectedOutputs = np.zeros(self.trainingSet["XOR"].len)
        for ip,op in self.trainingSet["XOR"].items():
            self.inputs = np.array(ip) # set inputs
            self.fwdProp()
            d = self.halfTotalSquareError()
        # loop through number of outputs
        # compute error

    def saveWeights(self):
        pass

    def loadWeights(self):
        pass



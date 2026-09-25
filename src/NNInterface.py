from BaseInterface import BaseInterface, ACTIV
import numpy as np
import math

class NNInterface(BaseInterface):
    class Activation():
        def __init__(self, activation: ACTIV = ACTIV.kBinary):
            self.activation = activation

        def function(self, x: np.array) -> float:
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

        def bipolar_sigmoid(self, x: np.ndarray | float) -> np.ndarray | float:
            return 2 / (1 + np.exp(-x)) - 1

        def deriv_bipolar_sigmoid(self, x: np.ndarray | float) -> np.ndarray | float:
            return 0.5 * (1 - (self.bipolar_sigmoid(x) ** 2))

        def binary_sigmoid(self, x: np.ndarray | float) -> np.ndarray | float:
            return 1 / (1 + np.exp(-x))

        def deriv_binary_sigmoid(self, x: np.ndarray | float) -> np.ndarray | float:
            return self.binary_sigmoid(x) * (1 - self.binary_sigmoid(x))


    def __init__(self):
        super().__init__()
        self._errorThresh: float = 0.05
        self.activation: NNInterface.Activation

        self.input:     np.array
        self.nodes:     np.array
        self.weights:   list[np.array]
        self.weightedSum: list[float]

        self.outputs: np.array

    def initWeights(self, initialWeightRange) -> None:
        pass

    def zeroWeights(self):
        pass



class NeuralNetwork(NNInterface):
    def __init__(self, inputs: int = 2, hiddenNodes: list[int] | None = [4, 4], outputs: int = 1,
                 learningRate: float = 0.2, momentum: float = 0.0,
                 mode: ACTIV = ACTIV.kBinary, initialWeightRange: tuple[float, float] = (-0.5, 0.5)):
        super().__init__()

        if inputs < 2 or outputs < 1 or hiddenNodes is None or len(hiddenNodes) < 1:
            raise ValueError(f"inputs ({inputs}), outputs ({outputs}), and hiddenNodes ({hiddenNodes}) must be positive")

        self.learningRate = learningRate
        self.momentum = momentum
        self.activation = NNInterface.Activation(activation=mode)
        self.rng = np.random.default_rng(4)

        self._layerSizes = [inputs, *hiddenNodes, outputs]
        self._outputs = np.array([])

        self.input = np.zeros(inputs)
        self.nodes = [np.zeros(n) for n in hiddenNodes]
        self.output = np.zeros(outputs)

        # Example: [2, 4, 4, 1] means input -> hidden -> hidden -> output.

        self.weightedSum: list
        self.initWeights(initialWeightRange)

    def initWeights(self, initialWeightRange: tuple[float, float]) -> None:
        low, high = initialWeightRange
        if low >= high:
            raise ValueError("initialWeightRange must have low < high")

        self.weights = [
            self.rng.uniform(low, high, size=(n_in + 1, n_out))
            for n_in, n_out in zip(self._layerSizes[:-1], self._layerSizes[1:])
        ]
        self.previousUpdates = [np.zeros_like(w) for w in self.weights]

    def zeroWeights(self) -> None:
        for w in self.weights:
            w.fill(0.0)
        for update in self.previousUpdates:
            update.fill(0.0)

    def fwdProp(self) -> float:
        output: float = 0.0
        self.weightedSums = []

        for index, layer in enumerate(self.weights):
            if index == len(self.weights) - 1:
                self.weightedSums.append(np.matmul(layer.T, np.append(self.nodes[index - 1], 1)))
                output: float = self.activation.function(self.weightedSums[-1])
                break

            if index == 0:
                self.weightedSums.append(np.matmul(layer.T, np.append(self.input, 1)))
                self.nodes[index] = self.activation.function(self.weightedSums[-1])
            else:
                self.weightedSums.append(np.matmul(layer.T, (np.append(self.nodes[index - 1], 1))))
                self.nodes[index] = self.activation.function(self.weightedSums[-1])

        return output
        # if len(self._outputs) > 1:
        #     self._outputs = np.append(self._outputs, output)
        # else:
        #     self._outputs = np.array([self.output])

    def halfTotalSquareError(self):
        error = (self._outputs - self.expectedOutputs).T # vector subtraction
        return 0.5 * np.dot(error, error)

    def backProp(self, target: float) -> None:
        # deltas[i] belongs to the neurons reached by self.weights[i].
        deltas: list[np.array] = [np.zeros(len(self.nodes[0]))] * len(self.weights)

        # Output layer: (target - output) * activation derivative
        deltas[-1] = ((target - self.output) * self.activation.derivative(self.weightedSums[-1]))

        # Hidden layers, from last to first.
        for layer in range(len(self.weights) - 2, -1, -1):
            # delete the bias row from weights, and then dot product between weights and delta
            weighted_error = np.dot(np.delete(self.weights[layer + 1], -1), deltas[layer + 1])
            # calculate the delta for the next row
            deltas[layer] = (weighted_error* self.activation.derivative(self.weightedSums[layer]))

        # Now that every delta uses the original weights, update the weights.
        for layer, weights in enumerate(self.weights):
            if layer == 0:
                previous_output = self.input
            else:
                previous_output = self.nodes[layer - 1, :]

            previous_with_bias = np.append(previous_output, 1.0)
            weights += self.learningRate * np.outer(previous_with_bias, deltas[layer])

    def trainModel(self):
        self.loadTrainingSet()
        self.expectedOutputs = np.array([])
        self._outputs = np.array([])
        for ip,op in self.trainingSet["XOR"].items():
            self.inputs = np.array(ip) # set inputs
            self.expectedOutputs = np.append(self.expectedOutputs, op)
            self._outputs = np.append(self._outputs, self.fwdProp())
        d = self.halfTotalSquareError()
        pass
        # loop through number of outputs
        # compute error

    def saveWeights(self):
        pass

    def loadWeights(self):
        pass



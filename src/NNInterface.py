from BaseInterface import BaseInterface, ACTIV, LEARN
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
                 activFunc: ACTIV = ACTIV.kBinary, learnMode: LEARN = LEARN.kOnline, initialWeightRange: tuple[float, float] = (-0.5, 0.5)):
        super().__init__()

        if inputs < 2 or outputs < 1 or hiddenNodes is None or len(hiddenNodes) < 1:
            raise ValueError(f"inputs ({inputs}), outputs ({outputs}), and hiddenNodes ({hiddenNodes}) must be positive")

        self.learningRate = learningRate
        self.momentum = momentum
        self.activation = NNInterface.Activation(activation=activFunc)
        self.learningMode = learnMode
        self.rng = np.random.default_rng(4)
        self._errorThresh = 0.05

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

    def backProp(self, target: float) -> list[np.ndarray]:
        # deltas[i] belongs to the neurons reached by self.weights[i].
        # there should be deltas for each node + the output node
        deltas = [None] * len(self.weights)

        # Output layer: (target - output) * activation derivative
        deltas[-1] = ((target - self._outputs[-1]) * self.activation.derivative(self.weightedSums[-1]))

        # Hidden layers in reverse order.
        for layer in range(len(self.weights) - 2, -1, -1):
            # get delta from layer above
            delta = deltas[layer + 1]

            # delete the bias row from weights, and then dot product between weights and delta
            w = np.delete(self.weights[layer + 1], -1, axis=0) #self.weights[layer + 1][:-1, :]
            weighted_error = w @ delta

            # calculate the delta for the next row
            deltas[layer] = weighted_error * self.activation.derivative(self.weightedSums[layer])

        # Now that every delta uses the original weights, update the weights.
        changes : list[np.ndarray] = []
        for layer, weights in (enumerate(self.weights)):
            # add bias row back into inputs
            previous_input = np.append(self.input if layer == 0 else self.nodes[layer-1], 1.0)

            # use outer product (mat mul) to get weight matrix size correct and scalar mult by learning rate
            changes.append(self.learningRate * np.outer(previous_input, deltas[layer]))

        # return changes to be applied later (depending on online or batch mode)
        return changes

    def updateWeights(self):
        pass

    def trainModel(self):
        self.loadTrainingSet()

        maxEpoch = 10000
        epoch = 0
        error: list[float] = []

        while True:
            self.expectedOutputs = np.array([])
            self._outputs = np.array([])
            avg_change = []
            for ip,op in self.trainingSet["XOR"].items():
                # set inputs and expected outputs
                self.inputs = np.array(ip) # set inputs
                self.expectedOutputs = np.append(self.expectedOutputs, op)

                # do forward propagation once for input set and compute error
                self._outputs = np.append(self._outputs, self.fwdProp())
                d = self.halfTotalSquareError()

                # do back propagation step
                avg_change.append(self.backProp(target=op))

                # apply back propagation to online learning mode
                if self.learningMode == LEARN.kOnline:
                    self.weights = [w + dw for w, dw in zip(self.weights, avg_change.pop())]

            if self.learningMode == LEARN.kBatch:
                self.weights = [w + dw for w, dw in zip(self.weights, [np.mean(change, axis=0) for change in zip(*avg_change)])]

            # do forward propagation again
            predictions = []
            for ip,op in self.trainingSet["XOR"].items():
                # set inputs and expected outputs
                self.inputs = np.array(ip)  # set inputs
                self.expectedOutputs = np.append(self.expectedOutputs, op)
                predictions.append(self.fwdProp())

            # compute error
            self._outputs = np.asarray(predictions)
            error.append(self.halfTotalSquareError())

            if error[-1] <= self._errorThresh or epoch >= maxEpoch:
                break

            epoch +=1
        pass
        # loop through number of outputs
        # compute error

    def saveWeights(self):
        pass

    def loadWeights(self):
        pass



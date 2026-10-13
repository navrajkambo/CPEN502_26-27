from BaseInterface import BaseInterface, ACTIV, LEARN
import numpy as np
import math

class NNInterface(BaseInterface):
    def __init__(self):
        super().__init__()
        self._errorThresh: float = 0.05


        self.inputs:     np.ndarray
        self.nodes:     np.ndarray
        self.weights:   list[np.ndarray]
        self.weightedSum: list[float]

        self.outputs: np.ndarray

    def initWeights(self, initialWeightRange) -> None:
        pass

    def zeroWeights(self):
        pass

class NeuralNetwork(NNInterface):
    def __init__(self, inputs: int = 2, hiddenNodes: list[int] | None = [4, 4], outputs: int = 1,
                 learningRate: float = 0.2, momentum: float = 0.0,
                 activFunc: ACTIV = ACTIV.kBinary, learnMode: LEARN = LEARN.kOnline, initialWeightRange: tuple[float, float] = (-0.5, 0.5), maxEpoch: int = 10000):
        super().__init__()

        if inputs < 2 or outputs < 1 or hiddenNodes is None or len(hiddenNodes) < 1:
            raise ValueError(f"inputs ({inputs}), outputs ({outputs}), and hiddenNodes ({hiddenNodes}) must be positive")

        self.learningRate = learningRate
        self.momentum = momentum
        self.activation = NNInterface.Activation(activation=activFunc)
        self.learningMode = learnMode
        self.rng = np.random.default_rng(4)
        self._errorThresh = 0.05
        self._maxEpoch = maxEpoch

        self._layerSizes = [inputs, *hiddenNodes, outputs]
        self._outputs = np.array([])
        self._lastWeights: list[np.ndarray]

        self.inputs = np.zeros(inputs)
        self.nodes = [np.zeros(n) for n in hiddenNodes]
        self.output = np.zeros(outputs)

        # Example: [2, 4, 4, 1] means input -> hidden -> hidden -> output.

        self.weightedSum: list
        self.initialWeightRange = initialWeightRange
        self.initWeights(initialWeightRange)

    def initWeights(self, initialWeightRange: tuple[float, float]) -> None:
        low, high = initialWeightRange
        if low >= high:
            raise ValueError("initialWeightRange must have low < high")

        self.weights = [
            self.rng.uniform(low, high, size=(n_in + 1, n_out))
            for n_in, n_out in zip(self._layerSizes[:-1], self._layerSizes[1:])
        ]
        self._lastWeights = [w.copy() for w in self.weights]

    def zeroWeights(self) -> None:
        for w in self.weights:
            w.fill(0.0)
        for update in self._lastWeights:
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
                self.weightedSums.append(np.matmul(layer.T, np.append(self.inputs, 1)))
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
        error = np.subtract(self._outputs.ravel(), self.expectedOutputs.ravel()).T # vector subtraction
        return 0.5 * np.dot(error,error)

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
            previous_input = np.append(self.inputs if layer == 0 else self.nodes[layer-1], 1.0)

            # use outer product (mat mul) to get weight matrix size correct and scalar mult by learning rate
            changes.append(self.learningRate * np.outer(previous_input, deltas[layer]))

        # return changes to be applied later (depending on online or batch mode)
        return changes

    def backPropSequential(self, target: float) -> None:
        # Reconstruct the current output from the saved forward-pass sum.
        output = self.activation.function(self.weightedSums[-1])
        target = np.asarray(target, dtype=float).reshape(-1)

        delta = (target - output)* self.activation.derivative(self.weightedSums[-1])

        # Update the output layer first, then move toward the input.
        for layer in range(len(self.weights) - 1, -1, -1):
            if layer < len(self.weights) - 1:
                # The next layer's weights have ALREADY been updated.
                next_weights = self.weights[layer + 1][:-1, :]
                delta = (next_weights @ delta) * self.activation.derivative(self.weightedSums[layer])

            previous_output = self.inputs if layer == 0 else self.nodes[layer - 1]

            previous_with_bias = np.append(previous_output, 1.0)

            # Includes the bias-weight update in the final row.
            change = self.learningRate * np.outer(previous_with_bias, delta)
            self.updateWeights(layer, change)

    def updateWeights(self, layer: int, change: np.ndarray) -> None:
        current = self.weights[layer]
        previous_update = current - self._lastWeights[layer]

        update = change + self.momentum * previous_update

        # Save the pre-update weights. A copy prevents shared-array problems.
        self._lastWeights[layer] = current.copy()
        current += update

    def trainModel(self, sequentialUpdates: bool = False, plot: bool = True, interactive: bool = False) -> tuple[bool, list[float]]:
        self.loadTrainingSet()
        error: list[float] = []
        success = False

        if sequentialUpdates and self.learningMode != LEARN.kOnline:
            raise ValueError("Sequential layer updates require online mode")

        for epoch in range(1, self._maxEpoch + 1):
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

                # apply back propagation to online learning mode
                if sequentialUpdates:
                    self.backPropSequential(target=op)
                else:
                    # do back propagation step
                    changes = self.backProp(target=op)
                    if self.learningMode == LEARN.kOnline:
                        for layer, change in enumerate(changes):
                            self.updateWeights(layer, change)
                    else:
                        avg_change.append(changes)
                    # self.weights = [w + dw for w, dw in zip(self.weights, avg_change.pop())]

            if self.learningMode == LEARN.kBatch:
                for layer, layer_changes in enumerate(zip(*avg_change)):
                    change = np.mean(layer_changes, axis=0)
                    self.updateWeights(layer, change)

            # do forward propagation again
            predictions = []
            self.expectedOutputs = np.array([])
            for ip,op in self.trainingSet["XOR"].items():
                # set inputs and expected outputs
                self.inputs = np.array(ip)  # set inputs
                self.expectedOutputs = np.append(self.expectedOutputs, op)
                predictions.append(self.fwdProp())

            # compute error
            self._outputs = np.asarray(predictions)
            error.append(self.halfTotalSquareError())

            if error[-1] <= self._errorThresh:
                success = True
                break

        if interactive:
            print(f"Reached {epoch} training iterations and achieved {error[-1]} error")
        if plot:
            self.plotGraph(error)
            self.plotNetwork()
            self.showPlots()
        if interactive:
            option = input("Save weights? [Y/n]")
            if option == "Y" or option == "y":
                filename = input("Enter file name: ")
                self.saveWeights(filename)
                print(f"{filename}.nnw saved, exiting...")
            elif option == "N" or option == "n":
                print("Training finished, exiting...")
            else:
                print("Invalid option, exiting...")

        return (success, error)

    def avgTrain(self, trials: int = 100, plot: bool = True, seed: int | None = None):
        if trials < 1:
            raise ValueError("trials must be at least 1")

        # Optional seed makes the whole experiment repeatable.
        experiment_rng = np.random.default_rng(seed)
        epochs = []
        trial_seeds = []

        for i in range(trials):
            trial_seed = int(experiment_rng.integers(0, 2 ** 32))
            trial_seeds.append(trial_seed)

            # Start a fresh model, including zeroed momentum history.
            self.rng = np.random.default_rng(trial_seed)
            self.initWeights(self.initialWeightRange)

            success, loss = self.trainModel(sequentialUpdates = True, plot = False, interactive = False)
            print(f"Trained model {i}")
            if success:
                epochs.append(len(loss))


        epochs = np.asarray(epochs, dtype=int)
        successes = len(epochs)
        failures = trials - successes

        mean = float(np.mean(epochs)) if successes else float("nan")
        sd = float(np.std(epochs, ddof=1)) if successes > 1 else float("nan")

        if plot:
            self.plotStatistics(epochs, successes, failures, epochs, mean, sd)

        return {
            "epochs": epochs,
            "mean": mean,
            "sd": sd,
            "successes": successes,
            "failures": failures,
            "seeds": trial_seeds,
        }







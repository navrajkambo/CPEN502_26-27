from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np
import yaml

class NET(Enum):
    IP = "Input Layer"
    HD = "Hidden Layer"
    OP = "Output Layer"
    WT = "Weight"

class ACTIV(Enum):
    kBinary     = "Binary"
    kBipolar    = "Bipolar"

class LEARN(Enum):
    kBatch     = "Batch"
    kOnline    = "Online"

class BaseInterface(ABC):
    class Activation():
        def __init__(self, activation: ACTIV = ACTIV.kBinary):
            self.activation = activation

        def function(self, x: np.ndarray) -> float:
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


    @abstractmethod
    def __init__(self):
        self.weights: list[np.ndarray]

        self._layerSizes: list[int]
        self._outputs: np.ndarray

        self.inputs: np.ndarray
        self.nodes: list[np.ndarray]
        self.output: np.ndarray

        self.activation: BaseInterface.Activation

    def loadTrainingSet(self) -> None:
        if self.activation.activation == ACTIV.kBinary:
            filename = "BinaryTrainingSet.yaml"
        elif self.activation.activation == ACTIV.kBipolar:
            filename = "BipolarTrainingSet.yaml"
        else:
            raise ValueError(f"Unknown activation function: {self.activation.activation}")

        with open(filename, encoding="utf-8") as file:
            data = yaml.safe_load(file)

        self.trainingSet = {}

        for gate_name, rows in data.items():
            self.trainingSet[gate_name] = {
                (row["inputs"]["x1"], row["inputs"]["x2"]): row["outputs"]["y"]
                for row in rows.values()
            }

    def saveWeights(self, filename: str = "weights.npz") -> None:
        data = {
            "layerSizes": np.asarray(self._layerSizes, dtype=int),
            **{
                f"weight_{i}": weights
                for i, weights in enumerate(self.weights)
            },
        }

        # Opening the file explicitly preserves the exact filename.
        with open("weights/"+filename+".nnw", "wb") as file:
            np.savez_compressed(file, **data)

    def loadWeights(self, filename: str = "weights.npz") -> None:
        with np.load(filename, allow_pickle=False) as data:
            saved_sizes = data["layerSizes"].tolist()

            if saved_sizes != self._layerSizes:
                raise ValueError(
                    f"Saved architecture {saved_sizes} does not match "
                    f"this network's architecture {self._layerSizes}"
                )

            loaded_weights = []

            for i, (n_in, n_out) in enumerate(
                    zip(self._layerSizes[:-1], self._layerSizes[1:])
            ):
                weights = np.array(data[f"weight_{i}"], dtype=float, copy=True)
                expected_shape = (n_in + 1, n_out)

                if weights.shape != expected_shape:
                    raise ValueError(
                        f"Weight {i} has shape {weights.shape}; "
                        f"expected {expected_shape}"
                    )

                if not np.all(np.isfinite(weights)):
                    raise ValueError(f"Weight {i} contains invalid values")

                loaded_weights.append(weights)

        # Replace existing weights only after all matrices pass validation.
        self.weights = loaded_weights
        self.previousUpdates = [np.zeros_like(w) for w in self.weights]

    def plotGraph(self, error_history: list[float]):
        # error_history: list[float], one value per completed epoch
        fig, ax = plt.subplots()
        plt.plot(range(1, len(error_history) + 1), error_history)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Training error")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.show()

    def plotNetwork( # Thank you chatGPT
            self,
            showWeights: bool = True,
            decimals: int = 2,
    ):
        sizes = self._layerSizes
        max_nodes = max(sizes)
        bias_y = -(max_nodes - 1) / 2 - 1.5

        fig, ax = plt.subplots(
            figsize=(max(8, 3 * len(sizes)), max(5, max_nodes + 2))
        )

        # Position each layer vertically around y = 0.
        positions = [
            [(layer, (count - 1) / 2 - neuron) for neuron in range(count)]
            for layer, count in enumerate(sizes)
        ]

        max_weight = max(
            float(np.max(np.abs(w))) for w in self.weights
        )
        max_weight = max(max_weight, 1e-12)

        # Connections, including the final bias row of each matrix.
        for layer, weights in enumerate(self.weights):
            sources = positions[layer] + [(layer, bias_y)]
            destinations = positions[layer + 1]

            for i, (x1, y1) in enumerate(sources):
                for j, (x2, y2) in enumerate(destinations):
                    weight = float(weights[i, j])
                    color = (
                        "tab:blue" if weight > 0
                        else "tab:red" if weight < 0
                        else "gray"
                    )

                    ax.plot(
                        [x1, x2], [y1, y2],
                        color=color,
                        linewidth=0.5 + 2 * abs(weight) / max_weight,
                        alpha=0.6,
                        zorder=1,
                    )

                    if showWeights:
                        # Stagger labels along the connections.
                        t = 0.35 + 0.35 * i / max(1, len(sources) - 1)
                        ax.text(
                            x1 + t * (x2 - x1),
                            y1 + t * (y2 - y1),
                            f"{weight:.{decimals}f}",
                            fontsize=7,
                            ha="center",
                            va="center",
                            color=color,
                            bbox=dict(
                                facecolor="white", edgecolor="none",
                                alpha=0.85, pad=0.5,
                            ),
                            zorder=2,
                        )

        # Neurons.
        for layer, layer_positions in enumerate(positions):
            for neuron, (x, y) in enumerate(layer_positions):
                label = (
                    f"x{neuron + 1}" if layer == 0
                    else f"y{neuron + 1}" if layer == len(sizes) - 1
                    else f"h{layer},{neuron + 1}"
                )
                ax.scatter(
                    x, y, s=900, facecolor="white",
                    edgecolor="black", zorder=3,
                )
                ax.text(x, y, label, ha="center", va="center", zorder=4)

            if layer < len(sizes) - 1:
                ax.scatter(
                    layer, bias_y, s=700, facecolor="lightgray",
                    edgecolor="black", zorder=3,
                )
                ax.text(
                    layer, bias_y, "1",
                    ha="center", va="center", zorder=4,
                )

            heading = (
                "Input" if layer == 0
                else "Output" if layer == len(sizes) - 1
                else f"Hidden {layer}"
            )
            ax.text(
                layer, (max_nodes - 1) / 2 + 1,
                heading, ha="center", fontweight="bold",
            )

        ax.set_xlim(-0.4, len(sizes) - 0.6)
        ax.set_ylim(bias_y - 0.7, (max_nodes - 1) / 2 + 1.5)
        ax.set_title("Neural network — blue: positive, red: negative")
        ax.axis("off")
        fig.tight_layout()
        #plt.show()

        return fig, ax

    def showPlots(self):
        plt.show()
    # @abstractmethod
    # def loadWeights(self): pass

    # @abstractmethod
    # def saveWeights(self): pass

    # @abstractmethod
    # def trainModel(self): pass

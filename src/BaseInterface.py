from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
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

class BaseInterface(ABC):
    @abstractmethod
    def __init__(self):
        self.weights: np.array

    def loadTrainingSet(self) -> None:
        with open("TrainingSet.yaml", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        self.trainingSet = {}

        for gate_name, rows in data.items():
            self.trainingSet[gate_name] = {
                (row["inputs"]["x1"], row["inputs"]["x2"]): row["outputs"]["y"]
                for row in rows.values()
            }

    # @abstractmethod
    # def loadWeights(self): pass

    # @abstractmethod
    # def saveWeights(self): pass

    # @abstractmethod
    # def trainModel(self): pass

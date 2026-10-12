from NNInterface import NeuralNetwork
from LUTInterface import LUTInterface

myNet = NeuralNetwork(hiddenNodes=[4,4], maxEpoch=50000)
myNet.trainModel()


from NNInterface import NeuralNetwork
from BaseInterface import ACTIV
from LUTInterface import LUTInterface

# Binary Activation function for 2-4-1 network
myNet = NeuralNetwork(hiddenNodes=[4], maxEpoch=50000, activFunc=ACTIV.kBinary)
myNet.trainModel()
# myNet.avgTrain(100, plot=True)

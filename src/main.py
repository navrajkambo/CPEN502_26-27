from NNInterface import NeuralNetwork
from BaseInterface import ACTIV, LEARN
from LUTInterface import LUTInterface
from robocode import RoboCode
import time
import atexit

# Binary Activation function for 2-4-1 network
# myNet = NeuralNetwork(hiddenNodes=[4], maxEpoch=50000, activFunc=ACTIV.kBipolar, learnMode=LEARN.kBatch, momentum=0.9)
# myNet.trainModel()
# myNet.avgTrain(100, plot=True)



rc = RoboCode(path="robocode")
atexit.register(rc.stop)

# Start the server and GUI
rc.start()

# Do something with RoboCode
time.sleep(30)

input("Press any key to exit...")

# Close both
rc.stop()

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from BaseInterface import Activation, ACTIV
from NNInterface import Layer, NeuralNetwork


class LayerActivationTests(unittest.TestCase):
    def test_activation_derivatives(self):
        x = np.array([-2.0, -0.3, 0.4, 2.0])
        step = 1e-6
        for activation in ACTIV:
            with self.subTest(activation=activation):
                numerical = (Activation.function(x + step, activation)
                             - Activation.function(x - step, activation)) / (2 * step)
                np.testing.assert_allclose(
                    Activation.derivative(x, activation), numerical, atol=1e-9)
        self.assertEqual(Activation.derivative(0.0, ACTIV.kReLU), 0)

    def test_mixed_layer_gradients_with_vector_output(self):
        # Rotate the activations so every type is tested in each position.
        activations = list(ACTIV)
        for offset in range(len(activations)):
            order = activations[offset:] + activations[:offset]
            net = NeuralNetwork(
                hiddenNodes=[Layer(3, a) for a in order[:-1]],
                outputs=2, activFunc=order[-1], learningRate=0.1)
            # Stay away from ReLU's non-differentiable point.
            for weights in net.weights:
                weights[:] = net.rng.uniform(0.1, 0.3, size=weights.shape)
            net.inputs = np.array([0.2, 0.6])
            target = np.array([0.7, -0.2])
            net.fwdProp()
            changes = net.backProp(target)
            for layer, weights in enumerate(net.weights):
                numerical = np.zeros_like(weights)
                for index in np.ndindex(weights.shape):
                    original = weights[index]
                    weights[index] = original + 1e-6
                    plus = 0.5 * np.sum((target - net.fwdProp()) ** 2)
                    weights[index] = original - 1e-6
                    minus = 0.5 * np.sum((target - net.fwdProp()) ** 2)
                    weights[index] = original
                    numerical[index] = (plus - minus) / 2e-6
                np.testing.assert_allclose(
                    changes[layer] / net.learningRate, -numerical, atol=1e-8)

    def test_sequential_updates_use_each_layers_derivative(self):
        net = NeuralNetwork(hiddenNodes=[Layer(3, ACTIV.kBipolar), Layer(2, ACTIV.kReLU)],
                            outputs=2, activFunc=ACTIV.kLinear, momentum=0.0)
        for w in net.weights:
            w[:] = 0.2
        net.inputs = np.array([0.3, 0.7])
        target = np.array([1.0, -0.5])
        output = net.fwdProp()
        expected = [w.copy() for w in net.weights]
        delta = target - output
        for layer in range(len(expected) - 1, -1, -1):
            if layer < len(expected) - 1:
                delta = (expected[layer + 1][:-1] @ delta) * Activation.derivative(
                    net.weightedSums[layer], net.activations[layer])
            inputs = net.inputs if layer == 0 else net.nodes[layer - 1]
            expected[layer] += net.learningRate * np.outer(np.append(inputs, 1), delta)
        net.backPropSequential(target)
        for actual, wanted in zip(net.weights, expected):
            np.testing.assert_allclose(actual, wanted)

    def test_checkpoint_activation_validation_and_momentum_reset(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.nnw"
            net = NeuralNetwork(hiddenNodes=[Layer(3, ACTIV.kBipolar)],
                                activFunc=ACTIV.kLinear, momentum=0.9)
            # Existing saveWeights API adds weights/ and .nnw to the name.
            with patch("builtins.open", return_value=path.open("wb")):
                net.saveWeights("model")
            net.updateWeights(0, np.ones_like(net.weights[0]))
            net.loadWeights(path)
            for weights, previous in zip(net.weights, net._lastWeights):
                np.testing.assert_array_equal(weights, previous)
            incompatible = NeuralNetwork(hiddenNodes=[Layer(3, ACTIV.kBinary)],
                                         activFunc=ACTIV.kLinear)
            with self.assertRaisesRegex(ValueError, "activations"):
                incompatible.loadWeights(path)


if __name__ == "__main__":
    unittest.main()

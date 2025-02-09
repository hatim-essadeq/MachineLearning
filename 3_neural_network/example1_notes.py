import numpy as np

neuron = 10


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


learning_rate = 0.1  # or another small value


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        # print('inputs \n', self.input)
        # print()
        self.weights1 = np.random.rand(self.input.shape[1], neuron)
        # print('weights1 \n', self.weights1)
        # print()
        self.weights2 = np.random.rand(neuron, 1)
        # print('weights2 \n', self.weights2)
        # print()
        self.y = y
        # print('y \n', self.y)
        # print()
        self.output = np.zeros(self.y.shape)  # y hat
        # print('output \n', self.output)
        print()

    def feedforward(self):
        self.z_layer1 = sigmoid(np.dot(self.input, self.weights1))
        print('layer 1 \n', self.z_layer1)
        print()
        self.output = sigmoid(np.dot(self.z_layer1, self.weights2))

        print('output \n', self.output)
        print()

    def backprop(self):
        # Calculate the error derivative at the output:
        delta_output = -2 * (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights2 = np.dot(self.z_layer1.T, delta_output)

        # Calculate the error derivative for the hidden layer:
        delta_hidden = np.dot(delta_output, self.weights2.T) * sigmoid_derivative(self.z_layer1)
        d_weights1 = np.dot(self.input.T, delta_hidden)

        # Update weights using a learning rate:
        self.weights1 -= learning_rate * d_weights1
        self.weights2 -= learning_rate * d_weights2

    def predict(self, xPredicted):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        self.input = xPredicted
        print("Output prediction: \n")
        self.feedforward()


X = np.array([
    [12, 15, 14],
    [8, 10, 12],
    [7, 9, 10],
    [16, 18, 15],
    [10, 10, 10],
    [4, 8, 6],
    [20, 20, 20],
    [0, 5, 10],
    [13, 14, 12],
    [6, 12, 8]
])
# X = X / np.amax(X, axis=0)  # maximum of X array

y = np.array([[1],
              [1],
              [0],
              [1],
              [1],
              [0],
              [1],
              [0],
              [1],
              [0]])

nn = NeuralNetwork(X, y)

for i in range(10000):
    nn.feedforward()
    nn.backprop()
    print('--------------------------------')
#
print(nn.output)

xPredicted = np.array(([4, 8, 9]), dtype=float)
# xPredicted = np.array(([12, 10, 15]), dtype=float)
# xPredicted = xPredicted / np.amax(xPredicted, axis=0)  # maximum of xPredicted (our input data for the prediction)

nn.predict(xPredicted)

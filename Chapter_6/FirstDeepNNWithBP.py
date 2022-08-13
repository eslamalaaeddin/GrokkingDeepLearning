# Page 125
import numpy as np

np.random.seed(1)

alpha = 0.2

inputs = np.array([[1, 0, 1],
                   [0, 1, 1],
                   [0, 0, 1],
                   [1, 1, 1],
                   [0, 1, 1],
                   [1, 0, 1]])

outputs = np.array([[0, 1, 0, 1, 1, 0]]).T

weights_input_hidden = 2 * np.random.random((3, 4)) - 1  # No Neurons in prev, No weight coming from each neuron
weights_hidden_output = 2 * np.random.random((4, 1)) - 1


def relu(x):
    return (x > 0) * x


def relu_derivative(output):
    return output > 0


prediction_hidden_output = 0

# Change each [i] -> [i:i+1] ==> Diff is [i] -> [1 0 1] ,, [i + 1] -> [[1 0 1]]
for iteration in range(60):
    err = 0
    for i in range(len(outputs)):
        prediction_input_hidden = np.dot(inputs[i: i + 1], weights_input_hidden)
        prediction_input_hidden_re = relu(prediction_input_hidden)
        prediction_hidden_output = np.dot(prediction_input_hidden_re, weights_hidden_output)
        error_output_hidden = (prediction_hidden_output - outputs[i: i + 1]) ** 2
        err += np.sum(error_output_hidden)

        delta_output_hidden = prediction_hidden_output - outputs[i: i + 1]
        delta_hidden_input = np.dot(delta_output_hidden, weights_hidden_output.T) * relu_derivative(
            prediction_input_hidden_re)

        # I take prediction_input_hidden_re as it is considered the input.
        weights_hidden_output -= np.dot(prediction_input_hidden_re.T, delta_output_hidden) * alpha
        weights_input_hidden -= np.dot(inputs[i: i + 1].T, delta_hidden_input) * alpha

        # outputs_hidden = prediction_input_hidden_re
        # if iteration % 10 == 9:
        # print("Error: " + str(err))
        print("Error: " + str(round(err, 4)) + "   Pred: " + str(round(prediction_hidden_output.sum(), 4)))
    print('----------------------------------------------------')


# print(weights_hidden_output.T)
# print(weights_input_hidden)
# sum = weights_hidden_output.T + weights_input_hidden
# print(np.sum(sum, axis=1))
import numpy as np


def weighted_sum(inputs, weights):
    prediction = 0
    for i in range(len(inputs)):
        prediction += inputs[i] * weights[i]
    return prediction


def neural_network(inputs, weights):
    prediction = weighted_sum(inputs, weights)
    return prediction


# Inputs
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

inputs = [toes[0], wlrec[0], nfans[0]]

# Weights
weights = [.1, .2, 0]

prediction = neural_network(inputs, weights)
print(prediction)

#################### numpy version ####################
toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

inputs = np.array([toes[0], wlrec[0], nfans[0]])

weights = np.array([.1, .2, 0])


def weighted_sum_np(inputs, weights):
    prediction = np.dot(inputs, weights)
    return prediction


def neural_network_np(inputs, weights):
    prediction = weighted_sum_np(inputs, weights)
    return prediction


prediction = neural_network_np(inputs, weights)
print(prediction)

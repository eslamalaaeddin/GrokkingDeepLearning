import numpy as np


def neural_network(input, weights):
    predictions = []
    for weight_row in weights:
        pred_per_input = []
        for input, weight in zip(inputs, weight_row):
            pred_per_input.append(input * weight)
        predictions.append(sum(pred_per_input))
    return predictions


# Inputs
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

inputs = [toes[0], wlrec[0], nfans[0]]

# Weights
#          toes  win  fans
weights = [[0.1, 0.1, -0.3],  # hurt?
           [0.1, 0.2, 0.0],  # win?
           [0.0, 1.3, 0.1]]  # sad?
predictions = neural_network(inputs, weights)
print(predictions)


################ numpy version ###############
#
# def neural_network_numpy(inputs, weights):
#     predictions = np.multiply(inputs, weights)
#     return np.sum(predictions, axis=1)
#
#
# # Inputs
# toes = [8.5, 9.5, 9.9, 9.0]
# wlrec = [0.65, 0.8, 0.8, 0.9]
# nfans = [1.2, 1.3, 0.5, 1.0]
# inputs = np.array([[toes[0], wlrec[0], nfans[0]]])
#
# # Weights
# #          toes  win  fans
# weights = np.array([[0.1, 0.1, -0.3],  # hurt?
#                     [0.1, 0.2, 0.0],  # win?
#                     [0.0, 1.3, 0.1]])  # sad?
# predictions = neural_network_numpy(inputs, weights)
# print(predictions)

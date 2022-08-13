import numpy as np

# Page 108
weights = np.array([0.5, 0.48, -0.7])
alpha = 0.01
# streetlights = np.array([[1, 0, 1],
#                          [0, 1, 1],
#                          [0, 0, 1],
#                          [1, 1, 1],
#                          [0, 1, 1],
#                          [1, 0, 1]])
# walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])
# Below inputs and outputs are problematic for this neural network
# as pressure on weights is neutral, every weight has an equal number of positive and negative signs
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])
walk_vs_stop = np.array([1, 1, 0, 0])
for iteration in range(5000):
    cumulative_error = 0
    for input, target in zip(streetlights, walk_vs_stop):
        prediction = input.dot(weights)
        error = (target - prediction) ** 2
        cumulative_error += error
        delta = prediction - target
        weights = weights - (alpha * (input * delta))
        # print("Error:" + str(error) + " Prediction:" + str(prediction))
    # If you print the weights update in each iteration, the NN will come to a point where it will stuck (No Weight Updating)
    print(np.round(weights, 4))
    print('------- ' + str(round(cumulative_error, 4)) + ' -------')

# print(weights)

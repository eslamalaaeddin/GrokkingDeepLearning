import numpy as np

weights = np.array([0.5, 0.48, -0.7])
alpha = 0.01
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])
walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])
input = streetlights[1]
goal_prediction = walk_vs_stop[1]
for iteration in range(500):
    prediction = input.dot(weights)
    error = (goal_prediction - prediction) ** 2
    delta = prediction - goal_prediction
    weights = weights - (alpha * (input * delta))
    print("Error:" + str(error) + " Prediction:" + str(prediction))

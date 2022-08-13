weight = 0.5
goal_prediction = 0.6
input = 2
alpha = 0.1
for iteration in range(20):
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2
    # From a mathematical POV, I don't know why the derivative is like so.
    derivative = input * (prediction - goal_prediction)
    weight = weight - (alpha * derivative)
    print("Error:" + str(round(error, 3)) + " -- Pred:" + str(round(prediction, 3)) + " -- weight:" + str(round(weight, 3)))

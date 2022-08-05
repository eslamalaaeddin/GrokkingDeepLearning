toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

inputs = [toes[0], wlrec[0], nfans[0]]

weights = [0.1, 0.2, -.1]

win_or_lose_binary = [1, 1, 0, 1]
true_output = win_or_lose_binary[0]

alpha = .01


def neural_network(inputs, weights):
    prediction = 0
    for input, weight in zip(inputs, weights):
        prediction += input * weight
    return prediction


for _ in range(30):
    prediction = neural_network(inputs, weights)
    error = (prediction - true_output) ** 2
    delta = prediction - true_output

    # Update weights
    for i in range(len(weights)):
        weights[i] = weights[i] - alpha * inputs[i] * delta

    print("Error: " + str(error) + " -- Prediction: " + str(prediction))

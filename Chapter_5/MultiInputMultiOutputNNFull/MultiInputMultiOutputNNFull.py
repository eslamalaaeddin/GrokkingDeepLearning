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
hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]
alpha = .01
true = [hurt[0], win[0], sad[0]] # .1  1 .1
for _ in range(150):
    errors = []
    delta = []
    predictions = neural_network(inputs, weights)
    for i, prediction in enumerate(predictions):
        errors.append((prediction - true[i]) ** 2)
        delta.append(prediction - true[i])
        # weights[i] = weights[i] - alpha * inputs[i] * delta[i]
        weights[i] = [w - alpha * inputs[i] * delta[i] for w in weights[i]]
    print("Error: " + str(sum(errors) / len(errors)) + " -- Prediction: " + str(predictions))


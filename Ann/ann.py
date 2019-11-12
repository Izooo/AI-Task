import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def input_value(nums,inputs):
    vals_list=[]
    numT = inputs
    for k in range(0, len(inputs)):
        vals = float((numT[k]-min(nums))/(max(nums)-min(nums)))
        vals_list.append(vals)
    array1 = np.array(vals_list)
    input_array = array1.reshape(6,3)	
    return input_array

def expected_output_value(nums,outputs):
    output_vals_list=[]
    numT = outputs
    for k in range(len(outputs)):
        output_vals = (numT[k]-min(nums))/(max(nums)-min(nums))
        output_vals_list.append(output_vals)
        
    return output_vals_list

numbers=[30,40,50,20,40,50,20,15,50,20,15,60,20,15,60,70,15,60,70,50,60,70,50,40]
inputs=[30,40,50,40,50,20,50,20,15,20,15,60,15,60,70,60,70,50]
expected_outputs=[20,15,60,70,50,40]

learning_rate = 0.5


expect_output = expected_output_value(numbers,expected_outputs)

training_input = input_value(numbers,inputs)

print("Training Input", training_input)

training_outputs = np.array([expect_output]).T

print("Training Output", training_outputs)


class NeuralNetworkExample:
    def __init__(self, x, y):
        self.input = training_input
        self.weights1 = [[0.2, 0.3, 0.2], [0.1, 0.1, 0.1]]
        self.weights2 = [[0.5, 0.1]]
        self.target = expect_output
        self.learning_rate = 0.5


    def fowardPropagation(self):
        new_inputs = []
        for epoch in self.input:
            epoch = np.array(epoch, ndmin=2).T
            layer1 = sigmoid(np.dot(self.weights1, epoch))
            layer2 = sigmoid(np.dot(self.weights2, layer1))
            new_inputs.append(layer2)

        return new_inputs

    def back_propagation(self, input, target):
        target_vector = np.array(target, ndmin=2).T
        print(target_vector)
        input = np.array(input, ndmin=2).T
        

        output_vector1 = np.dot(self.weights1, input)
        output_vector_hidden = sigmoid(output_vector1)

        output_vector2 = np.dot(
            self.weights2, output_vector_hidden)
        output_vector_network = sigmoid(output_vector2)

        output_errors = target_vector - output_vector_network

        print("Input: \n", input)
        print("Expected: ", target)
        print("Actual: ", output_vector_network)
        print("Error :", output_errors, "\n")

        # update the weights:
        tmp = output_errors * output_vector_network * \
            (1.0 - output_vector_network)

        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)

        self.weights2 += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights2.T, output_errors)

        # update the weights:
        tmp = hidden_errors * output_vector_hidden * \
            (1.0 - output_vector_hidden)
        self.weights1 += self.learning_rate * \
            np.dot(tmp, input.T)

    def train(self):
        for i in range(len(self.input)):
            self.back_propagation(self.input[i], self.target[i])

    def print_matrices(self):
        print("Layer 1: ", self.weights1)
        print("Layer 2: ", self.weights2)

def main():
    size_of_learn_sample = int(len(training_input)*0.9)
    print(size_of_learn_sample)

    NN = NeuralNetworkExample(training_input, expect_output)

    for iteration in range(1000):
        NN.train()
        NN.print_matrices()

    # NN.print_matrices()
   

if __name__ == "__main__":
    main()
    
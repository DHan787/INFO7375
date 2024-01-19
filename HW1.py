class McCullochPittsNeuron:
    def __init__(self, num_inputs, weights, threshold):
        self.num_inputs = num_inputs
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError("Number of inputs must match the neuron's expected number of inputs.")

        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))

        # Apply step function
        output = 1 if weighted_sum >= self.threshold else 0
        return output

if __name__ == "__main__":
    # Define the artificial neuron parameters
    num_inputs = 3
    weights = [0.5, -0.8, 0.2]
    threshold = 0.2

    # Create an instance of McCullochPittsNeuron
    neuron = McCullochPittsNeuron(num_inputs, weights, threshold)

    # Input values
    input_values = [1, 0, 1]

    # Activate the neuron
    output = neuron.activate(input_values)

    # Display the result
    print(f"Input Values: {input_values}")
    print(f"Neuron Output: {output}")

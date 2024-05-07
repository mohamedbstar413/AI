import numpy as np
import matplotlib.pyplot as plt
#=======================================================================================
#                           Utility code
#=======================================================================================
def sigmoid(inp):
    return 1 / (1 + np.exp(-1 * inp))

def quad(inp):
    return inp**2

class NeuralNetwork:
    def __init__(self, hid_layr_wets, out_lyr_wets, act_type='quad'):
        self.l_r = 0.5
        self.hidden_layer = NeuralLayer(hid_layr_wets, act_type)
        self.output_layer = NeuralLayer(out_lyr_wets, act_type)
    def feed_forward(self, inputs):
        #every row has the weights incoming to the neuron represented by this row
        hid_output = self.hidden_layer.feed_forward(inputs)
        network_output = self.output_layer.feed_forward(hid_output)
        return network_output
    def compute_delta(self, target):
        self.dE_dO_nets = np.zeros(len(self.output_layer.neurons))

        for o, o_node in enumerate(self.output_layer.neurons):
            dE_dO_out = target[o] - o_node.output
            dO_dnet = o_node.calc_activ_deriv(o_node.output) * dE_dO_out
            self.dE_dO_nets[o] = dO_dnet
            print('delta of node ', o, ' is ', dO_dnet)

        self.dE_dH_nets = np.zeros(len(self.hidden_layer.neurons))
        for i, h_node in enumerate(self.hidden_layer.neurons):
            tmp = 0
            for o, o_node in enumerate(self.output_layer.neurons):
                tmp += (self.dE_dO_nets[o] * o_node.weights[i])
            h_node_activ_deriv = h_node.calc_activ_deriv(h_node.inp)
            self.dE_dH_nets[i] = int(tmp) * h_node_activ_deriv

    def update_weights(self):
        #update weights at output layer
        for o, o_node in enumerate(self.output_layer.neurons):
            for i, weight in enumerate(o_node.weights):
                dE_dW = self.dE_dO_nets[o] * o_node.inp[i]
                new_weight = weight - dE_dW *self.l_r
                o_node.weights[i] = new_weight
        #update weights at hidden layer
        for h, h_node in enumerate(self.hidden_layer.neurons):
            for i, weight in enumerate(h_node.weights):
                dE_dW = self.dE_dH_nets[h] * h_node.inp[i]
                new_weight = weight - self.l_r * dE_dW
                h_node.weights[i] = new_weight
    def train_step(self, inputs, target):
        output = self.feed_forward(inputs)
        print('outputs are ', output)
        self.compute_delta(target)
        self.update_weights()
class NeuralLayer:
    def __init__(self, weights, activation_type):
        self.weights = weights
        self.activation_type = activation_type
        self.neurons = []
        for i in range(len(weights)):
            self.neurons.append(Neuron(weights[i], self.activation_type))
    def feed_forward(self, inputs):
        outputs = []
        for i, node in enumerate(self.neurons):
            res = node.calc_output(inputs)
            outputs.append(res)
        return outputs

class Neuron:
    def __init__(self, weights, activation_type):
        self.weights = weights
        self.activation_type = activation_type
    def calc_output(self, h_input):
        self.inp = h_input
        self.output = np.dot(self.inp, self.weights)
        self.output = self.calc_activation(self.output)
        return self.output
    def calc_activation(self, h_input):
        if(self.activation_type == 'sigmoid'):
            return sigmoid(np.dot(h_input, self.weights))
        elif(self.activation_type == 'quad'):
            return quad(np.dot(h_input, self.weights))
        else:
            return h_input
    def calc_activ_deriv(self, h_input):
        if(self.activation_type == 'sigmoid'):
            return sigmoid(np.dot(h_input, self.weights))*(1-sigmoid(np.dot(h_input, self.weights)))
        elif(self.activation_type == 'quad'):
            return 2*np.dot(h_input, self.weights)
        return 1
def poly():     # 2 x 2 x 2
    hidden_layer_weights = np.array([[1, 1],
                                     [2, 1]])
    output_layer_weights = np.array([[2, 1],
                                     [1, 0]])

    nn = NeuralNetwork(hidden_layer_weights, output_layer_weights, 'quad')

    nn.train_step([1, 1], [290, 14])
if __name__ == '__main__':
    poly()
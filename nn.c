#include "nn.h"

void get_activations(nn_layer layer, const double *inputs, double *activations_out)
{
	for (unsigned neuron_index = 0; neuron_index < layer.neuron_count; ++neuron_index) {
		activations_out[neuron_index] = 0;
		for (unsigned input_index = 0; input_index < layer.input_size; ++input_index) {
			unsigned weight_index = layer.neuron_count * neuron_index + input_index;
			activations_out[neuron_index] += layer.weights[weight_index] * inputs[input_index];
		}
		activations_out[neuron_index] += layer.biases[neuron_index];
	}
}

double relu(double x)
{
	if (x < 0)
		return 0;
	else
		return x;
}

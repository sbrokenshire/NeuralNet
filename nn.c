#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void init_weights_and_biases(nn_layer *layer)
{
	time_t t;	
	srand((unsigned) time(&t));

	for (unsigned neuron_index = 0; neuron_index < layer->neuron_count; neuron_index++)
	{
		const double non_zero_offset = 0.01;

		double random_n = (rand() % 99);
		layer->biases[neuron_index] = random_n/100 + non_zero_offset;

		for (unsigned input_index = 0; input_index < layer->input_size; input_index++)
		{
			double random_n = (rand() % 99);
			unsigned weight_array_index = neuron_index * layer->input_size + input_index;
			layer->weights[weight_array_index] = random_n/100 + non_zero_offset;
		}
	}
}

void nn_net_init(nn_net *neural_net)
{
	neural_net->layer_count = 0;
	neural_net->batch_size = 0;
}

void nn_add_layer(nn_net *neural_net, unsigned neuron_count, activation_fn activation_fn)
{
	if (neural_net->layer_count == 0)
		neural_net->layers[neural_net->layer_count].input_size = 1;
	else
		neural_net->layers[neural_net->layer_count].input_size = neural_net->layers[neural_net->layer_count - 1].neuron_count;

	neural_net->layers[neural_net->layer_count].neuron_count = neuron_count;
	neural_net->layers[neural_net->layer_count].activation_fn = activation_fn;

	init_weights_and_biases(&neural_net->layers[neural_net->layer_count]);

	neural_net->layer_count++;
}

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

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


static layer mLayers[20];

void nn_add_layer(unsigned neuron_count, activation_fn activation_fn)
{

}

void nn_add_layer(nn_net *neural_net, unsigned neuron_count, activation_fn activation_fn)
{
	if (neural_net.layer_count = 0)
	{
		neural_net->layers[neural_net->layer_count].neuron_count = neuron_count;
		neural_net->layers[neural_net->layer_count].input_size = 1;
		neural_net->layers[neural_net->layer_count].weights;
		neural_net->layers[neural_net->layer_count].biases;
		neural_net->layers[neural_net->layer_count].activation_fn = activation_fn;
	}


	time_t t;	
	srand((unsigned) time(&t));

	for (unsigned layer_index = 0; layer_index < layer_count; layer_index++)
	{
		for (unsigned neuron_index = 0; neuron_index < layers->neuron_count; neuron_index++)
		{
			double random_n = (rand() % 100);

			layers[layer_index].biases[neuron_index] = random_n/100;

			for (unsigned input_index = 0; input_index < layers[layer_index].input_size; input_index++)
			{
				double random_n = (rand() % 100);

				unsigned weight_array_index = neuron_index * layers[layer_index].input_size + input_index;
				layers[layer_index].weights[weight_array_index] = random_n/100;
			}
		}
	}
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

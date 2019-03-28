#include "nn.h"
#include "backprop.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void init_layer_values(nn_layer *layer)
{
	time_t t;	
	srand((unsigned) time(&t));

	for (unsigned neuron_index = 0; neuron_index < layer->neuron_count; neuron_index++)
	{
		const double non_zero_offset = 0.01;

		double random_n = (rand() % 99);
		layer->biases[neuron_index] = random_n/100 + non_zero_offset;
		//layer->bias_deltas[neuron_index] = 0;

		for (unsigned input_index = 0; input_index < layer->input_size; input_index++)
		{
			double random_n = (rand() % 99);
			unsigned weight_array_index = neuron_index * layer->input_size + input_index;
			layer->weights[weight_array_index] = random_n/100 + non_zero_offset;
			//layer->weight_deltas[weight_array_index] = 0;
		}
	}
}

static void calculate_backprop(nn_net *neural_net)
{
	/*double bias_deltas[MAX_LAYER_SIZE];
	double input_deltas[MAX_LAYER_SIZE];
	double weight_deltas[MAX_LAYER_WEIGHTS];

	for (unsigned layer_index = neural_net->layer_count; layer_index > 0; layer_index--)
	{
		double summed_cost = 0.0;
		if (layer_index == neural_net->layer_count)
			summed_cost = get_summed_costs(layer_activations_out, training_data->expected_activations, neural_net->layers[layer_index].neuron_count);
		else
		{
			for (unsigned neuron_index = 0; neuron_index < neural_net->layers[layer_index].neuron_count; ++neuron_index)
			{
			}
		}

		get_bias_deltas(neural_net->layers[layer_index], layer_inputs, layer_activations_out, &relu_deriv, &bias_deltas, summed_cost);
		get_input_deltas(neural_net->layers[layer_index], layer_inputs, layer_activations_out, &relu_deriv, &input_deltas, summed_cost);
		get_Weights_deltas(neural_net->layers[layer_index], layer_inputs, layer_activations_out, &relu_deriv, &weight_deltas, summed_cost);

		for (unsigned neuron_index = 0; neuron_index < neural_net->layers[layer_index].neuron_count; ++neuron_index)
		{
			neural_net->layers[layer_index].bias_deltas[neuron_index] += bias_deltas[neuron_index];
			for (unsigned input_index 0; input_index < neural_net->layers[layer_index].input_size; input_index++)
			{
				unsigned weight_index = layer.neuron_count * neuron_index + input_index;
				neural_net->layers[layer_index].weight_deltas[weight_index] += weight_deltas[weight_index];
			}
		}
	}*/
}

static bool apply_backprop(nn_net *neural_net, double *batch_activations)
{
	for (unsigned layer_index = 0; layer_index < neural_net->layer_count; layer_index++)
	{
		for (unsigned neuron_index = 0; neuron_index < neural_net->layers[layer_index].neuron_count; neuron_index++)
		{
			neural_net->layers[layer_index].biases[neuron_index] *= neural_net->layers[layer_index].bias_deltas[neuron_index] / neural_net->batch_size;
			neural_net->layers[layer_index].bias_deltas[neuron_index] = 0;

			for (unsigned input_index = 0; input_index < neural_net->layers[layer_index].input_size; input_index++)
			{
				unsigned weight_array_index = neuron_index * neural_net->layers[layer_index].input_size + input_index;

				neural_net->layers[layer_index].weights[weight_array_index] *= neural_net->layers[layer_index].weight_deltas[weight_array_index] / neural_net->batch_size;
				neural_net->layers[layer_index].weight_deltas[weight_array_index] = 0;
			}
		}
	}
}

static double relu_deriv(double x)
{
	if (x < 0)
		return 0;
	else
		return 1;
}

static double get_summed_costs(double *activations, double *desired_activations, unsigned neuron_count)
{
	double summed_costs = 0.0;
	for (unsigned activation_index = 0; activation_index < neuron_count; activation_index++)
	{
		summed_costs += activations[activation_index] - desired_activations[activation_index];
	}

	return summed_costs;
}

void nn_net_init(nn_net *neural_net)
{
	neural_net->layer_count = 0;
	neural_net->batch_size = 0;
}

void nn_train(nn_net *neural_net, nn_training_data *training_data, unsigned training_data_size)
{
    unsigned output_layer_index = neural_net->layer_count - 1;
	unsigned output_count = neural_net->layers[output_layer_index].neuron_count;
	unsigned batch_size = neural_net->batch_size;

	double *batch_input;
	batch_input = (double*) malloc(sizeof(double) * batch_size * output_count);
	double *activations_out;
	activations_out = (double*) malloc(sizeof(double) * MAX_LAYER_SIZE);
    
	for (unsigned training_data_index = 0; training_data_index < training_data_size; training_data_index++)
	{
		/*if(training_data_index % batch_size == 0)
			apply_backprop(neural_net, batch_input);*/

		nn_process_inputs(neural_net, &training_data[training_data_index], activations_out);
		memcpy(&batch_input[output_count * (training_data_index % batch_size)], activations_out, sizeof(double) * output_count);
	}

	free(batch_input);
	free(activations_out);
}

void nn_add_layer(nn_net *neural_net, unsigned neuron_count, activation_fn activation_fn)
{
	if (neural_net->layer_count == 0)
		neural_net->layers[neural_net->layer_count].input_size = 1;
	else
		neural_net->layers[neural_net->layer_count].input_size = neural_net->layers[neural_net->layer_count - 1].neuron_count;

	neural_net->layers[neural_net->layer_count].neuron_count = neuron_count;
	neural_net->layers[neural_net->layer_count].activation_fn = activation_fn;

	init_layer_values(&neural_net->layers[neural_net->layer_count]);

	neural_net->layer_count++;
}

void nn_process_inputs(nn_net *neural_net, nn_training_data *training_data, double *activations_out)
{
	double layer_activations_out[MAX_LAYER_SIZE];
	double layer_inputs[MAX_LAYER_SIZE];

	memcpy(layer_inputs, training_data->inputs, sizeof(double) * MAX_LAYER_SIZE);

	for (unsigned layer_index = 0; layer_index < neural_net->layer_count; layer_index++)
	{
		get_activations(neural_net->layers[layer_index], layer_inputs, layer_activations_out);
		memcpy(layer_inputs, layer_activations_out, sizeof(double) * MAX_LAYER_SIZE);
	}

	memcpy(activations_out, layer_activations_out, sizeof(double) * MAX_LAYER_SIZE);
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

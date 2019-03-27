#include "nn.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

static void a_single_neurons_activation_is_the_weighted_sum_of_its_inputs_plus_its_bias_fed_through_its_activation_function(void)
{
	/* given */
	double weights[] = { 0.1, 0.2, 0.3 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 3,
		.activation_fn = &relu
	};
	memcpy(layer.weights, weights, sizeof(double) * layer.input_size * layer.neuron_count);
	memcpy(layer.biases, biases, sizeof(double) * layer.neuron_count);

	double inputs[] = { 10, 20, 30 };

	/* when */
	const double expected_activation = 14.5; /* relu(0.1*10 + 0.2*20 + 0.3*30 + 0.5) */
	double activation = 0;
	get_activations(layer, inputs, &activation);

	/* then */
	assert(activation == expected_activation);
	printf("a_single_neurons_activation_is_the_weighted_sum_of_its_inputs_plus_its_bias_fed_through_its_activation_function: passed\n");
}

static void a_layers_activations_are_an_array_of_individual_neuron_activations(void)
{
	/* given */
	double weights[] = {
		 0.1, 0.2, 0.3,
		 0.4, 0.5, 0.6,
		 0.7, 0.8, 0.9
	};
	double biases[] = { 0.5, 0.6, 0.7 };
	nn_layer layer = {
		.neuron_count = 3,
		.input_size = 3,
		.activation_fn = &relu
	};
	memcpy(layer.weights, weights, sizeof(double) * layer.input_size * layer.neuron_count);
	memcpy(layer.biases, biases, sizeof(double) * layer.neuron_count);

	double inputs[] = { 10, 20, 30 };

	/* when */
	const double expected_activations[] = { 14.5, 32.6, 50.7 };
	double activations[] = { 0, 0, 0 };
	get_activations(layer, inputs, activations);

	/* then */
	for (unsigned i = 0; i < layer.neuron_count; ++i)
		assert(activations[i] == expected_activations[i]);


	printf("a_layers_activations_are_an_array_of_individual_neuron_activations: passed\n");
}

static void relu_of_a_positive_number_is_that_number(void)
{
	double inputs[] = { 4, 5.8, 358.7, 0 };
	for (unsigned i = 0; i < 4; ++i)
		assert(relu(inputs[i]) == inputs[i]);

	printf("relu_of_a_positive_number_is_that_number: passed\n");
}

static void relu_of_a_negative_number_is_zero(void)
{
	double inputs[] = { -4, -5.8, -358.7, -0 };
	for (unsigned i = 0; i < 4; ++i)
		assert(relu(inputs[i]) == 0);
	
	printf("relu_of_a_negative_number_is_zero: passed\n");
}

static void multi_layer_neural_net_has_weights_and_biases_set_on_initialisation(void)
{
	nn_net neural_net;
	nn_net_init(&neural_net);

	nn_add_layer(&neural_net, 3, &relu);
	nn_add_layer(&neural_net, 2, &relu);
	nn_add_layer(&neural_net, 3, &relu);

	bool is_set = false;

	for (unsigned neuron_index = 0; neuron_index < neural_net.layers[0].neuron_count; ++neuron_index)
	{
		is_set = (neural_net.layers[0].weights[neuron_index] != 0);
		assert(is_set);

		is_set = (neural_net.layers[0].biases[neuron_index] != 0);
		assert(is_set);
	}

	for (unsigned neuron_index = 0; neuron_index < neural_net.layers[1].neuron_count; ++neuron_index)
	{
		for (unsigned input_index = 0; input_index < neural_net.layers[1].input_size; input_index++)
		{
			is_set = (neural_net.layers[1].weights[neuron_index * neural_net.layers[1].input_size + input_index] != 0);
			assert(is_set);
		}

		is_set = (neural_net.layers[1].biases[neuron_index] != 0);
		assert(is_set);
	}

	for (unsigned neuron_index = 0; neuron_index < neural_net.layers[2].neuron_count; ++neuron_index)
	{
		for (unsigned input_index = 0; input_index < neural_net.layers[2].input_size; input_index++)
		{
			is_set = (neural_net.layers[2].weights[input_index * neural_net.layers[2].input_size + input_index] != 0);
			assert(is_set);
		}

		is_set = (neural_net.layers[2].biases[neuron_index] != 0);
		assert(is_set);
	}
	
	printf("multi_layer_neural_net_has_weights_and_biases_set_on_initialisation: passed\n");
}

int main(void)
{
	a_single_neurons_activation_is_the_weighted_sum_of_its_inputs_plus_its_bias_fed_through_its_activation_function();
	a_layers_activations_are_an_array_of_individual_neuron_activations();
	relu_of_a_positive_number_is_that_number();
	relu_of_a_negative_number_is_zero();
	multi_layer_neural_net_has_weights_and_biases_set_on_initialisation();

	printf("all tests passed\n");
	return 0;
}

#include "nn.h"

#include <assert.h>
#include <stdio.h>

static void a_single_neurons_activation_is_the_weighted_sum_of_its_inputs_plus_its_bias_fed_through_its_activation_function(void)
{
	/* given */
	double weights[] = { 0.1, 0.2, 0.3 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 3,
		.weights = weights,
		.biases = biases,
		.activation_fn = &relu
	};
	double inputs[] = { 10, 20, 30 };

	/* when */
	const double expected_activation = 14.5; /* relu(0.1*10 + 0.2*20 + 0.3*30 + 0.5) */
	double activation = 0;
	get_activations(layer, inputs, &activation);

	/* then */
	assert(activation == expected_activation);
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
		.weights = weights,
		.biases = biases,
		.activation_fn = &relu
	};
	double inputs[] = { 10, 20, 30 };

	/* when */
	const double expected_activations[] = { 14.5, 32.6, 50.7 };
	double activations[] = { 0, 0, 0 };
	get_activations(layer, inputs, activations);

	/* then */
	for (unsigned i = 0; i < layer.neuron_count; ++i)
		assert(activations[i] == expected_activations[i]);
}

static void relu_of_a_positive_number_is_that_number(void)
{
	double inputs[] = { 4, 5.8, 358.7, 0 };
	for (unsigned i = 0; i < 4; ++i)
		assert(relu(inputs[i]) == inputs[i]);
}

static void relu_of_a_negative_number_is_zero(void)
{
	double inputs[] = { -4, -5.8, -358.7, -0 };
	for (unsigned i = 0; i < 4; ++i)
		assert(relu(inputs[i]) == 0);
}

int main(void)
{
	a_single_neurons_activation_is_the_weighted_sum_of_its_inputs_plus_its_bias_fed_through_its_activation_function();
	a_layers_activations_are_an_array_of_individual_neuron_activations();
	relu_of_a_positive_number_is_that_number();
	relu_of_a_negative_number_is_zero();

	printf("all tests passed\n");
	return 0;
}

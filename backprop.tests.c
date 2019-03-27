#include "backprop.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define TOLERANCE 0.001
#define FLOAT_EQ(x, y) ((x) < (y) ? (y) - (x) < TOLERANCE : (x) - (y) < TOLERANCE)

static double relu_deriv(double x)
{
	if (x < 0)
		return 0;
	else
		return 1;
}

static void bias_change_for_a_single_neuron_can_be_calculated(void)
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
	const double desired_activation = 30;
	const double activations = 5;
	const double summed_cost = 25;

	/* when */
	const double expected_delta = 50.0; /* calculated by hand */
	double bias_delta = 0;
	bool success = get_bias_deltas(layer, inputs, &desired_activation, &activations, relu_deriv, &bias_delta, summed_cost);

	/* then */
	assert(success);
	assert(FLOAT_EQ(bias_delta, expected_delta));
	
	printf("bias_change_for_a_single_neuron_can_be_calculated: passed\n");
}

static void input_change_for_a_single_neuron_with_a_single_input(void)
{
	/* given */
	double weights[] = { 0.2 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 1,
		.activation_fn = &relu,
	};
	memcpy(layer.weights, weights, sizeof(double) * layer.input_size * layer.neuron_count);
	memcpy(layer.biases, biases, sizeof(double) * layer.neuron_count);

	double inputs[] = { 5 };
	const double desired_activation = 30;
	const double activations = 5;
	const double summed_cost = 25;

	/* when */
	const double expected_delta = 10; /* calculated by hand */
	double input_delta = 0;
	bool success = get_input_deltas(layer, inputs, &desired_activation, &activations, relu_deriv, &input_delta, summed_cost);

	/* then */
	assert(success);
	assert(FLOAT_EQ(input_delta, expected_delta));
	
	printf("input_change_for_a_single_neuron_with_a_single_input: passed\n");
}

static void weight_change_for_a_single_neuron_with_a_single_input(void)
{
	/* given */
	double weights[] = { 0.2 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 1,
		.activation_fn = &relu,
	};
	memcpy(layer.weights, weights, sizeof(double) * layer.input_size * layer.neuron_count);
	memcpy(layer.biases, biases, sizeof(double) * layer.neuron_count);

	double inputs[] = { 5 };
	const double desired_activation = 30;
	const double activations = 5;
	const double summed_cost = 25;

	/* when */
	const double expected_delta = 250; /* calculated by hand */
	double weight_delta = 0;
	bool success = get_weight_deltas(layer, inputs, &desired_activation, &activations, relu_deriv, &weight_delta, summed_cost);

	/* then */
	assert(success);
	assert(FLOAT_EQ(weight_delta, expected_delta));
	
	printf("weight_change_for_a_single_neuron_with_a_single_input: passed\n");
}

static void input_change_for_a_single_neuron_with_multiple_inputs(void)
{
	/* given */
	double weights[] = { 0.1, 0.2, 0.3 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 3,
		.activation_fn = &relu,
	};
	memcpy(layer.weights, weights, sizeof(double) * layer.input_size * layer.neuron_count);
	memcpy(layer.biases, biases, sizeof(double) * layer.neuron_count);

	double inputs[] = { 10, 20, 30 };
	const double desired_activation = 30;
	const double activations = 5;
	const double summed_cost = 25;

	/* when */
	const double expected_delta = 30; /* calculated by hand */
	double input_delta = 0.0;
	bool success = get_input_deltas(layer, inputs, &desired_activation, &activations, relu_deriv, &input_delta, summed_cost);

	/* then */
	assert(success);
	assert(FLOAT_EQ(input_delta, expected_delta));
	
	printf("input_change_for_a_single_neuron_with_multiple_inputs: passed\n");
}

static void weight_change_for_a_single_neuron_with_multiple_inputs(void)
{
	/* given */
	double weights[] = { 0.1, 0.2, 0.3 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 3,
		.activation_fn = &relu,
	};
	memcpy(layer.weights, weights, sizeof(double) * layer.input_size * layer.neuron_count);
	memcpy(layer.biases, biases, sizeof(double) * layer.neuron_count);

	double inputs[] = { 10, 20, 30 };
	const double desired_activation = 30;
	const double activations = 5;
	const double summed_cost = 25;

	/* when */
	const double expected_delta = 3000; /* calculated by hand */
	double weight_delta = 0;
	bool success = get_weight_deltas(layer, inputs, &desired_activation, &activations, relu_deriv, &weight_delta, summed_cost);

	/* then */
	assert(success);
	assert(FLOAT_EQ(weight_delta, expected_delta));
	
	printf("weight_change_for_a_single_neuron_with_multiple_inputs: passed\n");
}

static void bias_change_for_a_layer_can_be_calculated(void)
{
	/* given */
	double weights[] = {
		0.1, 0.2, -0.3,
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
	const double desired_activations[] = { 30, 20, 10 };
	const double activations[] = { -5, 10, 15 };
	const double summed_cost = -40;

	/* when */
	const double expected_deltas[] = { 0, -80, -80 }; /* calculated by hand */
	double bias_deltas[] = { 0, 0, 0 };
	bool success = get_bias_deltas(layer, inputs, desired_activations, activations, relu_deriv, bias_deltas, summed_cost);

	/* then */
	assert(success);
	for (unsigned index = 0; index < layer.neuron_count; index++)
	{
		assert(FLOAT_EQ(bias_deltas[index], expected_deltas[index]));
	}
	
	printf("bias_change_for_a_layer_can_be_calculated: passed\n");
}

static void input_change_for_a_layer_can_be_calculated(void)
{
	/* given */
	double weights[] = {
		0.1, 0.2, -0.3,
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
	const double desired_activations[] = { 30, 20, 10 };
	const double activations[] = { -5, 10, 15 };
	const double summed_cost = -40;

	/* when */
	const double expected_deltas[] = { 0, -120, -192 }; /* calculated by hand */
	double input_deltas[] = { 0, 0, 0 };
	bool success = get_input_deltas(layer, inputs, desired_activations, activations, relu_deriv, input_deltas, summed_cost);

	/* then */
	assert(success);
	for (unsigned index = 0; index < layer.neuron_count; index++)
	{
		assert(FLOAT_EQ(input_deltas[index], expected_deltas[index]));
	}
	
	printf("input_change_for_a_layer_can_be_calculated: passed\n");
}

static void weight_change_for_a_layer_can_be_calculated(void)
{
	/* given */
	double weights[] = {
		0.1, 0.2, -0.3,
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
	const double desired_activations[] = { 30, 20, 10 };
	const double activations[] = { -5, 10, 15 };
	const double summed_cost = -40;

	/* when */
	const double expected_deltas[] = { 0, -4800, -4800 }; /* calculated by hand */
	double weight_deltas[] = { 0, 0, 0 };
	bool success = get_weight_deltas(layer, inputs, desired_activations, activations, relu_deriv, weight_deltas, summed_cost);

	/* then */
	assert(success);
	for (unsigned index = 0; index < layer.neuron_count; index++)
	{
		assert(FLOAT_EQ(weight_deltas[index], expected_deltas[index]));
	}
	
	printf("weight_change_for_a_layer_can_be_calculated: passed\n");
}

int main(void)
{
	bias_change_for_a_single_neuron_can_be_calculated();
	input_change_for_a_single_neuron_with_a_single_input();
	weight_change_for_a_single_neuron_with_a_single_input();
	input_change_for_a_single_neuron_with_multiple_inputs();
	weight_change_for_a_single_neuron_with_multiple_inputs();

	bias_change_for_a_layer_can_be_calculated();
	input_change_for_a_layer_can_be_calculated();
	weight_change_for_a_layer_can_be_calculated();

	printf("all tests passed\n");
	return 0;
}

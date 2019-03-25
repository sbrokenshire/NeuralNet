#include "backprop.h"

#include <assert.h>
#include <stdio.h>

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
		.weights = weights,
		.biases = biases,
		.activation_fn = &relu
	};
	double inputs[] = { 10, 20, 30 };
	const double desired_activation = 30;

	/* when */
	const double expected_delta = -31.0; /* calculated by hand */
	double bias_delta = 0;
	bool success = get_bias_deltas(layer, inputs, &desired_activation, relu_deriv, &bias_delta);

	/* then */
	assert(success);
	assert(FLOAT_EQ(bias_delta, expected_delta));
}

static void input_change_for_a_single_neuron_with_a_single_input(void)
{
	/* given */
	double weights[] = { 0.2 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 1,
		.weights = weights,
		.biases = biases,
		.activation_fn = &relu,
	};
	double inputs[] = { 5 };
	const double desired_activation = 30;

	/* when */
	const double expected_delta = -11.4; /* calculated by hand */
	double input_delta = 0;
	bool success = get_input_deltas(layer, inputs, &desired_activation, relu_deriv, &input_delta);

	/* then */
	assert(success);
	assert(FLOAT_EQ(input_delta, expected_delta));
}

static void weight_change_for_a_single_neuron_with_a_single_input(void)
{
	/* given */
	double weights[] = { 0.2 };
	double biases[] = { 0.5 };
	nn_layer layer = {
		.neuron_count = 1,
		.input_size = 1,
		.weights = weights,
		.biases = biases,
		.activation_fn = &relu,
	};
	double inputs[] = { 5 };
	const double desired_activation = 30;

	/* when */
	const double expected_delta = -285; /* calculated by hand */
	double weight_delta = 0;
	bool success = get_weight_deltas(layer, inputs, &desired_activation, relu_deriv, &weight_delta);

	/* then */
	assert(success);
	assert(FLOAT_EQ(weight_delta, expected_delta));
}

int main(void)
{
	bias_change_for_a_single_neuron_can_be_calculated();
	input_change_for_a_single_neuron_with_a_single_input();
	weight_change_for_a_single_neuron_with_a_single_input();

	printf("all tests passed\n");
	return 0;
}

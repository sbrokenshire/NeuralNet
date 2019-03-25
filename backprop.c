#include "backprop.h"

#define MAX_LAYER_SIZE 256

bool get_bias_deltas(nn_layer layer,
                     const double *inputs,
                     const double *desired_activations,
                     activation_fn activation_deriv,
                     double *deltas_out)
{
	if (layer.neuron_count > MAX_LAYER_SIZE)
		return false;

	double activations[MAX_LAYER_SIZE];
	get_activations(layer, inputs, activations);

	double cost = activations[0] - desired_activations[0];
	deltas_out[0] = 2 * activation_deriv(activations[0]) * cost;

	return true;
}

bool get_input_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      activation_fn activation_deriv,
                      double *deltas_out)
{
	if (layer.neuron_count > MAX_LAYER_SIZE)
		return false;

	double activations[MAX_LAYER_SIZE];
	get_activations(layer, inputs, activations);

	double cost = activations[0] - desired_activations[0];
	deltas_out[0] = 2 * layer.weights[0] * activation_deriv(activations[0]) * cost;

	return true;
}

bool get_weight_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      activation_fn activation_deriv,
                      double *deltas_out)
{
	if (layer.neuron_count > MAX_LAYER_SIZE)
		return false;

	double activations[MAX_LAYER_SIZE];
	get_activations(layer, inputs, activations);

	double cost = activations[0] - desired_activations[0];
	deltas_out[0] = 2 * inputs[0] * activation_deriv(activations[0]) * cost;

	return true;
}

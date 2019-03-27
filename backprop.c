#include "backprop.h"

bool get_bias_deltas(nn_layer layer,
                     const double *inputs,
                     const double *desired_activations,
                     const double *activations,
                     activation_fn activation_deriv,
                     double *deltas_out,
                     double summed_costs)
{
	if (layer.neuron_count > MAX_LAYER_SIZE)
		return false;

	for (unsigned neuron_index = 0; neuron_index < layer.neuron_count; neuron_index++)
	{
		deltas_out[neuron_index] = 2 * activation_deriv(activations[neuron_index]) * summed_costs;
	}

	return true;
}

bool get_input_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      const double *activations,
                      activation_fn activation_deriv,
                      double *deltas_out,
                      double summed_costs)
{
	if (layer.neuron_count > MAX_LAYER_SIZE)
		return false;

	double summed_layer_weights = 0.0;
	for (unsigned neuron_index = 0; neuron_index < layer.neuron_count; neuron_index++)
	{
		summed_layer_weights = 0.0;
		for (unsigned weights_index = 0; weights_index < layer.input_size; weights_index++)
		{
			unsigned weight_array_index = neuron_index * layer.input_size + weights_index;
			summed_layer_weights += layer.weights[weight_array_index];
		}

		deltas_out[neuron_index] = 2 * summed_layer_weights * activation_deriv(activations[neuron_index]) * summed_costs;
	}

	return true;
}

bool get_weight_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      const double *activations,
                      activation_fn activation_deriv,
                      double *deltas_out,
                      double summed_costs)
{
	if (layer.neuron_count > MAX_LAYER_SIZE)
		return false;

	double summed_input_activations = 0.0;
	for (unsigned input_index = 0; input_index < layer.input_size; input_index++)
	{
		summed_input_activations += inputs[input_index];
	}

	for (unsigned neuron_index = 0; neuron_index < layer.neuron_count; neuron_index++)
	{
		deltas_out[neuron_index] = 2 * summed_input_activations * activation_deriv(activations[neuron_index]) * summed_costs;
	}

	return true;
}

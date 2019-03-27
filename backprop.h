#ifndef BACKPROP_H
#define BACKPROP_H

#include "nn.h"

#include <stdbool.h>

bool get_bias_deltas(nn_layer layer,
                     const double *inputs,
                     const double *desired_activations,
                     const double *activations,
                     activation_fn activation_deriv,
                     double *deltas_out,
                     double summed_costs);

bool get_input_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      const double *activations,
                      activation_fn activation_deriv,
                      double *deltas_out,
                      double summed_costs);

bool get_weight_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      const double *activations,
                      activation_fn activation_deriv,
                      double *deltas_out,
                      double summed_costs);

#endif

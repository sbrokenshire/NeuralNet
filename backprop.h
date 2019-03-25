#ifndef BACKPROP_H
#define BACKPROP_H

#include "nn.h"

#include <stdbool.h>

bool get_bias_deltas(nn_layer layer,
                     const double *inputs,
                     const double *desired_activations,
                     activation_fn activation_deriv,
                     double *deltas_out);

bool get_input_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      activation_fn activation_deriv,
                      double *deltas_out);

bool get_weight_deltas(nn_layer layer,
                      const double *inputs,
                      const double *desired_activations,
                      activation_fn activation_deriv,
                      double *deltas_out);

#endif

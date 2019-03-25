#ifndef NN_H
#define NN_H

typedef double (*activation_fn)(double);

typedef struct {
	unsigned neuron_count;
	unsigned input_size;
	double *weights;
	double *biases;
	activation_fn activation_fn;
} nn_layer;

void get_activations(nn_layer layer, const double *inputs, double *activations_out);

double relu(double x);

#endif

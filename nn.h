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

typedef struct {
	nn_layer layers[20];
	unsigned batch_size;
	unsigned layer_count;
} nn_net;

void nn_add_layer(nn_net *neural_net, unsigned neuron_count, activation_fn activation_fn);
void nn_net_init(nn_layer *layers, unsigned layer_count);
void get_activations(nn_layer layer, const double *inputs, double *activations_out);

double relu(double x);

#endif

#ifndef NN_H
#define NN_H

typedef double (*activation_fn)(double);

#define MAX_LAYER_SIZE 128
#define MAX_LAYER_WEIGHTS MAX_LAYER_SIZE*MAX_LAYER_SIZE

typedef struct {
	unsigned neuron_count;
	unsigned input_size;
	double weights[MAX_LAYER_WEIGHTS];
	double biases[MAX_LAYER_SIZE];
	activation_fn activation_fn;
} nn_layer;

typedef struct {
	nn_layer layers[10];
	unsigned batch_size;
	unsigned layer_count;
} nn_net;

void nn_net_init(nn_net *neural_net);
void nn_add_layer(nn_net *neural_net, unsigned neuron_count, activation_fn activation_fn);
void get_activations(nn_layer layer, const double *inputs, double *activations_out);

double relu(double x);

#endif

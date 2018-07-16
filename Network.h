#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    unsigned int layers_number;
    Layer **layers;
    float learning_rate;
} Network;

void createNetwork(const unsigned int layers_number, const unsigned int *layers_neurons, const float learning_rate, Network **created);

void network_feed_forward(const Network *network, Matrix *input, Matrix **output);

void train(Network *network, Matrix *input, Matrix *output);

void printNetwork(const Network *network, const unsigned int show_values);

void freeNetwork(Network *network);

#endif

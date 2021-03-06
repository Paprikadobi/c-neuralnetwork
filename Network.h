#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    unsigned int layers_number;
    Layer **layers;
    float learning_rate;
} Network;

void create_network(const unsigned int layers_number, Layer **layers, const float learning_rate, Network **created);

void network_feed_forward(const Network *network, Matrix *input, Matrix **output);

void train(Network *network, const Data *training_data);

void print_network(const Network *network, const unsigned int show_values);

void free_network(Network *network);

#endif

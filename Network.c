#include <stdlib.h>
#include <stdio.h>

#include "Matrix.h"
#include "Layer.h"
#include "Network.h"

void create_network(const unsigned int layers_number, const unsigned int *layers_neurons, const float learning_rate, Network **created) {
    Network *network = malloc(sizeof(Network));
    network->layers_number = layers_number;
    network->layers = malloc(sizeof(Layer) * layers_number);
    network->learning_rate = learning_rate;
    for(size_t i = layers_number; i--;) {
        create_layer(layers_neurons[i], layers_neurons[i + 1], &(network->layers[i]));
    }
    *created = network;
}

void network_feed_forward(const Network *network, Matrix *input, Matrix **output) {
    Matrix *tmp;
    for(size_t i = 0; i < network->layers_number; i++) {
        feed_forward(network->layers[i], input, &tmp);
        input = tmp;
    }
    *output = tmp;
}

void train(Network *network, Matrix *input, Matrix *output) {
    Matrix *guess, *error;
    network_feed_forward(network, input, &guess);
    multiply(guess, -1);
    error = guess;
    matrix_addition(output, error);
    multiply(error, network->learning_rate);
   for(size_t i = network->layers_number; i--;) {
        update(network->layers[i], &error);
    }
}

void print_network(const Network *network, const unsigned int show_values) {
    for(size_t i = 0; i < network->layers_number; i++) {
        printf("Layer %zu: \n\n", i);
        print_layer(network->layers[i], show_values);
    }
}

void free_network(Network *network) {
    for(size_t i = 0; i < network->layers_number; i++) {
        free_layer(network->layers[i]);
    }
    free(network);
}

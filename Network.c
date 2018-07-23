#include <stdlib.h>
#include <stdio.h>

#include "Matrix.h"
#include "Data.h"
#include "Layer.h"
#include "Network.h"

void create_network(const unsigned int layers_number, Layer **layers, const float learning_rate, Network **created) {
    Network *network = malloc(sizeof(Network));
    network->layers_number = layers_number;
    network->layers = layers;
    network->learning_rate = learning_rate;
    *created = network;
}

void network_feed_forward(const Network *network, Matrix *input, Matrix **output) {
    Matrix *tmp;
    Matrix *inp;
    copy(input, &inp);
    for(size_t i = 0; i < network->layers_number; i++) {
        switch (network->layers[i]->type) {
            case FULLY_CONNECTED_LAYER:
                feed_forward(network->layers[i], inp, &tmp);
                break;
            case FILTER_LAYER:
                filter(network->layers[i], inp, &tmp);
                break;
            case POOLING_LAYER:
                pool(network->layers[i], inp, &tmp);
                break;
        }
        inp = tmp;
    }
    *output = tmp;
}

void train(Network *network, const Data *training_data) {
    Matrix **inputs = training_data->x;
    Matrix **outputs = training_data->y;
    Matrix *guess_error;
    printf("Training\n");
    for(size_t i = training_data->count; i--;) {
        network_feed_forward(network, inputs[i], &guess_error);
        multiply(guess_error, -1);
        matrix_addition(outputs[i], guess_error);
        multiply(guess_error, network->learning_rate);
        for(size_t j = network->layers_number; j--;) {
            switch(network->layers[j]->type) {
                case FULLY_CONNECTED_LAYER:
                    update(network->layers[j], &guess_error);
                    break;
                case FILTER_LAYER:
                    update_filter(network->layers[j], &guess_error);
                    break;
                case POOLING_LAYER:
                    update_pool(network->layers[j], &guess_error);
                    break;
            }
        }
        free_matrix(guess_error);
    }
    printf("Training ended\n");
}

void print_network(const Network *network, const unsigned int show_values) {
    printf("Learning rate: %.3f\n\n", network->learning_rate);
    for(size_t i = 0; i < network->layers_number; i++) {
        printf("Layer %zu: %s\n", i, layer_description[network->layers[i]->type]);
        print_layer(network->layers[i], show_values);
    }
}

void free_network(Network *network) {
    for(size_t i = 0; i < network->layers_number; i++) {
        free_layer(network->layers[i]);
    }
    free(network);
}

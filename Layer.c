#include <stdio.h>
#include <stdlib.h>

#include "Matrix.h"
#include "Layer.h"

void create_layer(const unsigned int inputs, const unsigned int outputs, Layer **created) {
    Layer *layer = malloc(sizeof(Layer));
    layer->inputs = inputs;
    layer->outputs = outputs;
    create_matrix(outputs, inputs, &(layer->weights));
    create_matrix(outputs, 1, &(layer->biases));
    randomize(-1, 1, layer->weights);
    randomize(-1, 1, layer->biases);
    *created = layer;
}

void feed_forward(Layer *layer, Matrix *input, Matrix **output) {
    Matrix *result;
    layer->input = input;
    matrix_multiplication(layer->weights, input, &result);
    matrix_addition(layer->biases, result);
    *output = result;
}

void update(Layer *layer, Matrix **error) {
    Matrix *input_T, *delta_W, *weights_T, *new_error;
    transpose(layer->input, &input_T);
    matrix_multiplication(*error, input_T, &delta_W);
    matrix_addition(delta_W, layer->weights);
    matrix_addition(*error, layer->biases);
    transpose(layer->weights, &weights_T);
    matrix_multiplication(weights_T, *error, &new_error);
    
    free_matrix(*error);
    free_matrix(input_T);
    free_matrix(delta_W);
    free_matrix(weights_T);
    
    *error = new_error;
}

void print_layer(const Layer *layer, const unsigned int show_values) {
    printf("Inputs: %d, Outputs: %d\n\n", layer->inputs, layer->outputs);
    if(show_values) {
        printf("Weights: \n");
        print_matrix(layer->weights);
        printf("Biases: \n");
        print_matrix(layer->biases);
    }
}

void free_layer(Layer *layer) {
    free_matrix(layer->weights);
    free_matrix(layer->biases);
    free(layer);
}

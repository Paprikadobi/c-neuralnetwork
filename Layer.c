#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"
#include "Layer.h"

void createLayer(const unsigned int inputs, const unsigned int outputs, Layer **created) {
    Layer *layer = malloc(sizeof(Layer));
    layer->inputs = inputs;
    layer->outputs = outputs;
    createMatrix(outputs, inputs, &(layer->weights));
    createMatrix(outputs, 1, &(layer->biases));
    randomize(-1, 1, layer->weights);
    randomize(-1, 1, layer->biases);
    *created = layer;
}

void feed_forward(Layer *layer, Matrix *input, Matrix **output) {
    Matrix *result;
    layer->input = input;
    matrixMultiplication(layer->weights, input, &result);
    matrixAddition(layer->biases, result);
    *output = result;
}

void update(Layer *layer, Matrix **error) {
    Matrix *input_T, *delta_W, *weights_T, *new_error;
    transpose(layer->input, &input_T);
    matrixMultiplication(*error, input_T, &delta_W);
    matrixAddition(delta_W, layer->weights);
    matrixAddition(*error, layer->biases);
    transpose(layer->weights, &weights_T);
    matrixMultiplication(weights_T, *error, &new_error);
    
    freeMatrix(*error);
    freeMatrix(input_T);
    freeMatrix(delta_W);
    freeMatrix(weights_T);
    
    *error = new_error;
}

void printLayer(const Layer *layer, const unsigned int show_values) {
    printf("Inputs: %d, Outputs: %d\n\n", layer->inputs, layer->outputs);
    if(show_values) {
        printf("Weights: \n");
        printMatrix(layer->weights);
        printf("Biases: \n");
        printMatrix(layer->biases);
    }
}

void freeLayer(Layer *layer) {
    freeMatrix(layer->weights);
    freeMatrix(layer->biases);
    free(layer);
}

#ifndef LAYER_H
#define LAYER_H

typedef struct {
    unsigned int inputs;
    unsigned int outputs;
    Matrix *input;
    Matrix *weights;
    Matrix *biases;
} Layer;

void createLayer(const unsigned int inputs, const unsigned int outputs, Layer **created);

void feed_forward(Layer *layer, Matrix *input, Matrix **output);

void update(Layer *layer, Matrix **error);

void printLayer(const Layer *layer, const unsigned int show_values);

void freeLayer(Layer *layer);

#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Matrix.h"
#include "Layer.h"

const char *layer_description[3] = {"Fully connected layer", "Filter layer", "Pooling layer"};

void create_layer(Shape *input_shape, Shape *output_shape, Layer **created) {
    Layer *layer = malloc(sizeof(Layer));
    layer->input_shape = input_shape;
    layer->output_shape = output_shape;
    layer->input = NULL;
    layer->output = NULL;
    *created = layer;
}

void create_fully_connected_layer(float (*activation)(float), float (*derivative)(float), Layer *layer) {
    Fully_connected_layer *fully_connected = malloc(sizeof(Fully_connected_layer));
    fully_connected->activation = activation;
    fully_connected->derivative = derivative;
    fully_connected->weights = create_matrix(create_shape(1, layer->output_shape->size, layer->input_shape->size));
    fully_connected->biases = create_matrix(create_shape(1, layer->output_shape->size, 1));
    randomize(-1, 1, fully_connected->weights);
    randomize(-1, 1, fully_connected->biases);
    layer->fully_connected = fully_connected;
    layer->type = FULLY_CONNECTED_LAYER;
}

void create_filter_layer(Shape *filter_shape, Layer *layer) {
    Filter_layer *filter = malloc(sizeof(Filter_layer));
    filter->weights = create_matrix(filter_shape);
    filter->bias = create_matrix(create_shape(filter_shape->count, 1, 1));
    randomize(-1, 1, filter->weights);
    randomize(-1, 1, filter->bias);
    layer->filter = filter;
    layer->type = FILTER_LAYER;
}

void create_pooling_layer(Shape *shape, Layer *layer) {
    Pooling_layer *pool = malloc(sizeof(Pooling_layer));
    pool->pooling_shape = shape;
    layer->pool = pool;
    layer->type = POOLING_LAYER;
}

void feed_forward(Layer *layer, Matrix *input, Matrix **output) {
    Fully_connected_layer *fully_connected = layer->fully_connected;
    Matrix *result;
    if(layer->input)
        free_matrix(layer->input);
    layer->input = input;
    reshape(input, layer->input_shape);
    matrix_multiplication(fully_connected->weights, input, &result);
    matrix_addition(fully_connected->biases, result);
    map(result, fully_connected->activation);
    reshape(result, layer->output_shape);
    if(layer->output)
        free_matrix(layer->output);
    layer->output = result;
    copy(result, output);
}

void filter(Layer *layer, Matrix *input, Matrix **output) {
    Filter_layer *filter = layer->filter;
    if(layer->input)
        free_matrix(layer->input);
    layer->input = input;
    reshape(input, layer->input_shape);
    Shape *shape = create_shape(input->shape->count * filter->weights->shape->count, input->shape->rows - filter->weights->shape->rows + 1, input->shape->columns - filter->weights->shape->columns + 1);
    Matrix *result = create_matrix(shape);
    set(0, result);
    
    for(size_t i = 0; i < input->shape->count; i++) {
        for(size_t j = 0; j < result->shape->rows; j++) {
            for(size_t k = 0; k < result->shape->columns; k++) {
                for(size_t l = 0; l < filter->weights->shape->count; l++) {
                    for(size_t m = 0; m < filter->weights->shape->rows; m++) {
                        for(size_t n = 0; n < filter->weights->shape->columns; n++) {
                            result->data[((i * filter->weights->shape->rows + l) * result->shape->rows + j) * result->shape->columns + k] = input->data[(i * input->shape->rows + j + m) * input->shape->columns + k + n] * filter->weights->data[(l * filter->weights->shape->rows + m) * filter->weights->shape->columns + n];
                        }
                    }
                result->data[((i * filter->weights->shape->rows + l) * result->shape->rows + j) * result->shape->columns + k] += filter->bias->data[l];
                }
            }
        }
    }
    reshape(result, layer->output_shape);
    if(layer->output)
        free_matrix(layer->output);
    layer->output = result;
    copy(result, output);
}

void pool(Layer *layer, Matrix *input, Matrix **output) {
    Pooling_layer *pool = layer->pool;
    if(layer->input)
        free_matrix(layer->input);
    layer->input = input;
    reshape(input, layer->input_shape);
    Shape *shape = create_shape(input->shape->count, input->shape->rows / pool->pooling_shape->rows, input->shape->columns / pool->pooling_shape->columns);
    Matrix *result = create_matrix(shape);
    for(size_t c = 0; c < input->shape->count; c++) {
        for(size_t i = 0; i < input->shape->rows; i += pool->pooling_shape->rows) {
            for(size_t j = 0; j < input->shape->columns; j += pool->pooling_shape->columns) {
                float max = 0;
                for(size_t k = 0; k < pool->pooling_shape->rows; k++) {
                    for(size_t l = 0; l < pool->pooling_shape->columns; l++) {
                        float value = input->data[(c * input->shape->rows + i + k) * input->shape->columns + j + l];
                        if(!max || max < value) {
                            max = value;
                        }
                    }
                }
                result->data[((c * input->shape->rows + i) / pool->pooling_shape->rows) * (input->shape->columns / pool->pooling_shape->columns) + (j / pool->pooling_shape->columns)] = max;
            }
        }
    }
    reshape(result, layer->output_shape);
    if(layer->output)
        free_matrix(layer->output);
    layer->output = result;
    copy(result, output);
}

void update(Layer *layer, Matrix **error) {
    Fully_connected_layer *fully_connected = layer->fully_connected;
    Matrix *input_T, *delta_W, *weights_T, *new_error;
    map(layer->output, fully_connected->derivative);
    matrix_mult(*error, layer->output);
    transpose(fully_connected->weights, &weights_T);
    matrix_multiplication(weights_T, *error, &new_error);
    transpose(layer->input, &input_T);
    matrix_multiplication(*error, input_T, &delta_W);
    matrix_addition(delta_W, fully_connected->weights);
    matrix_addition(*error, fully_connected->biases);
    free_matrix(*error);
    free_matrix(input_T);
    free_matrix(delta_W);
    free_matrix(weights_T);
    *error = new_error;
}

void update_filter(Layer *layer, Matrix **error) {
    Filter_layer *filter = layer->filter;
    Matrix *new_error;
    Matrix *delta_W = create_matrix(filter->weights->shape);
    set(0, delta_W);
    float *delta_b = malloc(sizeof(float) * (*error)->shape->count);
    for(size_t i = (*error)->shape->count; i--;)
        delta_b[i] = 0;
    Matrix *input = layer->input;
    Matrix *result = layer->output;
    
    for(size_t i = 0; i < input->shape->count; i++) {
        for(size_t j = 0; j < result->shape->rows; j++) {
            for(size_t k = 0; k < result->shape->columns; k++) {
                for(size_t l = 0; l < filter->weights->shape->count; l++) {
                    for(size_t m = 0; m < filter->weights->shape->rows; m++) {
                        for(size_t n = 0; n < filter->weights->shape->columns; n++) {
                            delta_W->data[(l * filter->weights->shape->rows + m) * filter->weights->shape->columns + n] += (*error)->data[((i * filter->weights->shape->rows + l) * result->shape->rows + j) * result->shape->columns + k] * input->data[(i * input->shape->rows + j + m) * input->shape->columns + k + n];
                        }
                    }
                    delta_b[l] += (*error)->data[((i * filter->weights->shape->rows + l) * result->shape->rows + j) * result->shape->columns + k];
                }
            }
        }
    }
    
    add(filter->bias, delta_b);
    matrix_addition(delta_W, filter->weights);
    normalize(filter->weights, 1);
    
    free_matrix(delta_W);
    
//    free_matrix(*error);
//    *error = new_error;
}

void update_pool(Layer *layer, Matrix **error) {
    Pooling_layer *pool = layer->pool;
    Matrix *new_error = create_matrix(layer->input_shape);
    Matrix *input = layer->input;
    Matrix *output = layer->output;
    set(0, new_error);
    reshape(*error, layer->output_shape);
    
    for(size_t i = 0; i < output->shape->count; i++) {
        for(size_t j = 0; j < output->shape->rows; j++) {
            for(size_t k = 0; k < output->shape->columns; k++) {
                for(size_t l = 0; l < pool->pooling_shape->rows; l++) {
                    for(size_t m = 0; m < pool->pooling_shape->columns; m++) {
                        if(input->data[(i * input->shape->rows + pool->pooling_shape->rows * j + l) * input->shape->columns + pool->pooling_shape->columns * k + m] == output->data[(i * output->shape->rows + j) * output->shape->columns + k]) {
                            new_error->data[(i * input->shape->rows + pool->pooling_shape->rows * j + l) * input->shape->columns + pool->pooling_shape->columns * k + m] = (*error)->data[(i * output->shape->rows + j) * output->shape->columns + k];
                            l = pool->pooling_shape->rows;
                            m = pool->pooling_shape->columns;
                        }
                    }
                }
            }
        }
    }
    
    free_matrix(*error);
    *error = new_error;
}

void print_layer(const Layer *layer, const unsigned int show_values) {
    printf("Input shape: ");
    print_shape(layer->input_shape);
    printf("Output shape: ");
    print_shape(layer->output_shape);
    if(show_values) {
        switch (layer->type) {
            case FULLY_CONNECTED_LAYER:
                if(show_values) {
                    printf("Weights: \n");
                    print_matrix(layer->fully_connected->weights);
                    printf("Biases: \n");
                    print_matrix(layer->fully_connected->biases);
                }
                break;
            case FILTER_LAYER:
                if(show_values) {
                    printf("Weights: \n");
                    print_matrix(layer->filter->weights);
                    printf("Biases: \n");
                    print_matrix(layer->filter->bias);
                }
                break;
            case POOLING_LAYER:
                printf("Pooling shape: ");
                print_shape(layer->pool->pooling_shape);
                break;
        }
    }
    printf("\n");
}

void free_layer(Layer *layer) {
    free(layer->input_shape);
    free(layer->output_shape);
    free_matrix(layer->input);
    free_matrix(layer->output);
    switch (layer->type) {
        case FULLY_CONNECTED_LAYER:
            free_matrix(layer->fully_connected->weights);
            free_matrix(layer->fully_connected->biases);
            break;
        case FILTER_LAYER:
            free_matrix(layer->filter->weights);
            free_matrix(layer->filter->bias);
            break;
        case POOLING_LAYER:
            free(layer->pool->pooling_shape);
            break;
    }
    free(layer);
}

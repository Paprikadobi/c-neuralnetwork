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
    create_matrix(layer->output_shape->size, layer->input_shape->size, &(fully_connected->weights));
    create_matrix(layer->output_shape->size, 1, &(fully_connected->biases));
    randomize(-1, 1, fully_connected->weights);
    randomize(-1, 1, fully_connected->biases);
    layer->fully_connected = fully_connected;
    layer->type = FULLY_CONNECTED_LAYER;
}

void create_filter_layer(Shape *filter_shape, Layer *layer) {
    Filter_layer *filter = malloc(sizeof(Filter_layer));
    create_matrix(filter_shape->rows, filter_shape->columns, &(filter->weights));
    create_matrix(1, 1, &(filter->bias));
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
    matrix_multiplication(fully_connected->weights, input, &result);
    matrix_addition(fully_connected->biases, result);
    map(result, fully_connected->activation);
    if(layer->output)
        free_matrix(layer->output);
    layer->output = result;
    copy(result, output);
}

void filter(Layer *layer, Matrix *input, Matrix **output) {
    Filter_layer *filter = layer->filter;
    Matrix *result;
    if(layer->input)
        free_matrix(layer->input);
    layer->input = input;
    reshape(input, layer->input_shape);
    filter_matrix(input, filter->weights, &result);
    add(result, filter->bias->data[0]);
    reshape(result, layer->output_shape);
    if(layer->output)
        free_matrix(layer->output);
    layer->output = result;
    copy(result, output);
}

void pool(Layer *layer, Matrix *input, Matrix **output) {
    Pooling_layer *pool = layer->pool;
    Matrix *result;
    if(layer->input)
        free_matrix(layer->input);
    layer->input = input;
    reshape(input, layer->input_shape);
    create_matrix(input->shape->rows / pool->pooling_shape->rows, input->shape->columns / pool->pooling_shape->columns, &result);
    for(size_t i = 0; i < input->shape->rows; i += pool->pooling_shape->rows) {
        for(size_t j = 0; j < input->shape->columns; j += pool->pooling_shape->columns) {
            float max = 0;
            for(size_t k = 0; k < pool->pooling_shape->rows; k++) {
                for(size_t l = 0; l < pool->pooling_shape->columns; l++) {
                    float value = input->data[(i + k) * input->shape->columns + j + l];
                    if(!max || max < value) {
                        max = value;
                    }
                }
            }
            result->data[(i / pool->pooling_shape->rows) * (input->shape->columns / pool->pooling_shape->columns) + (j / pool->pooling_shape->columns)] = max;
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
    Matrix *delta_W;
    create_matrix(filter->weights->shape->rows, filter->weights->shape->columns, &delta_W);
    set(0, delta_W);
    float delta_b = 0;
    
    for(size_t i = 0; i < (*error)->shape->rows; i++) {
        for(size_t j = 0; j < (*error)->shape->columns; j++) {
            delta_b += (*error)->data[i * (*error)->shape->columns + j];
            for(size_t k = 0; k < filter->weights->shape->rows; k++) {
                for(size_t l = 0; l < filter->weights->shape->columns; l++) {
                    delta_W->data[k * filter->weights->shape->columns + l] += (*error)->data[i * (*error)->shape->columns + j] * layer->input->data[(i + k) * layer->input_shape->columns + j + l];
                }
            }
        }
    }
    
    add(filter->bias, delta_b);
    matrix_addition(delta_W, filter->weights);
    
    free_matrix(delta_W);
    
//    free_matrix(*error);
//    *error = new_error;
}

void update_pool(Layer *layer, Matrix **error) {
    Pooling_layer *pool = layer->pool;
    Matrix *new_error;
    Matrix *input = layer->input;
    create_matrix(layer->input_shape->rows, layer->input_shape->columns, &new_error);
    set(0, new_error);
    reshape(*error, layer->output_shape);
    for(size_t i = 0; i < input->shape->rows; i += pool->pooling_shape->rows) {
        for(size_t j = 0; j < input->shape->columns; j += pool->pooling_shape->columns) {
            for(size_t k = 0; k < pool->pooling_shape->rows; k++) {
                for(size_t l = 0; l < pool->pooling_shape->columns; l++) {
                    if(layer->input->data[(i + k) * input->shape->columns + j + l] == layer->output->data[(i / pool->pooling_shape->rows) * (input->shape->columns / pool->pooling_shape->columns) + (j / pool->pooling_shape->columns)]) {
                        k = pool->pooling_shape->rows;
                        l = pool->pooling_shape->columns;
                        new_error->data[(i + k) * input->shape->columns + j + l] = (*error)->data[(i / pool->pooling_shape->rows) * (input->shape->columns / pool->pooling_shape->columns) + (j / pool->pooling_shape->columns)];
                        
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

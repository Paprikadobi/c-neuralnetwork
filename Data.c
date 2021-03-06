#include <stdio.h>
#include <stdlib.h>

#include "Matrix.h"
#include "Data.h"

void create_data(const unsigned int count, const unsigned int input_rows, float **inputs, const unsigned int output_rows, float **outputs, Data **created) {
    Data *data = malloc(sizeof(Data));
    data->count = count;
    data->x = malloc(sizeof(Matrix) * count);
    data->y = malloc(sizeof(Matrix) * count);
    for(size_t i = count; i--;) {
        data->x[i] = create_matrix(create_shape(1, input_rows, 1));
        data->y[i] = create_matrix(create_shape(1, output_rows, 1));
        data->x[i]->data = inputs[i];
        data->y[i]->data = outputs[i];
    }
    free(inputs);
    free(outputs);
    *created = data;
}

void print_data(const Data *data, const unsigned int show_values) {
    printf("Count: %d\n", data->count);
    if(show_values) {
        for(size_t i = 0; i < data->count; i++) {
            printf("Input: \n");
            print_matrix(data->x[i]);
            printf("Output: \n");
            print_matrix(data->y[i]);
        }
    }
}

void free_data(Data *data) {
    for(size_t i = data->count; i--;) {
        free_matrix(data->x[i]);
        free_matrix(data->y[i]);
    }
    free(data->x);
    free(data->y);
    free(data);
}

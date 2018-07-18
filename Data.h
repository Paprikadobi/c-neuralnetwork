#ifndef DATA_H
#define DATA_H

typedef struct {
    unsigned int count;
    Matrix **x;
    Matrix **y;
} Data;

void create_data(const unsigned int count, const unsigned int input_rows, float **inputs, const unsigned int output_rows, float **outputs, Data **created);

void print_data(const Data *data, const unsigned int show_values);

void free_data(Data *data);

#endif

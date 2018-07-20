#ifndef MATRIX_H
#define MATRIX_H

#define BLK_SIZE 128
#define min(a, b) (a < b ? a : b)

typedef struct {
    float *data;
    unsigned int rows;
    unsigned int columns;
} Matrix;

void create_matrix(const unsigned int rows, const unsigned int columns, Matrix **created);

void set(const float num, Matrix *matrix);

void randomize(const float min, const float max, Matrix *matrix);

void add(Matrix *matrix, const float num);

unsigned int matrix_addition(const Matrix *a, Matrix *b);

void multiply(Matrix *matrix, const float num);

unsigned int matrix_mult(Matrix *a, const Matrix *b);

unsigned int matrix_multiplication(const Matrix *a, const Matrix *b, Matrix **c);

unsigned int reshape(Matrix *matrix, const unsigned int rows, const unsigned int columns);

void filter_matrix(const Matrix *matrix, const Matrix *filter, Matrix **created);

void transpose(const Matrix *a, Matrix **a_T);

void map(Matrix *matrix, float (*function)(float));

void copy(const Matrix *a, Matrix **created);

void print_matrix(const Matrix *matrix);

void free_matrix(Matrix *matrix);

unsigned int matrixes_equals(Matrix *first, Matrix *second);

#endif

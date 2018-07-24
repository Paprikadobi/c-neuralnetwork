#ifndef MATRIX_H
#define MATRIX_H

#define BLK_SIZE 128
#define min(a, b) (a < b ? a : b)

typedef struct {
    unsigned int size;
    size_t count;
    size_t rows;
    size_t columns;
} Shape;

typedef struct {
    float *data;
    Shape *shape;
} Matrix;

Shape *create_shape(const size_t count, const size_t rows, const size_t columns);

void print_shape(Shape *shape);

unsigned int shapes_equals(Shape *shape1, Shape *shape2);

Matrix *create_matrix(Shape *shape);

void set(const float num, Matrix *matrix);

void randomize(const float min, const float max, Matrix *matrix);

void add(Matrix *matrix, const float *nums);

unsigned int matrix_addition(const Matrix *a, Matrix *b);

void multiply(Matrix *matrix, const float num);

unsigned int matrix_mult(Matrix *a, const Matrix *b);

unsigned int matrix_multiplication(const Matrix *a, const Matrix *b, Matrix **c);

unsigned int reshape(Matrix *matrix, Shape *shape);

void normalize(Matrix *matrix, const float value);

void transpose(const Matrix *a, Matrix **a_T);

void map(Matrix *matrix, float (*function)(float));

void copy(const Matrix *a, Matrix **created);

void print_matrix(const Matrix *matrix);

void free_matrix(Matrix *matrix);

unsigned int matrixes_equals(Matrix *first, Matrix *second);

#endif

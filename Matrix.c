#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "Matrix.h"

#define TRUE 1
#define FALSE 0

Shape *create_shape(const size_t count, const size_t rows, const size_t columns) {
    Shape *shape = malloc(sizeof(Shape));
    shape->size = count * rows * columns;
    shape->count = count;
    shape->rows = rows;
    shape->columns = columns;
    return shape;
}

void print_shape(Shape *shape) {
    printf("(%zu, %zu, %zu)\n", shape->count, shape->rows, shape->columns);
}

unsigned int shapes_equals(Shape *shape1, Shape *shape2) {
    return shape1->count == shape2->count && shape1->rows == shape2->rows && shape1->columns == shape2->columns;
}

Matrix *create_matrix(Shape *shape) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->data = malloc(sizeof(float) * shape->size);
    matrix->shape = malloc(sizeof(Shape));
    matrix->shape->size = shape->size;
    matrix->shape->count = shape->count;
    matrix->shape->rows = shape->rows;
    matrix->shape->columns = shape->columns;
    return matrix;
}

void set(const float num, Matrix *matrix) {
    for(size_t i = 0; i < matrix->shape->size; i++)
            matrix->data[i] = num;
}

void randomize(const float min, const float max, Matrix *matrix) {
    for(size_t i = 0; i < matrix->shape->size; i++)
            matrix->data[i] = (((float) rand() / (float) RAND_MAX) - 0.5) * (max - min) + (max + min) * 0.5;
}

void add(Matrix *matrix, const float *nums) {
    for(size_t i = matrix->shape->count; i--;)
        for(size_t j = matrix->shape->rows * matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->rows * matrix->shape->columns + j] += nums[i];
}

unsigned int matrix_addition(const Matrix *a, Matrix *b) {
    if(!shapes_equals(a->shape, b->shape))
        return FALSE;
    for(size_t i = a->shape->size; i--;)
        b->data[i] += a->data[i];
    return TRUE;
}

void multiply(Matrix *matrix, const float num) {
    for(size_t i = matrix->shape->size; i--;)
        matrix->data[i] *= num;
}

unsigned int matrix_mult(Matrix *a, const Matrix *b) {
    if(!shapes_equals(a->shape, b->shape))
        return FALSE;
    for(size_t i = a->shape->size; i--;)
        a->data[i] *= b->data[i];
    return TRUE;
}

unsigned int matrix_multiplication(const Matrix *a, const Matrix *b, Matrix **c) {
    if(a->shape->columns != b->shape->rows)
        return FALSE;
    Shape *shape = create_shape(a->shape->count, a->shape->rows, b->shape->columns);
    Matrix *matrix = create_matrix(shape);
    set(0, matrix);
    for(size_t c = 0; c < shape->count; c++)
        for(size_t ii = 0; ii < shape->rows; ii += BLK_SIZE)
            for(size_t jj = 0; jj < a->shape->columns; jj += BLK_SIZE)
                for(size_t kk = 0; kk < b->shape->columns; kk += BLK_SIZE)
                    for(size_t i = ii; i < min(shape->rows, ii + BLK_SIZE); i++)
                        for(size_t j = jj; j < min(a->shape->columns, jj + BLK_SIZE); j++)
                            for(size_t k = kk; k < min(b->shape->columns, kk + BLK_SIZE); k++)
                                matrix->data[(c * shape->rows + i) * shape->columns + k] += a->data[(c * a->shape->rows + i) * a->shape->columns + j] * b->data[(c * b->shape->rows + j) * b->shape->columns + k];
    *c = matrix;
    return TRUE;
}

unsigned int reshape(Matrix *matrix, Shape *shape) {
    if(matrix->shape->size != shape->size)
        return FALSE;
    matrix->shape = create_shape(shape->count, shape->rows, shape->columns);
    return TRUE;
}

void normalize(Matrix *matrix, const float value) {
    Shape *shape = matrix->shape;
    for(size_t i = shape->count; i--;) {
        float sum = 0;
        for(size_t j = shape->rows; j--;)
            for(size_t k = shape->columns; k--;)
                sum += fabs(matrix->data[(i * shape->rows + j) * shape->columns + k]);
        for(size_t j = shape->rows; j--;)
            for(size_t k = shape->columns; k--;)
                matrix->data[(i * shape->rows + j) * shape->columns + k] *= value / sum;
    }
}

void transpose(const Matrix *a, Matrix **a_t) {
    Shape *shape = create_shape(a->shape->count, a->shape->columns, a->shape->rows);
    Matrix *matrix = create_matrix(shape);
    for(size_t i = shape->count; i--;)
        for(size_t j = shape->rows; j--;)
            for(size_t k = shape->columns; k--;)
                matrix->data[(i * shape->rows + j) * shape->columns + k] = a->data[(i * shape->columns + k) * shape->rows + j];
    *a_t = matrix;
}

void map(Matrix *matrix, float (*function)(float)) {
    for(size_t i = matrix->shape->size; i--;)
        matrix->data[i] = function(matrix->data[i]);
}

void copy(const Matrix *a, Matrix **created) {
    Matrix *matrix = create_matrix(a->shape);
    for(size_t i = matrix->shape->size; i--;)
        matrix->data[i] = a->data[i];
    *created = matrix;
}

void print_matrix(const Matrix *matrix) {
    Shape *shape = matrix->shape;
    print_shape(shape);
    printf("[");
    for(size_t i = 0; i < shape->count; i++) {
        printf("[");
        for(size_t j = 0; j < shape->rows; j++) {
            printf("[");
            for(size_t k = 0; k < shape->columns; k++) {
                printf("%.4f", matrix->data[(i * shape->rows + j) * shape->columns + k]);
                if(k != shape->columns - 1)
                    printf(", ");
            }
            printf("]");
            if(j != shape->rows - 1)
                printf(",\n");
        }
        printf("]");
        if(i != shape->count - 1)
            printf("\n\n");
    }
    printf("]\n\n");
}

void free_matrix(Matrix *matrix) {
    free(matrix->data);
    free(matrix->shape);
    free(matrix);
}

unsigned int matrixes_equals(Matrix *first, Matrix *second) {
    Shape *shape1 = first->shape;
    Shape *shape2 = second->shape;
    if(!shapes_equals(shape1, shape2))
        return FALSE;
    for(size_t i = shape1->size; i--;) {
        if(first->data[i] != second->data[i])
            return FALSE;
    }
    return TRUE;
}

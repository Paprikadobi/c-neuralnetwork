#include <stdio.h>
#include <stdlib.h>

#include "Matrix.h"

#define TRUE 1
#define FALSE 0

void create_shape(const unsigned int rows, const unsigned int columns, Shape **created) {
    Shape *shape = malloc(sizeof(Shape));
    shape->rows = rows;
    shape->columns = columns;
    shape->size = rows * columns;
    *created = shape;
}

void print_shape(Shape *shape) {
    printf("(%d, %d)\n", shape->rows, shape->columns);
}

void create_matrix(const unsigned int rows, const unsigned int columns, Matrix **created) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->data = malloc(sizeof(float) * rows * columns);
    create_shape(rows, columns, &(matrix->shape));
    *created = matrix;
}

void set(const float num, Matrix *matrix) {
    for(size_t i = matrix->shape->rows; i--;)
        for(size_t j = matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->columns + j] = num;
}

void randomize(const float min, const float max, Matrix *matrix) {
    for(size_t i = matrix->shape->rows; i--;)
        for(size_t j = matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->columns + j] = (((float) rand() / (float) RAND_MAX) - 0.5) * (max - min) + (max + min) * 0.5;
}

void add(Matrix *matrix, const float num) {
    for(size_t i = matrix->shape->rows; i--;)
        for(size_t j = matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->columns + j] += num;
}

unsigned int matrix_addition(const Matrix *a, Matrix *b) {
    if(a->shape->rows != b->shape->rows && a->shape->columns != b->shape->columns)
        return FALSE;
    for(size_t i = a->shape->rows; i--;) {
        for(size_t j = a->shape->columns; j--;)
            b->data[i * a->shape->columns + j] += a->data[i * a->shape->columns + j];
    }
    return TRUE;
}

void multiply(Matrix *matrix, const float num) {
    for(size_t i = matrix->shape->rows; i--;)
        for(size_t j = matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->columns + j] *= num;
}

unsigned int matrix_mult(Matrix *a, const Matrix *b) {
    if(a->shape->rows != b->shape->rows && a->shape->columns != b->shape->columns)
        return FALSE;
    for(size_t i = a->shape->rows; i--;)
        for(size_t j = a->shape->columns; j--;)
            a->data[i * a->shape->columns + j] *= b->data[i * a->shape->columns + j];
    return TRUE;
}

unsigned int matrix_multiplication(const Matrix *a, const Matrix *b, Matrix **c) {
    if(a->shape->columns != b->shape->rows)
        return FALSE;
    Matrix *matrix;
    create_matrix(a->shape->rows, b->shape->columns, &matrix);
    set(0, matrix);
    for(size_t ii = 0; ii < matrix->shape->rows; ii += BLK_SIZE)
        for(size_t jj = 0; jj < a->shape->columns; jj += BLK_SIZE)
            for(size_t kk = 0; kk < b->shape->columns; kk += BLK_SIZE)
                for(size_t i = ii; i < min(matrix->shape->rows, ii + BLK_SIZE); i++)
                    for(size_t j = jj; j < min(a->shape->columns, jj + BLK_SIZE); j++)
                        for(size_t k = kk; k < min(b->shape->columns, kk + BLK_SIZE); k++)
                            matrix->data[i * matrix->shape->columns + k] += a->data[i * a->shape->columns + j] * b->data[j * b->shape->columns + k];
    *c = matrix;
    return TRUE;
}

unsigned int reshape(Matrix *matrix, Shape *shape) {
    if(matrix->shape->rows * matrix->shape->columns / shape->rows != shape->columns)
        return FALSE;
    create_shape(shape->rows, shape->columns, &(matrix->shape));
    return TRUE;
}

void filter_matrix(const Matrix *matrix, const Matrix *filter, Matrix **created) {
    Matrix *result;
    create_matrix(matrix->shape->rows - filter->shape->rows, matrix->shape->columns - filter->shape->columns, &result);
    set(0, result);
    for(size_t i = 0; i < matrix->shape->rows - filter->shape->rows; i++)
        for(size_t j = 0; j < matrix->shape->columns - filter->shape->columns; j++)
            for(size_t k = 0; k < filter->shape->rows; k++)
                for(size_t l = 0; l < filter->shape->columns; l++)
                    result->data[i * result->shape->columns + j] += matrix->data[(i + k) * matrix->shape->columns + j + l] * filter->data[k * filter->shape->columns + l];
    *created = result;
}

void transpose(const Matrix *a, Matrix **a_t) {
    Matrix *matrix;
    create_matrix(a->shape->columns, a->shape->rows, &matrix);
    for(size_t i = matrix->shape->rows; i--;)
        for(size_t j = matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->columns + j] = a->data[j * a->shape->columns + i];
    *a_t = matrix;
}

void map(Matrix *matrix, float (*function)(float)) {
    for(size_t i = matrix->shape->rows; i--;)
        for(size_t j = matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->columns + j] = function(matrix->data[i * matrix->shape->columns + j]);
}

void copy(const Matrix *a, Matrix **created) {
    Matrix *matrix;
    create_matrix(a->shape->rows, a->shape->columns, &matrix);
    for(size_t i = matrix->shape->rows; i--;)
        for(size_t j = matrix->shape->columns; j--;)
            matrix->data[i * matrix->shape->columns + j] = a->data[i * matrix->shape->columns + j];
    *created = matrix;
}

void print_matrix(const Matrix *matrix) {
    printf("%d x %d\n", matrix->shape->rows, matrix->shape->columns);
    for(size_t i = 0; i < matrix->shape->rows; i++) {
        for(size_t j = 0; j < matrix->shape->columns; j++) {
            printf("| %7.4f ", matrix->data[i * matrix->shape->columns + j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void free_matrix(Matrix *matrix) {
    free(matrix->data);
    free(matrix->shape);
    free(matrix);
}

unsigned int matrixes_equals(Matrix *first, Matrix *second) {
    if(first->shape->rows != second->shape->rows || first->shape->columns != second->shape->columns)
        return FALSE;
    for(size_t i = first->shape->rows; i--;) {
        for(size_t j = first->shape->columns; j--;) {
            if(first->data[i * first->shape->columns + j] != second->data[i * second->shape->columns + j])
                return FALSE;
        }
    }
    return TRUE;
}

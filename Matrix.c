#include <stdio.h>
#include <stdlib.h>

#include "Matrix.h"

void create_matrix(const unsigned int rows, const unsigned int columns, Matrix **created) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->data = malloc(sizeof(float) * rows * columns);
    matrix->rows = rows;
    matrix->columns = columns;
    *created = matrix;
}

void set(const float num, Matrix *matrix) {
    for(size_t i = matrix->rows; i--;)
        for(size_t j = matrix->columns; j--;)
            matrix->data[i * matrix->columns + j] = num;
}

void randomize(const float min, const float max, Matrix *matrix) {
    for(size_t i = matrix->rows; i--;)
        for(size_t j = matrix->columns; j--;)
            matrix->data[i * matrix->columns + j] = (((float) rand() / (float) RAND_MAX) - 0.5) * (max - min) + (max + min) * 0.5;
}

void add(Matrix *matrix, const float num) {
    for(size_t i = matrix->rows; i--;)
        for(size_t j = matrix->columns; j--;)
            matrix->data[i * matrix->columns + j] += num;
}

unsigned int matrix_addition(const Matrix *a, Matrix *b) {
    if(a->rows != b->rows && a->columns != b->columns)
        return FALSE;
    for(size_t i = a->rows; i--;) {
        for(size_t j = a->columns; j--;)
            b->data[i * a->columns + j] += a->data[i * a->columns + j];
    }
    return TRUE;
}

void multiply(Matrix *matrix, const float num) {
    for(size_t i = matrix->rows; i--;)
        for(size_t j = matrix->columns; j--;)
            matrix->data[i * matrix->columns + j] *= num;
}

unsigned int matrix_multiplication(const Matrix *a, const Matrix *b, Matrix **c) {
    if(a->columns != b->rows)
        return FALSE;
    Matrix *matrix;
    create_matrix(a->rows, b->columns, &matrix);
    set(0, matrix);
    for(size_t ii = 0; ii < matrix->rows; ii += BLK_SIZE)
        for(size_t jj = 0; jj < a->columns; jj += BLK_SIZE)
            for(size_t kk = 0; kk < b->columns; kk += BLK_SIZE)
                for(size_t i = ii; i < min(matrix->rows, ii + BLK_SIZE); i++)
                    for(size_t j = jj; j < min(a->columns, jj + BLK_SIZE); j++)
                        for(size_t k = kk; k < min(b->columns, kk + BLK_SIZE); k++)
                            matrix->data[i * matrix->columns + k] += a->data[i * a->columns + j] * b->data[j * b->columns + k];
    *c = matrix;
    return TRUE;
}

void transpose(const Matrix *a, Matrix **a_t) {
    Matrix *matrix;
    create_matrix(a->columns, a->rows, &matrix);
    for(size_t i = matrix->rows; i--;)
        for(size_t j = matrix->columns; j--;)
            matrix->data[i * matrix->columns + j] = a->data[j * a->columns + i];
    *a_t = matrix;
}

void print_matrix(const Matrix *matrix) {
    for(size_t i = matrix->rows; i--;) {
        for(size_t j = matrix->columns; j--;) {
            printf("| %7.4f ", matrix->data[i * matrix->columns + j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void free_matrix(Matrix *matrix) {
    free(matrix->data);
    free(matrix);
}

unsigned int matrixes_equals(Matrix *first, Matrix *second) {
    if(first->rows != second->rows || first->columns != second->columns)
        return FALSE;
    for(size_t i = first->rows; i--;) {
        for(size_t j = first->columns; j--;) {
            if(first->data[i * first->columns + j] != second->data[i * second->columns + j])
                return FALSE;
        }
    }
    return TRUE;
}

#ifndef MATRIX_H
#define MATRIX_H

#define BLK_SIZE 128
#define min(a, b) (a < b ? a : b)

typedef struct {
    float *data;
    unsigned int rows;
    unsigned int columns;
} Matrix;

void createMatrix(const unsigned int rows, const unsigned int columns, Matrix **matrix);

void set(const float num, Matrix *matrix);

void randomize(const float min, const float max, Matrix *matrix);

void add(Matrix *matrix, const float num);

unsigned int matrixAddition(const Matrix *a, const Matrix *b, Matrix **c);

void multiply(Matrix *matrix, const float num);

unsigned int matrixMultiplication(const Matrix *a, const Matrix *b, Matrix **c);

void transpose(const Matrix *a, Matrix **a_T);

void print(const Matrix *matrix);

void freeMatrix(Matrix *matrix);

#endif

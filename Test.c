#include <stdio.h>

#include "Matrix.h"
#include "Test.h"

int main() {
    matrix_set_test();
    matrix_multiply_test();
    matrix_transpose_test();
    matrix_multiplication_test();
    
    return 0;
}

void matrix_set_test() {
    Shape *shape = create_shape(1, 3, 2);
    Matrix *a = create_matrix(shape);
    Matrix *b = create_matrix(shape);
    
    float data[6] = {1, 1, 1, 1, 1, 1};
    set(1, a);
    b->data = data;
    test(a, b, "Matrix set");
}

void matrix_multiply_test() {
    Shape *shape = create_shape(1, 3, 2);
    Matrix *a = create_matrix(shape);
    Matrix *b = create_matrix(shape);
    
    float data_a[6] = {1, 1, 1, 1, 1, 1};
    float data_b[6] = {-1, -1, -1, -1, -1, -1};
    
    a->data = data_a;
    b->data = data_b;
    multiply(a, -1);
    test(a, b, "Matrix constant multiplication");
}

void matrix_transpose_test() {
    Matrix *a, *b, *c;
    Shape *shape_a, *shape_b;
    shape_a = create_shape(1, 3, 2);
    shape_b = create_shape(1, 2, 3);
    a = create_matrix(shape_a);
    c = create_matrix(shape_b);
    
    float data_a[6] = {1, 2, 3, 4, 5, 6};
    float data_c[6] = {1, 3, 5, 2, 4, 6};
    
    a->data = data_a;
    c->data = data_c;
    transpose(a, &b);
    test(b, c, "Matrix transpose");
}

void matrix_multiplication_test() {
    Matrix *a, *b, *c, *d;
    Shape *shape_a, *shape_b, shape_d;
    shape_a = create_shape(1, 3, 2);
    shape_b = create_shape(1, 2, 3);
    shape_d = create_shape(1, 3, 3);
    a = create_matrix(shape_a);
    b = create_matrix(shape_b);
    d = create_matrix(shape_d);
    
    float data_a[6] = {1, 1, 1, 1, 1, 1};
    float data_b[6] = {-0.5, -0.5, -0.5, -0.5, -0.5, -0.5};
    float data_d[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    
    a->data = data_a;
    b->data = data_b;
    d->data = data_d;
    matrix_multiplication(a, b, &c);
    test(c, d, "Matrix multiplication");
}

void test(Matrix *real, Matrix *expected, char *test_name) {
    if(matrixes_equals(real, expected))
        printf("PASSED test: %s\n", test_name);
    else {
        printf("FAILED test: %s\n", test_name);
        printf("Expected: \n");
        print_matrix(expected);
        printf("Real: \n");
        print_matrix(real);
    }
}

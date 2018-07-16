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
    Matrix *a, *b;
    create_matrix(3, 2, &a);
    create_matrix(3, 2, &b);
    
    set(1, a);
    float data[6] = {1, 1, 1, 1, 1, 1};
    b->data = data;
    test(a, b, "Matrix set");
}

void matrix_multiply_test() {
    Matrix *a, *b;
    create_matrix(3, 2, &a);
    create_matrix(3, 2, &b);
    
    set(1, a);
    multiply(a, -1);
    float data[6] = {-1, -1, -1, -1, -1, -1};
    b->data = data;
    test(a, b, "Matrix constant multiplication");
}

void matrix_transpose_test() {
    Matrix *a, *b, *c;
    create_matrix(3, 2, &a);
    create_matrix(2, 3, &c);
    
    set(1, a);
    set(1, c);
    transpose(a, &b);
    test(b, c, "Matrix transpose");
}

void matrix_multiplication_test() {
    Matrix *a, *b, *c, *d;
    create_matrix(3, 2, &a);
    create_matrix(2, 3, &b);
    create_matrix(3, 3, &c);
    create_matrix(3, 3, &d);
    
    set(1, a);
    set(-0.5, b);
    set(-1, d);
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

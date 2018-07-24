#include <math.h>

#include "ActivationFunctions.h"

#define max(x, y) (x > y ? x : y)

float sigmoid(const float x) {
    return 1.0 / (1.0 + exp(-x));
}

float d_sigmoid(const float x) {
    return x * (1.0 - x);
}

float tanh_f(const float x) {
    return (1.0 - exp(-2 * x)) / (1.0 + exp(-2 * x));
}

float d_tanh(const float x) {
    return 1.0 - x * x;
}

float relu(const float x) {
    return max(0.0, x);
}

float d_relu(const float x) {
    return x ? 1.0 : 0.0;
}

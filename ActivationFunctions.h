#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

float sigmoid(const float x);

float d_sigmoid(const float x);

float tanh_f(const float x);

float d_tanh(const float x);

float relu(const float x);

float relu_d(const float x);

#endif

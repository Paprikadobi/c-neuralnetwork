#ifndef LAYER_H
#define LAYER_H

typedef enum {
    FULLY_CONNECTED_LAYER, FILTER_LAYER, POOLING_LAYER
} Layer_type;

extern const char *layer_description[3];

typedef struct {
    Matrix *weights;
    Matrix *biases;
    float (*activation)(float);
    float (*derivative)(float);
} Fully_connected_layer;

typedef struct {
    Matrix *weights;
    Matrix *bias;
} Filter_layer;

typedef struct {
    Shape *pooling_shape;
} Pooling_layer;

typedef struct {
    Shape *input_shape;
    Shape *output_shape;
    Matrix *input;
    Matrix *output;
    union {
        Fully_connected_layer *fully_connected;
        Filter_layer *filter;
        Pooling_layer *pool;
    };
    Layer_type type;
} Layer;

void create_layer(Shape *input_shape, Shape *output_shape, Layer **created);

void create_fully_connected_layer(float (*activation)(float), float (*derivative)(float), Layer *layer);

void create_filter_layer(Shape *filter_shape, Layer *layer);

void create_pooling_layer(Shape *shape, Layer *layer);

void feed_forward(Layer *layer, Matrix *input, Matrix **output);

void filter(Layer *layer, Matrix *input, Matrix **output);

void pool(Layer *layer, Matrix *input, Matrix **output);

void update(Layer *layer, Matrix **error);

void update_filter(Layer *layer, Matrix **error);

void update_pool(Layer *layer, Matrix **error);

void print_layer(const Layer *layer, const unsigned int show_values);

void free_layer(Layer *layer);

#endif

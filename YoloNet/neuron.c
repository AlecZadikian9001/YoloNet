//
//  neuron.c
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neuron.h"

void free_neuron(Neuron* neuron) {
    free(neuron->weights);
    free(neuron);
}

Neuron* mk_neuron(int num_weights, scalar (*func)(int, scalar*, scalar*)) {
    Neuron* neuron = emalloc(sizeof(Neuron));
    neuron->func = func;
    neuron->weights = emalloc(sizeof(scalar) * num_weights);
    neuron->num_weights = num_weights;
    return neuron;
}

void randomize_neuron(Neuron* n, scalar start, scalar end) {
    scalar r;
    int mod = ((scalar) (end - start)) * SCALAR_GRANULARITY;
    for (int i = 0; i < n->num_weights; i++) {
        r = ((scalar) (rand() % mod)) / SCALAR_GRANULARITY + start;
        n->weights[i] = r;
    }
}

scalar activate_neuron(Neuron* n, scalar* input) {
    return n->func(n->num_weights, input, n->weights);
}

void train_neuron(Neuron* n, scalar output) {
    // TODO how the fuck do I do this?
}

void print_neuron(Neuron* neuron) {
    printf("[");
    for (int i = 0; i < neuron->num_weights; i++) {
        if (i != neuron->num_weights - 1) {
            printf("%f, ", neuron->weights[i]);
        } else {
            printf("%f ]\n", neuron->weights[i]);
        }
    }
}



/* NEURON FUNCTIONS BELOW */

scalar neuron_func_tanh(int num, scalar* input, scalar* weights) {
    scalar sum = 0;
    for (int i = 0; i < num; i++) {
        sum += weights[i] * input[i];
    }
    return tanh(sum);
}

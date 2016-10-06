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
#include <string.h>

#include "neuron.h"

/* defaults */
#define DEFAULT_LEARNING_RATE (0.5)
#define DEFAULT_RAND_RATE (0.01)
#define DEFAULT_RAND_START (-10)
#define DEFAULT_RAND_END (10)

/* misc */
#define SCALAR_GRANULARITY ((scalar) 100000)
#define RANDOM_GRANULARITY (10000)

void free_neuron(Neuron* neuron) {
    free(neuron->weights);
    free(neuron->best_weights);
    free(neuron->biases);
    free(neuron->best_biases);
    free(neuron);
}

Neuron* mk_neuron(int dimension, scalar (*func)(scalar), scalar (*dfunc)(scalar)) {
    Neuron* neuron = emalloc(sizeof(Neuron));
    neuron->virgin = 1;
    
    neuron->learning_rate = DEFAULT_LEARNING_RATE;
    neuron->rand_rate = DEFAULT_RAND_RATE;
    neuron->w_rand_start = DEFAULT_RAND_START;
    neuron->w_rand_end = DEFAULT_RAND_END;
    neuron->b_rand_start = DEFAULT_RAND_START;
    neuron->b_rand_end = DEFAULT_RAND_END;
    
    neuron->dimension = dimension;
    neuron->weights = emalloc(sizeof(scalar) * dimension);
    neuron->best_weights = emalloc(sizeof(scalar) * dimension);
    neuron->biases = emalloc(sizeof(scalar) * dimension);
    neuron->best_biases = emalloc(sizeof(scalar) * dimension);
    
    neuron->func = func;
    neuron->dfunc = dfunc;
    
    return neuron;
}

void randomize_neuron(Neuron* n) {
    scalar r;
    int mod = 0;
    int start = 0;
    int end = 0;
    for (int i = 0; i < n->dimension * 2; i++) {
        
        /* choose mod */
        if (i == 0) { // weights
            start = n->w_rand_start;
            end = n->w_rand_end;
            mod = ((scalar) (end - start)) * SCALAR_GRANULARITY;
        }
        else if (i == n -> dimension) { // biases
            start = n->b_rand_start;
            end = n->b_rand_end;
            mod = ((scalar) (end - start)) * SCALAR_GRANULARITY;
        }
        
        if (mod > 0) {
            r = ((scalar) (rand() % mod)) / SCALAR_GRANULARITY + start;
            if (i < n->dimension) {
                n->weights[i] = r;
            } else { // if i >= n->dimension
                n->biases[i - n->dimension] = r;
            }
        }
    }
}

scalar activate_neuron(Neuron* n, scalar* input, int best) { // best = 0 if use current, 1 if use best
    scalar* biases;
    scalar* weights;
    
    if (best) {
        biases = n->best_biases;
        weights = n->best_weights;
    } else { // if !best
        biases = n->biases;
        weights = n->weights;
    }
    
    scalar sum = 0;
    for (int i = 0; i < n->dimension; i++) {
        sum += biases[i] + weights[i] * input[i];
    }
    return n->func(sum);
}

/* try input on neuron with given "correct" output, and train one iteration */
void train_neuron(Neuron* n, scalar* input, scalar output) {
    
    /* randomization (if triggered) */
    if (n->rand_rate >= ((double) (rand() % RANDOM_GRANULARITY)) / ((double) RANDOM_GRANULARITY)) {
        TRACE("Randomizing neuron (probability %f)\n", n->rand_rate);
        randomize_neuron(n);
    }
    
    /* backpropogation weight update */
    // http://www.philbrierley.com/main.html?code/bpproof.html&code/codeleft.html :
    // ∂E^2/dW_i = ∂E^2/∂I_i * ∂I_i/∂W_i
    // ∂I_i/∂W_i = O_i
    // ∂E^2/∂I_i = 2E * ∂F(I_i)/∂(I_i)
    scalar error = activate_neuron(n, input, 0) - output;
    for (int i = 0; i < n->dimension; i++) {
        // delta = 2 * error * n->dfunc(input[i]) * input[i]; // ∂E^2/dW_i
        // (new W_i) = (old W_i) - (learning rate) * ∂E^2/dW_i
        n->weights[i] = n->weights[i] - n->learning_rate * (2 * error * n->dfunc(input[i]) * input[i]);
    }
    
    scalar new_error = activate_neuron(n, input, 0) - output;
    scalar sq_error = new_error * new_error;
    if (n->virgin || sq_error < n->best_sq_error) {
        if (n->virgin) {
            TRACE("Neuron was virgin; set E^2 to %f\n", sq_error);
            n->virgin = 0;
        } else {
            TRACE("Improved E^2 from %f to %f\n", n->best_sq_error, sq_error);
        }
        memcpy(n->best_weights, n->weights, sizeof(scalar) * n->dimension);
        memcpy(n->best_biases, n->biases, sizeof(scalar) * n->dimension);
        n->best_sq_error = sq_error;
    }
}

void print_neuron(Neuron* neuron) { // ik, this code is cancer
    printf("weights: [");
    for (int i = 0; i < neuron->dimension; i++) {
        if (i != neuron->dimension - 1) {
            printf("%f, ", neuron->weights[i]);
        } else {
            printf("%f]\n", neuron->weights[i]);
        }
    }
    printf("biases: [");
    for (int i = 0; i < neuron->dimension; i++) {
        if (i != neuron->dimension - 1) {
            printf("%f, ", neuron->biases[i]);
        } else {
            printf("%f]\n", neuron->biases[i]);
        }
    }
}



/* NEURON FUNCTIONS BELOW */

scalar neuron_func_tanh(scalar input) { // pH > 7
    return (scalar) tanh(input);
}

scalar neuron_dfunc_tanh(scalar input) { // 1 - \tanh^2(x)
    scalar t = tanh(input);
    return 1 - (t * t);
}

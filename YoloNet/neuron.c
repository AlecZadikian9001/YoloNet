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

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a < _b ? _a : _b; })

/* defaults */
#define DEFAULT_LEARNING_RATE (0.1)
#define DEFAULT_BACKPROP_RATE (0.1)
#define DEFAULT_RAND_RATE (0.01)
#define DEFAULT_RAND_START (-10)
#define DEFAULT_RAND_END (10)

/* misc */
#define SCALAR_GRANULARITY ((scalar) 10000000000)
#define RANDOM_GRANULARITY (1000000000000000)

void free_neuron(Neuron* neuron) {
    free(neuron->weights);
    free(neuron->biases);
    free(neuron);
}

Neuron* mk_neuron(int dimension, scalar (*func)(scalar), scalar (*dfunc)(scalar)) {
    Neuron* neuron = emalloc(sizeof(Neuron));
    neuron->last_output = NAN;
    
    neuron->learning_rate = DEFAULT_LEARNING_RATE;
    neuron->learning_rate_ptr = NULL;
    neuron->backprop_rate = DEFAULT_BACKPROP_RATE;
    neuron->backprop_rate_ptr = NULL;
    neuron->rand_rate = DEFAULT_RAND_RATE;
    neuron->w_rand_start = DEFAULT_RAND_START;
    neuron->w_rand_end = DEFAULT_RAND_END;
    neuron->b_rand_start = DEFAULT_RAND_START;
    neuron->b_rand_end = DEFAULT_RAND_END;
    
    neuron->dimension = dimension;
    neuron->backprop = emalloc(sizeof(scalar) * dimension);
    neuron->weights = emalloc(sizeof(scalar) * dimension);
    neuron->biases = emalloc(sizeof(scalar) * dimension);
    
    neuron->func = func;
    neuron->dfunc = dfunc;
    
    return neuron;
}

Neuron* clone_neuron(Neuron* n) {
    Neuron* ret = emalloc(sizeof(Neuron));
    
    memcpy(ret, n, sizeof(Neuron));
    
    ret->backprop = emalloc(sizeof(scalar) * n->dimension);
    ret->weights = emalloc(sizeof(scalar) * n->dimension);
    ret->biases = emalloc(sizeof(scalar) * n->dimension);
    
    memcpy(ret->backprop, n->backprop, sizeof(scalar) * n->dimension);
    memcpy(ret->weights, n->weights, sizeof(scalar) * n->dimension);
    memcpy(ret->biases, n->biases, sizeof(scalar) * n->dimension);
    
    return ret;
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

scalar get_neuron_sum(Neuron* n, scalar* input) {
    scalar* biases;
    scalar* weights;
    
    biases = n->biases;
    weights = n->weights;
    
    scalar sum = 0;
    for (int i = 0; i < n->dimension; i++) {
        sum += biases[i] + weights[i] * input[i];
    }
    return sum;
}

scalar activate_neuron(Neuron* n, scalar* input) {
    scalar sum = get_neuron_sum(n, input);
    scalar output = n->func(sum);
    n->last_output = output;
    return output;
}

/* 
 try input on neuron with given "correct" output, and train for one sequence item
 set the "correct" values to propogate to inputs
 */
void train_neuron(Neuron* n, scalar* input, scalar output) {
    
    /* randomization (if triggered) */
    scalar rand_roll = ((double) (rand() % RANDOM_GRANULARITY)) / ((double) RANDOM_GRANULARITY);
    if (n->rand_rate > rand_roll) {
        //TRACE("Randomizing neuron (probability %f)\n", n->rand_rate);
        randomize_neuron(n);
    }
    
    /* weight update */
    // http://www.philbrierley.com/main.html?code/bpproof.html&code/codeleft.html :
    // ∂E^2/∂I = 2E * f'(I)
    // ∂E^2/dW_i = ∂E^2/∂I_i * ∂I/∂W_i
    // ∂I/∂W_i = O_i
    // ∂E^2/∂O_i = ∂E^2/∂I * ∂I/∂O_i
    // ∂I/∂O_i = W_i
    // ∂I/∂B_i = 1
    scalar test = activate_neuron(n, input);
    scalar error = test - output;
    for (int i = 0; i < n->dimension; i++) {
        scalar sum = get_neuron_sum(n, input);
        
        // get rates
        scalar learning_rate;
        if (!n->learning_rate_ptr) {
            learning_rate = n->learning_rate;
        } else {
            learning_rate = *(n->learning_rate_ptr);
        }
        scalar backprop_rate;
        if (!n->backprop_rate_ptr) {
            backprop_rate = n->backprop_rate;
        } else {
            backprop_rate = *(n->backprop_rate_ptr);
        }
        learning_rate = min(learning_rate, 1.0);
        backprop_rate = min(backprop_rate, 1.0);
        
        // TODO Something fishy is going on here.
        scalar new_weight = n->weights[i] - learning_rate * ( (2 * error * n->dfunc(sum) * input[i]) ); // ∂E^2/∂W_i
        scalar new_backprop = input[i] - backprop_rate * ( (2 * error * n->dfunc(sum)) * n->weights[i] ); // ∂E^2/∂O_i
        scalar new_bias = n->biases[i] - learning_rate * ( (2 * error * n->dfunc(sum)) * 1 ); // ∂E^2/∂B = ∂E^2/∂I * ∂I/∂B
        
        // TODO temp overrides
        new_backprop = input[i] - backprop_rate * ( (2 * error * n->dfunc(sum)) );
        //new_bias = n->biases[i] - learning_rate * ( (2 * error) * 1 );
        //new_bias = n->biases[i];
        
        // sanity checks
        if (new_bias > 9000 || new_bias < -9000) {
            perror("new_bias is too extreme\n");
            exit(2);
        }
        if (new_weight != new_weight || new_backprop != new_backprop || new_bias != new_bias) {
            perror("new weight or backprop or bias = NAN\n");
            exit(2);
        }
        
        // apply changes
        n->weights[i] = new_weight;
        n->backprop[i] = new_backprop;
        n->biases[i] = new_bias;
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

scalar neuron_func_id(scalar input) { // identity
    return input;
}

scalar neuron_dfunc_id(scalar input) { // return 1
    return 1;
}

scalar neuron_func_tanh(scalar input) { // pH > 7
    return (scalar) tanh(input);
}

scalar neuron_dfunc_tanh(scalar input) { // 1 - \tanh^2(x)
    scalar t = tanh(input);
    return 1 - (t * t);
}

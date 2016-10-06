//
//  neuron.h
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#ifndef neuron_h
#define neuron_h

#include "general.h"

/* scalar */

typedef double scalar; // the number we always use

/* end */

/* Neuron struct, constructors, destructors, utils */

typedef struct {
    
    // vect0rz
    int dimension; // number of weights
    scalar* weights; // weights to be tuned
    scalar* best_weights; // best weights found so far
    scalar* biases; // biases to be tuned (same number as number of weights)
    scalar* best_biases; // best biases found so far (corresponds to best_weights)
    
    // stuff to tune
    scalar learning_rate; // learning rate
    scalar rand_rate; // chance of randomizing
    scalar w_rand_start; // start of random weight
    scalar w_rand_end; // end of random weight
    scalar b_rand_start; // start of random bias
    scalar b_rand_end; // end of random bias
    
    // l3rning st4te
    scalar best_sq_error; // best E^2 found thus far
    int virgin; // 0 = trained, 1 = fresh
    
    // input: return output
    scalar (*func)(scalar);
    
    // input: return first derivative of output
    scalar (*dfunc)(scalar);
} Neuron;

// minimal constructor; you get to set tuning params after
Neuron* mk_neuron(int dimension, scalar (*func)(scalar), scalar (*dfunc)(scalar));

// destructor
void free_neuron(Neuron* neuron);

// prints out a neuron's state
void print_neuron(Neuron* neuron);

/* end */

/* neuron methods */

void randomize_neuron(Neuron* n);
scalar activate_neuron(Neuron* n, scalar* input, int best);
void train_neuron(Neuron* n, scalar* input, scalar output);
    
/* end */

/* activation functions */

// tanh, a basic activation function
scalar neuron_func_tanh(scalar input);
scalar neuron_dfunc_tanh(scalar input);

#endif /* neuron_h */

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

#define SCALAR_GRANULARITY ((scalar) 100000)

typedef double scalar; // the number we always use

/* end */

/* Neuron struct, constructors, destructors, utils */

typedef struct {
    scalar* weights; // parameters to be tuned
    int num_weights; // number of weights
    scalar (*func)(int, scalar*, scalar*); // count, input, weight
} Neuron;

Neuron* mk_neuron(int num_weights, scalar (*func)(int, scalar*, scalar*));

void free_neuron(Neuron* neuron);

void print_neuron(Neuron* neuron);

/* end */

/* neural net stuff */

void randomize_neuron(Neuron* n, scalar start, scalar end);
scalar activate_neuron(Neuron* n, scalar* input);

/* end */

#endif /* neuron_h */

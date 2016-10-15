//
//  net.h
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#ifndef net_h
#define net_h

#include <stdio.h>
#include "neuron.h"

typedef struct Neural_Node Neural_Node;

typedef struct {
    int num_inputs;
    Neural_Node** input_nodes;
    int num_outputs;
    Neural_Node** output_nodes;
    int num_levels; // input to output
    int* nodes_per_level; // input to output
    Neural_Node*** levels; // input to output
    scalar error; // worst-case squared error
    scalar best_error; // best error encountered thus far
    Neuron** best_params; // best neuron params used thus far
    scalar learning_rate;
    scalar backprop_rate;
} Neural_Net;

Neural_Net* mk_deep_net(int num_inputs, int num_outputs, int num_layers, int* layers);
scalar* activate_net(Neural_Net* net, scalar* input, int best);
void begin_net_sequence(Neural_Net* net);
void train_net(Neural_Net* net, int num_trains, scalar** inputs, scalar** outputs);
void finish_net_sequence(Neural_Net* net);
scalar net_best_error(Neural_Net* net);
scalar get_net_error(Neural_Net* net, int num_trains, scalar** inputs, scalar** outputs, int best);
void randomize_net(Neural_Net* net);
void set_net_best(Neural_Net* net);

#endif /* net_h */

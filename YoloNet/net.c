//
//  net.c
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#include <stdlib.h>

#include "net.h"
#include "neuron.h"

#define NET_W_START (-1.5)
#define NET_W_END (-NET_W_START)
#define NET_B_START -0.5
#define NET_B_END (-NET_B_START)
#define NET_RAND_RATE 0.01

struct Neural_Node {
    Neuron* neuron; // NULL if net input
    int index; // index within layer
    int num_inputs; // 0 if net input
    struct Neural_Node** inputs; // NULL if net input
    int num_outputs; // 0 if net output
    struct Neural_Node** outputs; // NULL if net output
};

Neural_Node* mk_neural_node(Neuron* n, int index, int num_inputs, Neural_Node** inputs, int num_outputs, Neural_Node** outputs) {
    Neural_Node* nn = emalloc(sizeof(Neural_Node));
    nn->index = index;
    nn->neuron = n;
    nn->num_inputs = num_inputs;
    nn->inputs = inputs;
    nn->num_outputs = num_outputs;
    nn->outputs = outputs;
    return nn;
}

void free_neural_node(Neural_Node* nn) {
    free_neuron(nn->neuron);
    if (nn->inputs) {
        free(nn->inputs);
    }
    if (nn->outputs) {
        free(nn->outputs);
    }
    free(nn);
}

// TODO accept different functions instead of always using tanh
/* return output nodes */
Neural_Net* mk_deep_net(int num_inputs, int num_outputs, int num_layers, int* layers) {
    
    int num_levels = num_layers + 2;
    int* nodes_per_level = emalloc(sizeof(int) * num_levels);
    Neural_Node*** levels = emalloc(sizeof(Neural_Node**) * num_levels);
    
    /* input layer */
    Neural_Node** inputs = emalloc(sizeof(Neural_Node*) * num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        Neuron* n = mk_neuron(1, &neuron_func_id, &neuron_dfunc_id);
        
        n->weights[0] = 1.0;
        n->biases[0] = 0.0;
        n->rand_rate = 0;
        
        Neural_Node* in_node = mk_neural_node(n, i, 0, NULL, layers[0], NULL);
        inputs[i] = in_node;
    }
    levels[0] = inputs;
    nodes_per_level[0] = num_inputs;
    
    /* hidden layers */
    Neural_Node** last_nodes = inputs;
    int last_num_nodes = num_inputs;
    for (int i = 0; i < num_layers; i++) {
        
        int num_nodes = layers[i];
        int next_num_nodes;
        if (i < num_layers - 1) {
            next_num_nodes = layers[i+1];
        } else {
            next_num_nodes = num_outputs;
        }
        Neural_Node** nodes = emalloc(sizeof(Neural_Node*) * num_nodes);
        for (int j = 0; j < num_nodes; j++) {
            Neuron* n = mk_neuron(last_num_nodes, &neuron_func_tanh, &neuron_dfunc_tanh);
            
            n->b_rand_start = NET_B_START;
            n->b_rand_end = NET_B_END;
            n->w_rand_start = NET_W_START;
            n->w_rand_end = NET_W_END;
            n->rand_rate = NET_RAND_RATE;
            
            randomize_neuron(n);
            Neural_Node* nn = mk_neural_node(n, j, last_num_nodes, last_nodes, next_num_nodes, NULL);
            nodes[j] = nn;
        }
    
        for (int j = 0; j < last_num_nodes; j++) {
            last_nodes[j]->outputs = nodes;
        }
        
        levels[i + 1] = nodes;
        nodes_per_level[i + 1] = num_nodes;
        
        last_num_nodes = num_nodes;
        last_nodes = nodes;
    }
    
    /* output layer */
    Neural_Node** outputs = emalloc(sizeof(Neural_Node*) * num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        Neuron* n = mk_neuron(1, &neuron_func_id, &neuron_dfunc_id);
        
        n->weights[0] = 1.0;
        n->biases[0] = 0.0;
        n->rand_rate = 0;
        
        Neural_Node* nn = mk_neural_node(n, i, last_num_nodes, last_nodes, 0, NULL);
        outputs[i] = nn;
    }
    for (int j = 0; j < last_num_nodes; j++) {
        last_nodes[j]->outputs = outputs;
    }
    levels[num_layers + 1] = outputs;
    nodes_per_level[num_layers + 1] = num_outputs;
    
    Neural_Net* ret = emalloc(sizeof(Neural_Net));
    ret->num_outputs = num_outputs;
    ret->output_nodes = outputs;
    ret->num_inputs = num_inputs;
    ret->input_nodes = inputs;
    ret->levels = levels;
    ret->nodes_per_level = nodes_per_level;
    ret->num_levels = num_levels;
    
    return ret;
}

scalar* activate_net(Neural_Net* net, scalar* input, int best) {
    scalar* in_outs = NULL;
    for (int level_i = 0; level_i < net->num_levels; level_i++) {
        int nodes_per_level = net->nodes_per_level[level_i];
        scalar* new_in_outs = emalloc(sizeof(scalar) * nodes_per_level);
        
        int input_level = (in_outs == NULL);
        if (input_level) {
            in_outs = emalloc(sizeof(scalar) * 1);
        }
        for (int i = 0; i < nodes_per_level; i++) {
            if (input_level) {
                in_outs[0] = input[i];
            }
            Neuron* n = net->levels[level_i][i]->neuron;
            scalar out = activate_neuron(n, in_outs, best);
            new_in_outs[i] = out;
        }
        
        
        if (in_outs) {
            free(in_outs);
        }
        in_outs = new_in_outs;
    }
    return in_outs;
}

// TODO currently assumes every node connects to every other node in adjacent levels
void train_net_helper(Neural_Net* net, scalar* input, scalar* outputs) {
    for (int level_i = net->num_levels - 1; level_i >= 0; level_i--) {
        int nodes_per_level = net->nodes_per_level[level_i];
        
        scalar* ins;
        if (level_i > 0) {
            int nppl = net->nodes_per_level[level_i - 1];
            ins = emalloc(sizeof(scalar) * nppl);
            for (int i = 0; i < nppl; i++) {
                Neural_Node* last_nn = net->levels[level_i - 1][i];
                ins[i] = last_nn->neuron->last_output;
            }
        } else { // input
            ins = input;
        }
        
        for (int i = 0; i < nodes_per_level; i++) {
            Neural_Node* nn = net->levels[level_i][i];
            
            scalar* parent_output;
            int num_parents;
            if (level_i != net->num_levels - 1) { // if not output
                num_parents = nn->num_outputs;
                parent_output = emalloc(sizeof(scalar) * num_parents);
                for (int parent = 0; parent < nn->num_outputs; parent++) {
                    parent_output[parent] = nn->outputs[parent]->neuron->backprop[nn->index];
                }
            } else { // if output
                num_parents = 1;
                parent_output = emalloc(sizeof(scalar) * num_parents);
                parent_output[0] = outputs[i];
            }
            
            for (int parent = 0; parent < num_parents; parent++) {
                train_neuron(nn->neuron, ins, parent_output[parent]);
            }
            free(parent_output);
        }
        
        if (ins != input) {
            free(ins);
        }
    }
}

void net_func(Neural_Net* net, void (*func)(Neuron*)) {
    for (int level_i = 0; level_i < net->num_levels; level_i++) {
        int nodes_per_level = net->nodes_per_level[level_i];
        for (int i = 0; i < nodes_per_level; i++) {
            func(net->levels[level_i][i]->neuron);
        }
    }
}

void begin_net_sequence(Neural_Net* net) {
    net_func(net, &begin_neuron_sequence);
}

void train_net(Neural_Net* net, int num_trains, scalar** inputs, scalar** outputs) {
    for (int train_i = 0; train_i < num_trains; train_i++) {
        train_net_helper(net, inputs[train_i], outputs[train_i]);
    }
}

void finish_net_sequence(Neural_Net* net) {
    net_func(net, &finish_neuron_sequence);
}


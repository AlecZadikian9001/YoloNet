//
//  net.c
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#include <stdlib.h>

#include "net.h"

struct Neural_Node {
    Neuron* neuron; // NULL if net input
    int num_inputs; // 0 if net input
    struct Neural_Node** inputs; // NULL if net input
    int num_outputs; // 0 if net output
    struct Neural_Node** outputs; // NULL if net output
};

Neural_Node* mk_neural_node(Neuron* n, int num_inputs, Neural_Node** inputs, int num_outputs, Neural_Node** outputs) {
    Neural_Node* nn = emalloc(sizeof(Neural_Node*) * num_inputs);
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
Neural_Node** create_deep_net(int num_inputs, int num_outputs, int num_layers, int* layers) {
    
    /* input layer */
    Neural_Node** inputs = emalloc(sizeof(Neural_Node*) * num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        Neuron* in = mk_neuron(1, &neuron_func_id, &neuron_dfunc_id);
        Neural_Node* in_node = mk_neural_node(in, 0, NULL, layers[0], NULL);
        inputs[i] = in_node;
    }
    
    /* hidden layers */
    Neural_Node** last_nodes = inputs;
    int last_num_nodes = num_inputs;
    for (int i = 0; i < num_layers; i++) {
        
        int num_nodes = layers[i];
        int next_num_nodes;
        if (i < num_layers - 1) {
            next_num_nodes = layers[i+1];
        } else {
            next_num_nodes = 0;
        }
        Neural_Node** nodes = emalloc(sizeof(Neural_Node*) * num_nodes);
        for (int j = 0; j < num_nodes; j++) {
            Neuron* n = mk_neuron(last_num_nodes, &neuron_func_tanh, &neuron_dfunc_tanh);
            Neural_Node* nn = mk_neural_node(n, last_num_nodes, last_nodes, next_num_nodes, NULL);
            nodes[j] = nn;
        }
    
        for (int j = 0; j < last_num_nodes; j++) {
            last_nodes[j]->outputs = nodes;
        }
        
        last_num_nodes = num_nodes;
        last_nodes = nodes;
    }
    
    /* output layer */
    Neural_Node** outputs = emalloc(sizeof(Neural_Node*) * num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        Neuron* n = mk_neuron(1, &neuron_func_id, &neuron_dfunc_id);
        Neural_Node* nn = mk_neural_node(n, last_num_nodes, last_nodes, 0, NULL);
        outputs[i] = nn;
    }
    for (int j = 0; j < last_num_nodes; j++) {
        last_nodes[j]->outputs = outputs;
    }
    
    return NULL;
}


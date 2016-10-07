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
    
    /* input layer */
    Neural_Node** inputs = emalloc(sizeof(Neural_Node*) * num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        Neuron* in = mk_neuron(1, &neuron_func_id, &neuron_dfunc_id);
        randomize_neuron(in);
        Neural_Node* in_node = mk_neural_node(in, i, 0, NULL, layers[0], NULL);
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
            randomize_neuron(n);
            Neural_Node* nn = mk_neural_node(n, j, last_num_nodes, last_nodes, next_num_nodes, NULL);
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
        randomize_neuron(n);
        Neural_Node* nn = mk_neural_node(n, i, last_num_nodes, last_nodes, 0, NULL);
        outputs[i] = nn;
    }
    for (int j = 0; j < last_num_nodes; j++) {
        last_nodes[j]->outputs = outputs;
    }
    
    Neural_Net* ret = emalloc(sizeof(Neural_Net));
    ret->num_outputs = num_outputs;
    ret->output_nodes = outputs;
    
    return ret;
}

scalar activate_net_helper(Neural_Node* nn, scalar* input, int best) {
    scalar* new_input;
    if (!nn->inputs) { // input node
        new_input = input;
    } else { // internal or output node
        scalar prev_input[nn->num_inputs];
        for (int i = 0; i < nn->num_inputs; i++) {
            prev_input[i] = activate_net_helper(nn->inputs[i], input, best);
        }
        new_input = prev_input;
    }
    return activate_neuron(nn->neuron, new_input, best);
}

scalar* activate_net(Neural_Net* net, scalar* input, int best) { // TODO memoize with hashmap
    scalar* outputs = emalloc(sizeof(scalar) * net->num_outputs);
    for (int i = 0; i < net->num_outputs; i++) {
        Neural_Node* nn = net->output_nodes[i];
        scalar nn_output = activate_net_helper(nn, input, best);
        outputs[i] = nn_output;
    }
    return outputs;
}

void train_net_helper(Neural_Node* nn, scalar* input, scalar output) {
    scalar neuron_input[nn->num_inputs];
    for (int i = 0; i < nn->num_inputs; i++) {
        neuron_input[i] = activate_net_helper(nn->inputs[i], input, 0);
    }
    
    if (nn->outputs) { // non-output node
        for (int i = 0; i < nn->num_outputs; i++) {
            train_neuron(nn->neuron, neuron_input, nn->outputs[i]->neuron->backprop[nn->index]);
        }
    } else { // output node
        train_neuron(nn->neuron, neuron_input, output);
    }
    
    for (int i = 0; i < nn->num_inputs; i++) {
        train_net_helper(nn->inputs[i], input, output);
    }
}

void net_func_helper(Neural_Node* nn, void (*func)(Neuron*)) {
    func(nn->neuron);
    for (int i = 0; i < nn->num_inputs; i++) {
        net_func_helper(nn->inputs[i], func);
    }
}

void begin_net_sequence(Neural_Net* net) {
    for (int i = 0; i < net->num_outputs; i++) {
        net_func_helper(net->output_nodes[i], &begin_neuron_sequence);
    }
}

void train_net(Neural_Net* net, int num_trains, scalar** inputs, scalar** outputs) {
    for (int train_i = 0; train_i < num_trains; train_i++) {
        for (int output_i = 0; output_i < net->num_outputs; output_i++) {
            train_net_helper(net->output_nodes[output_i], inputs[train_i], outputs[train_i][output_i]);
        }
    }
}

void finish_net_sequence(Neural_Net* net) {
    for (int i = 0; i < net->num_outputs; i++) {
        net_func_helper(net->output_nodes[i], &finish_neuron_sequence);
    }
}


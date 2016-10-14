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

#define NET_W_START (-2.0)
#define NET_W_END (-NET_W_START)
#define NET_B_START -1.0
#define NET_B_END (-NET_B_START)
#define NET_RAND_RATE  (0.0000001)
#define NET_LEARN_RATE (0.0025)
#define NET_BACKPROP_RATE (0.5)

#define TRAIN(f_, ...) if(train_log) { printf("[TRAINING] "); printf((f_), __VA_ARGS__); }
#define TRAIN_(f_, ...) if(train_log) { printf((f_), __VA_ARGS__); }

int train_log = 0;

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

Neuron** save_net_neurons(Neural_Net* net) {
    int neuron_count = 0;
    for (int level_i = 0; level_i < net->num_levels; level_i++) {
        neuron_count += net->nodes_per_level[level_i];
    }
    
    Neuron** ret = emalloc(sizeof(Neuron**) * neuron_count);
    
    int i = 0;
    for (int level_i = 0; level_i < net->num_levels; level_i++) {
        for (int node_i = 0; node_i < net->nodes_per_level[level_i]; node_i++) {
            Neuron* n = clone_neuron(net->levels[level_i][node_i]->neuron);
            ret[i] = n;
            i += 1;
        }
    }
    
    return ret;
}

void free_net_neurons(Neural_Net* net, Neuron** neurons) {
    if (neurons == NULL) {
        return;
    }
    int i = 0;
    for (int level_i = 0; level_i < net->num_levels; level_i++) {
        for (int node_i = 0; node_i < net->nodes_per_level[level_i]; node_i++) {
            free_neuron(neurons[i]);
            i += 1;
        }
    }
    free(neurons);
}

void swap_net_neurons(Neural_Net* net, Neuron** neurons) {
    int i = 0;
    for (int level_i = 0; level_i < net->num_levels; level_i++) {
        for (int node_i = 0; node_i < net->nodes_per_level[level_i]; node_i++) {
            Neuron* tmp = neurons[i];
            neurons[i] = (net->levels[level_i][node_i]->neuron);
            net->levels[level_i][node_i]->neuron = tmp;
            i += 1;
        }
    }
}

// TODO accept different functions instead of always using tanh
/* return output nodes */
Neural_Net* mk_deep_net(int num_inputs, int num_outputs, int num_layers, int* layers) {
    
    int num_levels = num_layers + 2;
    int* nodes_per_level = emalloc(sizeof(int) * num_levels);
    Neural_Node*** levels = emalloc(sizeof(Neural_Node**) * num_levels);
    
    Neural_Net* ret = emalloc(sizeof(Neural_Net));
    ret->nodes_per_level = nodes_per_level;
    ret->num_levels = num_levels;
    ret->error = -1;
    ret->best_error = -1;
    ret->best_params = NULL;
    ret->backprop_rate = NET_BACKPROP_RATE;
    ret->learning_rate = NET_LEARN_RATE;
    
    /* input layer */
    Neural_Node** inputs = emalloc(sizeof(Neural_Node*) * 1);
    
    Neuron* n = mk_neuron(num_inputs, &neuron_func_id, &neuron_dfunc_id);
    
    for (int i = 0; i < n->dimension; i++) {
        n->weights[i] = 1.0;
        n->biases[i] = 0.0;
    }
    n->rand_rate = 0;
    n->learning_rate = 0;
    n->backprop_rate = NET_BACKPROP_RATE;
    n->backprop_rate_ptr = &(ret->backprop_rate);
    
    Neural_Node* in_node = mk_neural_node(n, 0, 0, NULL, layers[0], NULL);
    inputs[0] = in_node;
    
    levels[0] = inputs;
    nodes_per_level[0] = 1;
    
    /* hidden layers */
    Neural_Node** last_nodes = inputs;
    int last_num_nodes = 1;
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
            n->learning_rate = NET_LEARN_RATE;
            n->backprop_rate = NET_BACKPROP_RATE;
            n->learning_rate_ptr = &(ret->learning_rate);
            n->backprop_rate_ptr = &(ret->backprop_rate);
            
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
        Neuron* n = mk_neuron(last_num_nodes, &neuron_func_id, &neuron_dfunc_id);
        
        for (int j = 0; j < last_num_nodes; j++) {
            n->weights[j] = 1.0;
            n->biases[j] = 0.0;
        }
//        n->rand_rate = 0;
//        n->learning_rate = 0;
        n->rand_rate = NET_RAND_RATE;
        n->learning_rate = NET_LEARN_RATE;
        n->backprop_rate = NET_BACKPROP_RATE;
        n->learning_rate_ptr = &(ret->learning_rate);
        n->backprop_rate_ptr = &(ret->backprop_rate);
        
        Neural_Node* nn = mk_neural_node(n, i, last_num_nodes, last_nodes, 0, NULL);
        outputs[i] = nn;
    }
    for (int j = 0; j < last_num_nodes; j++) {
        last_nodes[j]->outputs = outputs;
    }
    levels[num_layers + 1] = outputs;
    nodes_per_level[num_layers + 1] = num_outputs;
    
    ret->num_outputs = num_outputs;
    ret->output_nodes = outputs;
    ret->num_inputs = num_inputs;
    ret->input_nodes = inputs;
    ret->levels = levels;
    
    return ret;
}

scalar* activate_net(Neural_Net* net, scalar* input, int best) {
    if (best) {
        swap_net_neurons(net, net->best_params);
    }
    
    scalar* in_outs = NULL;
    for (int level_i = 0; level_i < net->num_levels; level_i++) {
        int nodes_per_level = net->nodes_per_level[level_i];
        scalar* new_in_outs = emalloc(sizeof(scalar) * nodes_per_level);
        
        int input_level = (in_outs == NULL);
        if (input_level) {
            in_outs = input;
        }
        for (int i = 0; i < nodes_per_level; i++) {
            Neuron* n = net->levels[level_i][i]->neuron;
            scalar out = activate_neuron(n, in_outs);
            new_in_outs[i] = out;
        }
        
        
        if (in_outs && !input_level) {
            free(in_outs);
        }
        in_outs = new_in_outs;
    }
    
    if (best) {
        swap_net_neurons(net, net->best_params);
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
                scalar last_output = last_nn->neuron->last_output;
                ins[i] = last_output;
            }
        } else { // input
            ins = input;
        }
        
        TRAIN("\n", 0);
        TRAIN("Examining level %d, %d nodes, inputs:", level_i, nodes_per_level);
        int nppl;
        if (level_i > 0) {
            nppl = net->nodes_per_level[level_i - 1];
        } else {
            nppl = net->num_inputs;
        }
        for (int i = 0; i < nppl; i++) {
            TRAIN_(" %f", ins[i]);
        }
        TRAIN_("\n", 0);
        
    
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
            
            TRAIN("Node %d (index %d) pre-adjustment (weight, bias, backprop)s:", i, nn->index);
            for (int j = 0; j < nn->neuron->dimension; j++) {
                TRAIN_(" (%f, %f, %f)", nn->neuron->weights[j], nn->neuron->biases[j], nn->neuron->backprop[j]);
            }
            TRAIN_("\n", 0);
            
            activate_neuron(nn->neuron, ins);
            
            TRAIN("Node %d runtime vs expected outputs: ", i);
            for (int parent = 0; parent < num_parents; parent++) {
                
                TRAIN_(" (%f vs %f)", activate_neuron(nn->neuron, ins), parent_output[parent]);
                
                train_neuron(nn->neuron, ins, parent_output[parent]);
            }
            free(parent_output);
         
            TRAIN_("\n", 0);
            
            TRAIN("Node %d pst-adjustment (weight, bias, backprop)s:", i);
            for (int j = 0; j < nn->neuron->dimension; j++) {
                TRAIN_(" (%f, %f, %f)", nn->neuron->weights[j], nn->neuron->biases[j], nn->neuron->backprop[j]);
            }
            TRAIN_("\n", 0);
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
    net->error = -1;
    // TODO
}

scalar get_net_error(Neural_Net* net, int num_trains,
                     scalar** inputs, scalar** outputs, int best) {
    scalar worst_error = -1;
    scalar* outs;
    for (int train_i = 0; train_i < num_trains; train_i++) {
        outs = activate_net(net, inputs[train_i], best);
        for (int oi = 0; oi < net->num_outputs; oi++) {
            scalar error = outputs[train_i][oi] - outs[oi];
            scalar error_sq = error * error;
            if (error_sq > worst_error) {
                worst_error = error_sq;
            }
        }
        free(outs);
    }
    return worst_error;
}

void train_net(Neural_Net* net, int num_trains, scalar** inputs, scalar** outputs) {
    scalar worst_error = -1;
    for (int train_i = 0; train_i < num_trains; train_i++) {
        worst_error = get_net_error(net, num_trains, inputs, outputs, 0);
        scalar* out = activate_net(net, inputs[train_i], 0);
        free(out);
        train_net_helper(net, inputs[train_i], outputs[train_i]);
    }
    if (worst_error > net->error) {
        net->error = worst_error;
    }
}

void finish_net_sequence(Neural_Net* net) {
    if (net->best_error < 0 || net->error < net->best_error) {
        net->best_error = net->error;
        free_net_neurons(net, net->best_params);
        net->best_params = save_net_neurons(net);
    }
}


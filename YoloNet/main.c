//
//  main.c
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

/*
 ██╗   ██╗ ██████╗ ██╗      ██████╗         ███╗   ██╗███████╗████████╗
 ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗        ████╗  ██║██╔════╝╚══██╔══╝
  ╚████╔╝ ██║   ██║██║     ██║   ██║        ██╔██╗ ██║█████╗     ██║
   ╚██╔╝  ██║   ██║██║     ██║   ██║        ██║╚██╗██║██╔══╝     ██║
    ██║   ╚██████╔╝███████╗╚██████╔╝███████╗██║ ╚████║███████╗   ██║
    ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝
    Neural Network
    ╔═╗  ┬ ┬┌─┐┌─┐┬┌─  ┌┐ ┬ ┬  ╔═╗┬  ┌─┐┌─┐╔═╗
    ╠═╣  ├─┤├─┤│  ├┴┐  ├┴┐└┬┘  ╠═╣│  ├┤ │  ╔═╝
    ╩ ╩  ┴ ┴┴ ┴└─┘┴ ┴  └─┘ ┴   ╩ ╩┴─┘└─┘└─┘╚═╝
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "neuron.h"
#include "net.h"

#define LEARNING_RATE (0.00001)

scalar f(scalar i1, scalar i2) {
    return i1 + 6 * i2;
}

int main(int argc, const char * argv[]) {
    
    srand((int) time(NULL));
    
    int num_layers = 1;
    int layers[num_layers];
    for (int i = 0; i < num_layers; i++) {
        layers[i] = 1;
    }
    int num_inputs = 2;
    int num_outputs = 1;
    Neural_Net* net = mk_deep_net(num_inputs, num_outputs, num_layers, layers);
    
    int num_trains = 100;
    scalar* ins[num_trains];
    scalar* outs[num_trains];
    
    for (int i = 0; i < num_trains; i++) {
        scalar* in = emalloc(sizeof(scalar) * num_inputs);
        scalar* out = emalloc(sizeof(scalar) * num_outputs);
        
        for (int in_i = 0; in_i < num_inputs; in_i++) {
            in[in_i] = ((scalar) (rand() % 10000)) / 10000;
        }
        
        out[0] = f(in[0], in[1]);
        
        ins[i] = in;
        outs[i] = out;
    }
    
    int num_holds = 300;
    scalar* hins[num_holds];
    scalar* houts[num_holds];
    
    for (int i = 0; i < num_holds; i++) {
        scalar* in = emalloc(sizeof(scalar) * num_inputs);
        scalar* out = emalloc(sizeof(scalar) * num_outputs);
        
        for (int in_i = 0; in_i < num_inputs; in_i++) {
            in[in_i] = ((scalar) (rand() % 10000)) / 10000;
        }
        
        out[0] = f(in[0], in[1]);
        
        hins[i] = in;
        houts[i] = out;
    }
    
    scalar* outputs;
    for (int i = 0; i < num_trains; i++) {
        outputs = activate_net(net, ins[i], 0);
        printf("(%f, %f): %f\n", ins[i][0], ins[i][1], outputs[0]);
        free(outputs);
    }
    
    scalar last_error = INFINITY;
    scalar holdout_best = INFINITY;
    scalar holdout_error = INFINITY;
    scalar last_best_error = INFINITY;
    int repeats = 0;
    int reset = 0;
    for (int i = 0; 1; i++) {
        begin_net_sequence(net);
        train_net(net, num_trains, ins, outs);
        finish_net_sequence(net);
        
        net->learning_rate = LEARNING_RATE * pow(fabs(net->error), 1);
        
        scalar error_threshold = 0.1;
        
        holdout_error = get_net_error(net, num_holds, hins, houts, 0);
        if (holdout_error < holdout_best) {
            holdout_best = holdout_error;
        }
        
        if (net->error < error_threshold) {
            if (holdout_error < error_threshold) {
                printf("\n");
                printf("%d iterations\n", i);
                printf("current: %f, best: %f\n", net->error, net->best_error);
                break;
            }
        }
//        if (last_error != net->best_error) {
//            printf("\nnew best!\n");
//        }
        if (/*last_error != net->best_error ||*/ i % 100 == 0 || reset) {
            reset = 0;
            if (net->best_error == last_best_error) {
                repeats += 1;
                if (repeats > 10) {
                    printf("\nresetting to best!\n");
                    set_net_best(net);
                    repeats = 0;
                    reset = 1;
                }
            }
            printf("\rlearn: %.10e, current: %f, best: %f, holdout: %f, holdout best: %f",
                   net->learning_rate, net->error, net->best_error, holdout_error, holdout_best);
            fflush(stdout);
            last_best_error = net->best_error;
        }
        last_error = net->best_error;
    }
    
    printf("\ntraining data has error %f:\n", net->error);
    for (int i = 0; i < num_trains; i++) {
        outputs = activate_net(net, ins[i], 0);
        printf("(%f, %f): %f vs %f\n", ins[i][0], ins[i][1], outputs[0], outs[i][0]);
        free(outputs);
    }
    
    holdout_error = get_net_error(net, num_holds, hins, houts, 0);
    
    printf("\nholdout data has error %f:\n", holdout_error);
    for (int i = 0; i < num_holds; i++) {
        outputs = activate_net(net, hins[i], 0);
        printf("(%f, %f): %f vs %f\n", hins[i][0], hins[i][1], outputs[0], houts[i][0]);
        free(outputs);
    }
    
    return 0;
}

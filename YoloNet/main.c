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

scalar f(scalar i1, scalar i2, scalar i3) {
    return sin(i1) + i2 * i3;
}

int main(int argc, const char * argv[]) {
    
    srand((int) time(NULL));
    
    int num_layers = 3;
    int layers[] = {3, 10, 3};
    int num_inputs = 3;
    int num_outputs = 1;
    Neural_Net* net = mk_deep_net(num_inputs, num_outputs, num_layers, layers);
    
    int num_trains = 30;
    scalar* ins[num_trains];
    scalar* outs[num_trains];
    
    for (int i = 0; i < num_trains; i++) {
        scalar* in = emalloc(sizeof(scalar) * num_inputs);
        scalar* out = emalloc(sizeof(scalar) * num_outputs);
        
        for (int in_i = 0; in_i < num_inputs; in_i++) {
            in[in_i] = ((scalar) (rand() % 10000)) / 10000;
        }
        
        out[0] = f(in[0], in[1], in[2]);
        
        ins[i] = in;
        outs[i] = out;
    }
    
    int num_holds = 5;
    scalar* hins[num_trains];
    scalar* houts[num_trains];
    
    for (int i = 0; i < num_holds; i++) {
        scalar* in = emalloc(sizeof(scalar) * num_inputs);
        scalar* out = emalloc(sizeof(scalar) * num_outputs);
        
        for (int in_i = 0; in_i < num_inputs; in_i++) {
            in[in_i] = ((scalar) (rand() % 10000)) / 10000;
        }
        
        out[0] = f(in[0], in[1], in[2]);
        
        hins[i] = in;
        houts[i] = out;
    }
    
    scalar* outputs;
    for (int i = 0; i < num_trains; i++) {
        outputs = activate_net(net, ins[i], 0);
        printf("(%f, %f): %f\n", ins[i][0], ins[i][1], outputs[0]);
        free(outputs);
    }
    
    scalar last_error = -1;
    for (int i = 0; 1; i++) {
        begin_net_sequence(net);
        train_net(net, num_trains, ins, outs);
        finish_net_sequence(net);
        
//        calc_error = get_net_error(net, 4, ins, outs, 0);
//        if (net->error != calc_error) {
//            printf("WTF. error = %f, calc_error = %f\n", net->error, calc_error);
//            exit(9001);
//        }
        
        if (net->error < 0.05) {
            printf("\n");
            printf("%d iterations\n", i);
            printf("current: %f, best: %f\n", net->error, net->best_error);
            break;
        }
        if (last_error != net->best_error) {
            printf("\nnew best!\n");
        }
        if (last_error != net->best_error || i % 1000 == 0) {
            printf("\rcurrent: %f, best: %f", net->error, net->best_error);
            fflush(stdout);
        }
        last_error = net->best_error;
    }
    
    printf("\ntraining data has error %f:\n", net->error);
    for (int i = 0; i < num_trains; i++) {
        outputs = activate_net(net, ins[i], 0);
        printf("(%f, %f): %f vs %f\n", ins[i][0], ins[i][1], outputs[0], outs[i][0]);
        free(outputs);
    }
    
    scalar holdout_error = get_net_error(net, num_holds, hins, houts, 0);
    
    printf("\nholdout data has error %f:\n", holdout_error);
    for (int i = 0; i < num_holds; i++) {
        outputs = activate_net(net, hins[i], 0);
        printf("(%f, %f): %f vs %f\n", hins[i][0], hins[i][1], outputs[0], houts[i][0]);
        free(outputs);
    }
    
    return 0;
}

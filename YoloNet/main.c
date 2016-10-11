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

int main(int argc, const char * argv[]) {
    
    srand((int) time(NULL));
    
//    Neuron* n = mk_neuron(2, &neuron_func_tanh, &neuron_dfunc_tanh);
//    //n->b_rand_start = 0;
//    //n->b_rand_end = 0;
//    randomize_neuron(n);
//    
//    scalar in1[] = {2.0, 3.0};
//    scalar in2[] = {9.0, 9.0};
//    scalar** in_sequence = emalloc(2 * sizeof(scalar*));
//    in_sequence[0] = in1;
//    in_sequence[1] = in2;
//    
//    scalar out_sequence[] = {-0.3, 0.6};
    

//    scalar result;
//    
//    for (int i = 0; i < 100; i++) {
//        begin_neuron_sequence(n);
//        for (int j = 0; j < 100; j++) {
//            train_neuron(n, in_sequence[j % 2], out_sequence[j % 2]);
//        }
//        finish_neuron_sequence(n);
//        
//        for (int j = 0; j < 2; j++) {
//            result = activate_neuron(n, in_sequence[j], 1);
//            printf("(%f, %f): %f vs %f, error %f\n", in_sequence[j][0], in_sequence[j][1], result, out_sequence[j], n->best_sq_error);
//        }
//    }
//    
//    print_neuron(n);
//    free_neuron(n);
    
    int layers[] = {1, 1};
    int num_layers = 2;
    int num_inputs = 2;
    int num_outputs = 1;
    Neural_Net* net = mk_deep_net(num_inputs, num_outputs, num_layers, layers);
    
    scalar in1[] = {0.0, 1.0};
    scalar in2[] = {1.0, 0.0};
    scalar in3[] = {1.0, 1.0};
    scalar in4[] = {0.0, 0.0};
    
    scalar out1[] = {1.0};
    scalar out2[] = {1.0};
    scalar out3[] = {2.0};
    scalar out4[] = {0.0};
    
    scalar* ins[] = {in1, in2, in3, in4};
    scalar* outs[] = {out1, out2, out3, out4};
    
    scalar* outputs;
    
    for (int i = 0; i < 4; i++) {
        outputs = activate_net(net, ins[i], 0);
        printf("(%f, %f): %f\n", ins[i][0], ins[i][1], outputs[0]);
        free(outputs);
    }
    
    scalar errors[4];
    scalar error = 0;
    for (int i = 0; 1; i++) {
        begin_net_sequence(net);
        train_net(net, 1, ins + i%4, outs + i%4);
        finish_net_sequence(net);
        
//        calc_error = get_net_error(net, 4, ins, outs, 0);
//        if (net->error != calc_error) {
//            printf("WTF. error = %f, calc_error = %f\n", net->error, calc_error);
//            exit(9001);
//        }
        errors[i%4] = net->error;
        
        if (i % 4 == 3) {
            error = 0;
            for (int j = 0; j < 4; j++) {
                error += errors[j];
            }
            error = error / 4;
            if (error < 0.01 || i > 100000) {
                printf("\n");
                printf("%d iterations\n", i);
                printf("error: %f\n", error);
                break;
            }
        }
        if (i % 100000 == 0) {
            printf("\rerror: %f", error);
            fflush(stdout);
        }
    }
    
    printf("\ncurrent:\n");
    for (int i = 0; i < 4; i++) {
        outputs = activate_net(net, ins[i], 0);
        printf("(%f, %f): %f\n", ins[i][0], ins[i][1], outputs[0]);
        free(outputs);
    }
    
    printf("\nbest:\n");
    for (int i = 0; i < 4; i++) {
        outputs = activate_net(net, ins[i], 1);
        printf("(%f, %f): %f\n", ins[i][0], ins[i][1], outputs[0]);
        free(outputs);
    }
    
    return 0;
}

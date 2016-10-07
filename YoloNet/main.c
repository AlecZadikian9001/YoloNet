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
    Jank Neural Network
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
    
    Neuron* n = mk_neuron(2, &neuron_func_tanh, &neuron_dfunc_tanh);
    //n->b_rand_start = 0;
    //n->b_rand_end = 0;
    randomize_neuron(n);
    
    scalar in1[] = {2.0, 3.0};
    scalar in2[] = {9.0, 9.0};
    scalar** in_sequence = emalloc(2 * sizeof(scalar*));
    in_sequence[0] = in1;
    in_sequence[1] = in2;
    
    scalar out_sequence[] = {-0.3, 0.6};
    

    scalar result;
    
    for (int i = 0; i < 100; i++) {
        begin_neuron_sequence(n);
        for (int j = 0; j < 100; j++) {
            train_neuron(n, in_sequence[j % 2], out_sequence[j % 2]);
        }
        finish_neuron_sequence(n);
        
        for (int j = 0; j < 2; j++) {
            result = activate_neuron(n, in_sequence[j], 1);
            printf("(%f, %f): %f vs %f, error %f\n", in_sequence[j][0], in_sequence[j][1], result, out_sequence[j], n->best_sq_error);
        }
    }
    
    print_neuron(n);
    free_neuron(n);
    
    return 0;
}

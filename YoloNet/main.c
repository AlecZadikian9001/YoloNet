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
    
    scalar input[] = {1.0, 1.0};
    scalar output = 0.5;
    scalar result;
    
    for (int i = 0; 1; i++) {
        train_neuron(n, input, output);
        if (n->best_sq_error < 0.000001) {
            printf("good enough after %d iterations\n", i);
            break;
        }
    }
    result = activate_neuron(n, input, 1);
    printf("%f vs %f\n", result, output);
    print_neuron(n);
    free_neuron(n);
    
    return 0;
}

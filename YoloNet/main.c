//
//  main.c
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#include <stdio.h>
#include "general.h"
#include "neuron.h"
#include "net.h"

int main(int argc, const char * argv[]) {
    
    for (int i = 0; i < 100000; i++) {
        Neuron* n = mk_neuron(5, NULL);
        randomize_neuron(n, -1.9, 24.9);
        print_neuron(n);
        free_neuron(n);
    }
    
    return 0;
}

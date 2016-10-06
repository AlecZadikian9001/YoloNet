//
//  net.c
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#include "net.h"

struct Neural_Node {
    Neuron* neuron; // NULL if net input
    struct Neural_Node* inputs; // NULL if net input
};


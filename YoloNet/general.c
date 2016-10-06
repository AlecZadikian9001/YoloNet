//
//  general.c
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "general.h"

void *emalloc(unsigned size)
{
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "out of memory!\n");
        exit(1);
    }
    return p;
}

void azlog(int level, char* str) {
    if (LOG_LEVEL < level) {
        return;
    } else {
        printf("[%d] %s\n", level, str);
    }
}

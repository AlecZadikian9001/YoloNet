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
#include <stdarg.h>

#include "general.h"

void *emalloc(size_t size)
{
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "out of memory!\n");
        exit(1);
    }
    return p;
}

void azlog(int level, const char* format, ...) {
    if (LOG_LEVEL < level) {
        return;
    } else {
        va_list argptr;
        va_start(argptr, format);
        printf("[%d] ", level);
        printf(format, argptr);
        va_end(argptr);
    }
}

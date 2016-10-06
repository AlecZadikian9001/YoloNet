//
//  general.h
//  YoloNet
//
//  Created by Alec Zadikian on 10/5/16.
//  Copyright © 2016 AlecZ. All rights reserved.
//

#ifndef general_h
#define general_h

/* m3m management stuff */
void *emalloc(unsigned size);

/* l0gging (BROKEN FOR NOW) */
#define LOG_ERROR 0
#define LOG_WARN 1
#define LOG_VERBOSE 2
#define LOG_DEBUG 3
#define LOG_TRACE 4

#define LOG_LEVEL LOG_TRACE
void azlog(int level, const char* format, ...); // TODO broken

/* ratchet logging */
#define VERBOSE printf
#define TRACE printf
#define ERROR printf

#endif /* general_h */

/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025
**
**  \section sref Reference
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas,
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504, 2018. https://doi.org/10.1117/1.JBO.23.1.010504
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel
**          based Monte Carlo algorithm for accurate and efficient photon transport
**          modeling in complex bio-tissues," Biomed. Opt. Express, 11(11)
**          pp. 6262-6270. https://doi.org/10.1364/BOE.409468
**
**  \section sformat Formatting
**          Please always run "make pretty" inside the \c src folder before each commit.
**          The above command requires \c astyle to perform automatic formatting.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mcx_neurojson.h"

#define ALLOC_CHUNK  4096

#ifdef _WIN32
    #define popen   _popen
    #define pclose  _pclose
#endif

int runcommand(char* cmd, char* param, char** output) {
    int len = ALLOC_CHUNK, pos = 0;
    char buffer[256] = {'\0'}, fullcmd[ALLOC_CHUNK] = {'\0'};
    FILE* pipe = NULL;

    snprintf(fullcmd, ALLOC_CHUNK, "%s%s", cmd, param);
    pipe = strlen(fullcmd) ? popen(fullcmd, "r") : stdin;

    if (!pipe) {
        return -1;
    }

    free(*output);
    *output = (char*)calloc(ALLOC_CHUNK, 1);

    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        int buflen = strlen(buffer);

        if (buflen > len - pos - 1) {
            *output = (char*)realloc(*output, len + ALLOC_CHUNK);
            len += ALLOC_CHUNK;
        }

        strncpy(*output + pos, buffer, len - pos - 1);
        pos += buflen;
    }

    *output = (char*)realloc(*output, pos + 1);
    pclose(pipe);
    return pos;
}

/***************************************************************************//**
**  \mainpage Mesh-based Monte Carlo (MMC) - a 3D photon simulator
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2010-2024
**
**  \section sref Reference:
**  \li \c (\b Fang2010) Qianqian Fang, <a href="http://www.opticsinfobase.org/abstract.cfm?uri=boe-1-1-165">
**          "Mesh-based Monte Carlo Method Using Fast Ray-Tracing
**          in Plucker Coordinates,"</a> Biomed. Opt. Express, 1(1) 165-175 (2010).
**  \li \c (\b Fang2012) Qianqian Fang and David R. Kaeli,
**           <a href="https://www.osapublishing.org/boe/abstract.cfm?uri=boe-3-12-3223">
**          "Accelerating mesh-based Monte Carlo method on modern CPU architectures,"</a>
**          Biomed. Opt. Express 3(12), 3223-3230 (2012)
**  \li \c (\b Yao2016) Ruoyang Yao, Xavier Intes, and Qianqian Fang,
**          <a href="https://www.osapublishing.org/boe/abstract.cfm?uri=boe-7-1-171">
**          "Generalized mesh-based Monte Carlo for wide-field illumination and detection
**           via mesh retessellation,"</a> Biomed. Optics Express, 7(1), 171-184 (2016)
**  \li \c (\b Fang2019) Qianqian Fang and Shijie Yan,
**          <a href="http://dx.doi.org/10.1117/1.JBO.24.11.115002">
**          "Graphics processing unit-accelerated mesh-based Monte Carlo photon transport
**           simulations,"</a> J. of Biomedical Optics, 24(11), 115002 (2019)
**  \li \c (\b Yuan2021) Yaoshen Yuan, Shijie Yan, and Qianqian Fang,
**          <a href="https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-12-1-147">
**          "Light transport modeling in highly complex tissues using the implicit
**           mesh-based Monte Carlo algorithm,"</a> Biomed. Optics Express, 12(1) 147-161 (2021)
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

    snprintf(fullcmd, ALLOC_CHUNK, "%s%s", cmd, param);
    FILE* pipe = popen(fullcmd, "r");

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

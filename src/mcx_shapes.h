/*******************************************************************************
**
**  \mainpage Monte Carlo eXtreme (MCX)  - GPU accelerated 3D Monte Carlo transport simulation
**
**  \author Qianqian Fang <q.fang at neu.edu>
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  
**  \section slicense License
**        GNU General Public License v3, see LICENSE.txt for details
**
*******************************************************************************/

/***************************************************************************//**
\file    mcx_shapes.h

@brief   MCX shape JSON parser header
*******************************************************************************/

#ifndef _MCEXTREME_RASTERIZER_H
#define _MCEXTREME_RASTERIZER_H

#include <vector_types.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define MAX_SHAPE_ERR 256

typedef struct GridSpace{
        unsigned int **vol;
        uint3  *dim;
	float3 orig;
	int    rowmajor;
} Grid3D;

int mcx_load_jsonshapes(Grid3D *g, char *fname);
int mcx_parse_jsonshapes(cJSON *root, Grid3D *g);
int mcx_parse_shapestring(Grid3D *g, char *shapedata);
int mcx_raster_origin(cJSON *obj, Grid3D *g);
int mcx_raster_sphere(cJSON *obj, Grid3D *g);
int mcx_raster_subgrid(cJSON *obj, Grid3D *g);
int mcx_raster_box(cJSON *obj, Grid3D *g);
int mcx_raster_cylinder(cJSON *obj, Grid3D *g);
int mcx_raster_slabs(cJSON *obj, Grid3D *g);
int mcx_raster_layers(cJSON *obj, Grid3D *g);
int mcx_raster_upperspace(cJSON *obj, Grid3D *g);
int mcx_raster_grid(cJSON *obj, Grid3D *g);
int mcx_find_shapeid(char *shapename);
char *mcx_last_shapeerror();

#ifdef  __cplusplus
}
#endif

#endif

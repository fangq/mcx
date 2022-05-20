/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2022
**
**  \section sref Reference:
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
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_shapes.c
@brief   3D shape parsing and rasterization unit

In this unit, we load and parse a JSON-formatted shape file,
rasterize the 3D objects and subsequently add to a voxelated volume.
*******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "cjson/cJSON.h"
#include "mcx_shapes.h"

#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MAX(a,b)           ((a)>(b)?(a):(b))

const char* ShapeTags[] = {"Name", "Origin", "Grid", "Subgrid", "Sphere", "Box", "XSlabs",
                           "YSlabs", "ZSlabs", "XLayers", "YLayers", "ZLayers",
                           "Cylinder", "UpperSpace", NULL
                          };
int (*Rasterizers[])(cJSON* obj, Grid3D* g) = {NULL, mcx_raster_origin, mcx_raster_grid, mcx_raster_subgrid,
                                               mcx_raster_sphere, mcx_raster_box, mcx_raster_slabs, mcx_raster_slabs,
                                               mcx_raster_slabs, mcx_raster_layers, mcx_raster_layers,
                                               mcx_raster_layers, mcx_raster_cylinder, mcx_raster_upperspace, NULL
                                              };
char ErrorMsg[MAX_SHAPE_ERR] = {'\0'};


/*******************************************************************************/
/*! \fn int mcx_load_jsonshapes(Grid3D *g, char *fname)

    @brief Load a JSON-formatted shape file and process
    \param g A structure pointing to the volume and dimension data
    \param fname The file name string to the JSON shape file
*/

int mcx_load_jsonshapes(Grid3D* g, char* fname) {
    FILE* fp = fopen(fname, "rb");

    if (fp == NULL) {
        sprintf(ErrorMsg, "Can not read the JSON file");
        return -2;
    }

    if (g == NULL) {
        sprintf(ErrorMsg, "The background grid is not initialized");
        return -3;
    }

    if (strstr(fname, ".json") != NULL) {
        char* jbuf;
        int len, err;

        fseek (fp, 0, SEEK_END);
        len = ftell(fp) + 1;
        jbuf = (char*)malloc(len);
        rewind(fp);

        if (fread(jbuf, len - 1, 1, fp) != 1) {
            sprintf(ErrorMsg, "Failed when reading a JSON file from %s", fname);
            return -1;
        }

        jbuf[len - 1] = '\0';
        fclose(fp);

        if ((err = mcx_parse_shapestring(g, jbuf))) { /*error msg is generated inside*/
            free(jbuf);
            return err;
        }

        free(jbuf);
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_parse_shapestring(Grid3D *g, char *shapedata)

    @brief Load JSON-formatted shape definitions from a string
    \param g A structure pointing to the volume and dimension data
    \param shapedata A string containg the JSON shape data
*/

int mcx_parse_shapestring(Grid3D* g, char* shapedata) {
    if (g && shapedata) {
        cJSON* jroot = cJSON_Parse(shapedata);

        if (jroot) {
            int err;

            if ((err = mcx_parse_jsonshapes(jroot, g))) { /*error msg is generated inside*/
                return err;
            }

            cJSON_Delete(jroot);
        } else {
            char* ptrold, *ptr = (char*)cJSON_GetErrorPtr();

            if (ptr) {
                ptrold = strstr(shapedata, ptr);
            }

            if (ptr && ptrold) {
                char* offs = (ptrold - shapedata >= 50) ? ptrold - 50 : shapedata;

                while (offs < ptrold) {
                    fprintf(stderr, "%c", *offs);
                    offs++;
                }

                fprintf(stderr, "<error>%.50s\n", ptrold);
            }

            sprintf(ErrorMsg, "Invalid JSON file");
            return -2;
        }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_parse_jsonshapes(cJSON *root, Grid3D *g)

    @brief Parse a JSON-formatted shape file and rasterize the objects to the volume
    \param root A cJSON pointer points to the root obj of the shape block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_parse_jsonshapes(cJSON* root, Grid3D* g) {
    cJSON* shapes;
    int id, objcount = 1;
    int (*raster)(cJSON * obj, Grid3D * g) = NULL;

    if (g && g->dim && g->dim->x * g->dim->y * g->dim->z > 0) {
        if (g->vol && *(g->vol)) {
            (*(g->vol)) = (unsigned int*)realloc((void*)(*(g->vol)), sizeof(unsigned int) * g->dim->x * g->dim->y * g->dim->z);
        } else {
            (*(g->vol)) = (unsigned int*)calloc(sizeof(unsigned int), g->dim->x * g->dim->y * g->dim->z);
        }
    }

    shapes  = cJSON_GetObjectItem(root, "Shapes");

    if (shapes) {
        shapes = shapes->child;

        while (shapes && shapes->child) {
            id = mcx_find_shapeid(shapes->child->string);

            if (id >= 0 && id < sizeof(ShapeTags) / sizeof(char*)) {
                raster = Rasterizers[id];
            } else {
                sprintf(ErrorMsg, "The #%d element in the Shapes section has an undefined tag %s",
                        objcount, shapes->child->string);
                return -(objcount + 100);
            }

            if (raster) {
                id = raster(shapes->child, g);

                if (id) {
                    return id;
                }
            }

            objcount++;
            shapes = shapes->next;
        }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_origin(cJSON *obj, Grid3D *g)

    @brief Reset the origin of the domain, default is [0,0,0]
    \param obj A cJSON pointer points to the Origin obj block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_origin(cJSON* obj, Grid3D* g) {
    if (obj && cJSON_GetArraySize(obj) == 3) {
        g->orig.x = obj->child->valuedouble;
        g->orig.y = obj->child->next->valuedouble;
        g->orig.z = obj->child->next->next->valuedouble;
    } else {
        sprintf(ErrorMsg, "An Origin record does not contain a triplet");
        return 1;
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_sphere(cJSON *obj, Grid3D *g)

    @brief Rasterize a 3D sphere and add to the volume
    \param obj A cJSON pointer points to the sphere obj block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_sphere(cJSON* obj, Grid3D* g) {
    float O[3], R, R2, dx, dy, dz;
    int i, j, k, dimxy, dimyz, tag = 0;

    cJSON* val = cJSON_GetObjectItem(obj, "O");

    if (val && cJSON_GetArraySize(val) == 3) {
        O[0] = val->child->valuedouble;
        O[1] = val->child->next->valuedouble;
        O[2] = val->child->next->next->valuedouble;
    } else {
        sprintf(ErrorMsg, "A Sphere command misses O field");
        return 1;
    }

    val = cJSON_GetObjectItem(obj, "R");

    if (val) {
        R = val->valuedouble;
    } else {
        sprintf(ErrorMsg, "A Sphere command misses R field");
        return 2;
    }

    val = cJSON_GetObjectItem(obj, "Tag");

    if (val) {
        tag = val->valueint;
    }

    R2 = R * R;
    dimxy = g->dim->x * g->dim->y;
    dimyz = g->dim->y * g->dim->z;

    for (k = 0; k < g->dim->z; k++) {
        dz = (k + 0.5f) - O[2];

        for (j = 0; j < g->dim->y; j++) {
            dy = (j + 0.5f) - O[1];

            for (i = 0; i < g->dim->x; i++) {
                dx = (i + 0.5f) - O[0];

                if (dx * dx + dy * dy + dz * dz <= R2) {
                    (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
                }
            }
        }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_subgrid(cJSON *obj, Grid3D *g)

    @brief Rasterize a 3D rectangular region and add to the volume
    \param obj A cJSON pointer points to the rectangular obj block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_subgrid(cJSON* obj, Grid3D* g) {
    int O[3] = {0}, S[3] = {0};
    int i, j, k, dimxy, dimyz, tag = 0;

    cJSON* val = cJSON_GetObjectItem(obj, "O");

    if (val && cJSON_GetArraySize(val) == 3) {
        O[0] = val->child->valueint - 1 - g->orig.x; /*Subgrid: numbers start from 1*/
        O[1] = val->child->next->valueint - 1 - g->orig.y;
        O[2] = val->child->next->next->valueint - 1 - g->orig.z;
    } else {
        sprintf(ErrorMsg, "A Subgrid command misses O field");
        return 1;
    }

    val = cJSON_GetObjectItem(obj, "Size");

    if (val && cJSON_GetArraySize(val) == 3) {
        S[0] = val->child->valueint;
        S[1] = val->child->next->valueint;
        S[2] = val->child->next->next->valueint;
    } else {
        sprintf(ErrorMsg, "A Box command misses Size field");
        return 2;
    }

    val = cJSON_GetObjectItem(obj, "Tag");

    if (val) {
        tag = val->valueint;
    }

    dimxy = g->dim->x * g->dim->y;
    dimyz = g->dim->y * g->dim->z;

    for (k = 0; k < g->dim->z; k++) {
        if (k < O[2] || k > O[2] + S[2]) {
            continue;
        }

        for (j = 0; j < g->dim->y; j++) {
            if (j < O[1] || j > O[1] + S[1]) {
                continue;
            }

            for (i = 0; i < g->dim->x; i++) {
                if (i < O[0] || i > O[0] + S[0]) {
                    continue;
                }

                (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
            }
        }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_box(cJSON *obj, Grid3D *g)

    @brief Rasterize a 3D rectangular region and add to the volume
    \param obj A cJSON pointer points to the rectangular obj block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_box(cJSON* obj, Grid3D* g) {
    float O[3] = {0.f}, S[3] = {0.f}, dx, dy, dz;
    int i, j, k, dimxy, dimyz, tag = 0;

    cJSON* val = cJSON_GetObjectItem(obj, "O");

    if (val && cJSON_GetArraySize(val) == 3) {
        O[0] = val->child->valuedouble - g->orig.x;
        O[1] = val->child->next->valuedouble - g->orig.y;
        O[2] = val->child->next->next->valuedouble - g->orig.z;
    } else {
        sprintf(ErrorMsg, "A Box command misses O field");
        return 1;
    }

    val = cJSON_GetObjectItem(obj, "Size");

    if (val && cJSON_GetArraySize(val) == 3) {
        S[0] = val->child->valuedouble;
        S[1] = val->child->next->valuedouble;
        S[2] = val->child->next->next->valuedouble;
    } else {
        sprintf(ErrorMsg, "A Box command misses Size field");
        return 2;
    }

    val = cJSON_GetObjectItem(obj, "Tag");

    if (val) {
        tag = val->valueint;
    }

    dimxy = g->dim->x * g->dim->y;
    dimyz = g->dim->y * g->dim->z;

    for (k = 0; k < g->dim->z; k++) {
        dz = (k + 0.5f);

        if (dz < O[2] || dz > O[2] + S[2]) {
            continue;
        }

        for (j = 0; j < g->dim->y; j++) {
            dy = (j + 0.5f);

            if (dy < O[1] || dy > O[1] + S[1]) {
                continue;
            }

            for (i = 0; i < g->dim->x; i++) {
                dx = (i + 0.5f);

                if (dx < O[0] || dx > O[0] + S[0]) {
                    continue;
                }

                (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
            }
        }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_cylinder(cJSON *obj, Grid3D *g)

    @brief Rasterize a finite 3D cylindrical region and add to the volume
    \param obj A cJSON pointer points to the cylindrical obj block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_cylinder(cJSON* obj, Grid3D* g) {
    float C0[3], C1[3], v[3], d0, R, R2, d, dx, dy, dz;
    int i, j, k, dimxy, dimyz, tag = 0;

    cJSON* val = cJSON_GetObjectItem(obj, "C0");

    if (val && cJSON_GetArraySize(val) == 3) {
        C0[0] = val->child->valuedouble - g->orig.x;
        C0[1] = val->child->next->valuedouble - g->orig.y;
        C0[2] = val->child->next->next->valuedouble - g->orig.z;
    } else {
        sprintf(ErrorMsg, "A Cylinder command misses C0 field");
        return 1;
    }

    val = cJSON_GetObjectItem(obj, "C1");

    if (val && cJSON_GetArraySize(val) == 3) {
        C1[0] = val->child->valuedouble - g->orig.x;
        C1[1] = val->child->next->valuedouble - g->orig.y;
        C1[2] = val->child->next->next->valuedouble - g->orig.z;
    } else {
        sprintf(ErrorMsg, "A Cylinder command misses C1 field");
        return 1;
    }

    v[0] = C1[0] - C0[0];
    v[1] = C1[1] - C0[1];
    v[2] = C1[2] - C0[2];
    d0 = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    if (d0 == 0.f) {
        sprintf(ErrorMsg, "Coincident end points in the definition of Cylinder command");
        return 1;
    }

    v[0] /= d0;
    v[1] /= d0;
    v[2] /= d0;

    val = cJSON_GetObjectItem(obj, "R");

    if (val) {
        R = val->valuedouble;
    } else {
        sprintf(ErrorMsg, "A Sphere command misses R field");
        return 2;
    }

    val = cJSON_GetObjectItem(obj, "Tag");

    if (val) {
        tag = val->valueint;
    }

    R2 = R * R;
    dimxy = g->dim->x * g->dim->y;
    dimyz = g->dim->y * g->dim->z;

    for (k = 0; k < g->dim->z; k++) {
        dz = (k + 0.5f) - C0[2];

        for (j = 0; j < g->dim->y; j++) {
            dy = (j + 0.5f) - C0[1];

            for (i = 0; i < g->dim->x; i++) {
                dx = (i + 0.5f) - C0[0];
                d = v[0] * dx + v[1] * dy + v[2] * dz; /* (|PC0|*cos(theta)) */

                if (d > d0 || d < 0.f) {
                    continue;
                }

                d = dx * dx + dy * dy + dz * dz - d * d; /* (|PC0|*sin(theta))^2 */

                if (d <= R2) {
                    (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
                }
            }
        }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_slabs(cJSON *obj, Grid3D *g)

    @brief Rasterize a 3D layered-slab structure and add to the volume
    \param obj A cJSON pointer points to the layered-slab structure block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_slabs(cJSON* obj, Grid3D* g) {
    float* bd = NULL;
    int i, j, k, dimxy, dimyz, tag = 0, num = 0, num2, dir = -1, p;
    cJSON* item, *val;

    if (strcmp(obj->string, "XSlabs") == 0) {
        dir = 0;
    } else if (strcmp(obj->string, "YSlabs") == 0) {
        dir = 1;
    } else if (strcmp(obj->string, "ZSlabs") == 0) {
        dir = 2;
    } else {
        sprintf(ErrorMsg, "Unsupported layer command");
        return 1;
    }

    val = cJSON_GetObjectItem(obj, "Bound");

    if (val && val->type == cJSON_Array) {
        num = cJSON_GetArraySize(val);

        if (num == 0) {
            return 0;
        }

        if (num == 2 && val->child->type != cJSON_Array) {
            item = val;
            bd = (float*)malloc(cJSON_GetArraySize(obj) * sizeof(float));
            num = 1;
        } else {
            item = val->child;
            bd = (float*)malloc(cJSON_GetArraySize(obj) * 2 * sizeof(float));
        }

        for (i = 0; i < num; i++) {
            if (cJSON_GetArraySize(item) != 2) {
                sprintf(ErrorMsg, "The Bound field must contain number pairs");
                return 2;
            }

            bd[i << 1]    = MAX(item->child->valuedouble - 0.5, 0.0); /*inclusive of both ends*/
            bd[(i << 1) + 1] = MIN(item->child->next->valuedouble, (&(g->dim->x))[dir]);

            if (bd[(i << 1) + 1] < bd[i << 1]) {
                float tmp = bd[(i << 1) + 1];
                bd[(i << 1) + 1] = bd[i << 1];
                bd[i << 1] = tmp;
            }

            item = item->next;
        }
    } else {
        sprintf(ErrorMsg, "A %s command misses Bound field or not an array", obj->string);
        return 1;
    }

    if (num == 0) {
        return 0;
    }

    num2 = num << 1;

    val = cJSON_GetObjectItem(obj, "Tag");

    if (val) {
        tag = val->valueint;
    }

    dimxy = g->dim->x * g->dim->y;
    dimyz = g->dim->y * g->dim->z;

    if (dir == 0) {
        for (p = 0; p < num2; p += 2)
            for (k = 0; k < g->dim->z; k++)
                for (j = 0; j < g->dim->y; j++)
                    for (i = (int)(bd[p]); i < (int)(bd[p + 1]); i++) {
                        (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
                    }
    } else if (dir == 1) {
        for (p = 0; p < num2; p += 2)
            for (k = 0; k < g->dim->z; k++)
                for (j = (int)(bd[p]); j < (int)(bd[p + 1]); j++)
                    for (i = 0; i < g->dim->x; i++) {
                        (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
                    }
    } else if (dir == 2) {
        for (p = 0; p < num2; p += 2)
            for (k = (int)(bd[p]); k < (int)(bd[p + 1]); k++)
                for (j = 0; j < g->dim->y; j++)
                    for (i = 0; i < g->dim->x; i++) {
                        (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
                    }
    }

    if (bd) {
        free(bd);
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_layers(cJSON *obj, Grid3D *g)

    @brief Rasterize a 3D layer structure and add to the volume
    \param obj A cJSON pointer points to the layer structure block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_layers(cJSON* obj, Grid3D* g) {
    int* bd = NULL;
    int i, j, k, dimxy, dimyz, num = 0, num3, dir = -1, p;
    cJSON* item;

    if (strcmp(obj->string, "XLayers") == 0) {
        dir = 0;
    } else if (strcmp(obj->string, "YLayers") == 0) {
        dir = 1;
    } else if (strcmp(obj->string, "ZLayers") == 0) {
        dir = 2;
    } else {
        sprintf(ErrorMsg, "Unsupported command %s", obj->string);
        return 1;
    }

    if (obj && obj->type == cJSON_Array) {
        num = cJSON_GetArraySize(obj);

        if (num == 0) {
            return 0;
        }

        if (num == 3 && obj->child->type != cJSON_Array) {
            item = obj;
            bd = (int*)malloc(cJSON_GetArraySize(obj) * sizeof(int));
            num = 1;
        } else {
            item = obj->child;
            bd = (int*)malloc(cJSON_GetArraySize(obj) * 3 * sizeof(int));
        }

        for (i = 0; i < num; i++) {
            if (cJSON_GetArraySize(item) != 3) {
                sprintf(ErrorMsg, "The %s must contain integer triplets", obj->string);
                return 2;
            }

            bd[i * 3]  = MAX(item->child->valueint, 1) - 1; /*inclusive of both ends*/
            bd[i * 3 + 1] = MIN(item->child->next->valueint, (&(g->dim->x))[dir]);
            bd[i * 3 + 2] = item->child->next->next->valueint;

            if (bd[i * 3 + 1] < bd[i * 3]) {
                float tmp = bd[i * 3 + 1];
                bd[i * 3 + 1] = bd[i * 3];
                bd[i * 3] = tmp;
            }

            item = item->next;
        }
    } else {
        sprintf(ErrorMsg, "A %s object must be an array", obj->string);
        return 1;
    }

    if (num == 0) {
        return 0;
    }

    num3 = num * 3;

    dimxy = g->dim->x * g->dim->y;
    dimyz = g->dim->y * g->dim->z;

    if (dir == 0) {
        for (p = 0; p < num3; p += 3)
            for (k = 0; k < g->dim->z; k++)
                for (j = 0; j < g->dim->y; j++)
                    for (i = bd[p]; i < bd[p + 1]; i++) {
                        (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = bd[p + 2];
                    }
    } else if (dir == 1) {
        for (p = 0; p < num3; p += 3)
            for (k = 0; k < g->dim->z; k++)
                for (j = bd[p]; j < bd[p + 1]; j++)
                    for (i = 0; i < g->dim->x; i++) {
                        (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = bd[p + 2];
                    }
    } else if (dir == 2) {
        for (p = 0; p < num3; p += 3)
            for (k = bd[p]; k < bd[p + 1]; k++)
                for (j = 0; j < g->dim->y; j++)
                    for (i = 0; i < g->dim->x; i++) {
                        (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = bd[p + 2];
                    }
    }

    if (bd) {
        free(bd);
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_upperspace(cJSON *obj, Grid3D *g)

    @brief Rasterize a 3D semi-space region and add to the volume
    \param obj A cJSON pointer points to the semi-space object block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_upperspace(cJSON* obj, Grid3D* g) {
    float C[4], dx, dy, dz;
    int i, j, k, dimxy, dimyz, tag = 0;

    cJSON* val = cJSON_GetObjectItem(obj, "Coef");

    if (val && cJSON_GetArraySize(val) == 4) {
        C[0] = val->child->valuedouble;
        C[1] = val->child->next->valuedouble;
        C[2] = val->child->next->next->valuedouble;
        C[3] = val->child->next->next->next->valuedouble;
    } else {
        sprintf(ErrorMsg, "An UpperSpace command misses Coef field");
        return 1;
    }

    val = cJSON_GetObjectItem(obj, "Tag");

    if (val) {
        tag = val->valueint;
    }

    dimxy = g->dim->x * g->dim->y;
    dimyz = g->dim->y * g->dim->z;

    for (k = 0; k < g->dim->z; k++) {
        dz = (k + 0.5f);

        for (j = 0; j < g->dim->y; j++) {
            dy = (j + 0.5f);

            for (i = 0; i < g->dim->x; i++) {
                dx = (i + 0.5f);

                if (C[0]*dx + C[1]*dy + C[2]*dz > C[3]) {
                    (*(g->vol))[g->rowmajor ? i * dimyz + j * g->dim->z + k : k * dimxy + j * g->dim->x + i] = tag;
                }
            }
        }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_raster_grid(cJSON *obj, Grid3D *g)

    @brief Recreate the background grid with a different dimension or medium
    \param obj A cJSON pointer points to the background grid block
    \param g  A structure pointing to the volume and dimension data
*/

int mcx_raster_grid(cJSON* obj, Grid3D* g) {
    int dimxy, dimxyz;
    unsigned int tag = 0;

    cJSON* val = cJSON_GetObjectItem(obj, "Size");

    if (val && cJSON_GetArraySize(val) == 3) {
        g->dim->x = val->child->valuedouble;
        g->dim->y = val->child->next->valuedouble;
        g->dim->z = val->child->next->next->valuedouble;
        dimxy = g->dim->x * g->dim->y;
        dimxyz = dimxy * g->dim->z;

        if (dimxy * g->dim->z > 0) {
            if (g->vol && *g->vol) {
                free(*(g->vol));
            }

            *(g->vol) = (unsigned int*)calloc(dimxy * sizeof(int), g->dim->z);
        } else {
            *(g->vol) = NULL;
        }
    } else {
        sprintf(ErrorMsg, "A Grid command misses Size field");
        return 1;
    }

    val = cJSON_GetObjectItem(obj, "Tag");

    if (val) {
        int i;
        tag = val->valueint;

        if (g->vol && *(g->vol))
            for (i = 0; i < dimxyz; i++) {
                (*(g->vol))[i] = tag;
            }
    }

    return 0;
}

/*******************************************************************************/
/*! \fn int mcx_find_shapeid(char *shapename)

    @brief Look up the JSON object tag and return the index to the processing function
    \param shapename The string of the JSON shape object
*/

int mcx_find_shapeid(char* shapename) {
    int i = 0;

    while (ShapeTags[i]) {
        if (strcmp(shapename, ShapeTags[i]) == 0) {
            return i;
        }

        i++;
    }

    return -1;
}

/*******************************************************************************/
/*! \fn char * mcx_last_shapeerror()

    @brief return the last error message encountered in the processing
*/

char* mcx_last_shapeerror() {
    return ErrorMsg;
}

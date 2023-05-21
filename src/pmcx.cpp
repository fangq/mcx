/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Matin Raayai Ardakani <raayaiardakani.m at northeastern.edu>
**  \copyright Matin Raayai Ardakani, 2022
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
\file    pmcx.cpp

@brief   Python interface using Pybind11 for MCX
*******************************************************************************/
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_const.h"
#include "mcx_shapes.h"
#include <pybind11/iostream.h>

// Python binding for runtime_error exception in Python.
namespace pybind11 {
PYBIND11_RUNTIME_EXCEPTION(runtime_error, PyExc_RuntimeError);
}

namespace py = pybind11;

#if defined(USE_XOROSHIRO128P_RAND)
    #define RAND_WORD_LEN 4
#elif defined(USE_POSIX_RAND)
    #define RAND_WORD_LEN 4
#else
    #define RAND_WORD_LEN 4       /**< number of Words per RNG state */
#endif

float* det_ps = nullptr;     //! buffer to receive data from cfg.detphotons field
int dim_det_ps[2] = {0, 0};  //! dimensions of the cfg.detphotons array
int seed_byte = 0;

/**
 * Macro to find and extract a scalar property from a source Python dictionary configuration and assign it in a destination
 * MCX Config. The scalar is cast to the python type before assignment.
 */
#define GET_SCALAR_FIELD(src_pydict, dst_mcx_config, property, py_type) if ((src_pydict).contains(#property))\
    {try {(dst_mcx_config).property = py_type((src_pydict)[#property]);\
            std::cout << #property << ": " << (float) (dst_mcx_config).property << std::endl;} \
        catch (const std::runtime_error &err)\
        {throw py::type_error(std::string("Failed to assign MCX property " + std::string(#property) + ". Reason: " + err.what()));}\
    }

#define GET_VEC3_FIELD(src, dst, prop, type) if (src.contains(#prop)) {try {auto list = py::list(src[#prop]);\
            dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>()};\
            std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << "]\n";} \
        catch (const std::runtime_error &err ) {throw py::type_error(std::string("Failed to assign MCX property " + std::string(#prop) + ". Reason: " + err.what()));}}

#define GET_VEC4_FIELD(src, dst, prop, type) if (src.contains(#prop)) {try {auto list = py::list(src[#prop]);\
            dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list[3].cast<type>()}; \
            std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << ", " << dst.prop.w << "]\n";} \
        catch (const std::runtime_error &err ) {throw py::type_error(std::string("Failed to assign MCX property " + std::string(#prop) + ". Reason: " + err.what()));}}

#define GET_VEC34_FIELD(src, dst, prop, type) if (src.contains(#prop)) {try {auto list = py::list(src[#prop]);\
            dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list.size() == 4 ? list[3].cast<type>() : 1}; \
            std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z;\
            if (list.size() == 4) std::cout << ", " << dst.prop.w; std::cout << "]\n";}                                                 \
        catch (const std::runtime_error &err ) {throw py::type_error(std::string("Failed to assign MCX property " + std::string(#prop) + ". Reason: " + err.what()));}                                                                 \
    }

/**
 * Determines the type of volume passed to the interface and decides how to copy it to MCXConfig.
 * @param user_cfg
 * @param mcx_config
 */
void parseVolume(const py::dict& user_cfg, Config& mcx_config) {
    if (!user_cfg.contains("vol")) {
        throw py::value_error("Configuration must specify a 2/3/4D volume.");
    }

    auto volume_handle = user_cfg["vol"];

    // Free the volume
    if (mcx_config.vol) {
        free(mcx_config.vol);
    }

    unsigned int dim_xyz = 0;

    // Data type-specific logic
    if (py::array_t<int8_t>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<int8_t, py::array::f_style>::ensure(volume_handle);
        auto buffer = f_style_volume.request();
        int i = buffer.shape.size() == 4;
        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                          static_cast<unsigned int>(buffer.shape.at(i + 1)),
                          static_cast<unsigned int>(buffer.shape.at(i + 2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));

        if (i == 1) {
            if (buffer.shape.at(0) == 4) {
                mcx_config.mediabyte = MEDIA_ASGN_BYTE;
                memcpy(mcx_config.vol, buffer.ptr, dim_xyz * sizeof(unsigned int));
            } else if (buffer.shape.at(0) == 8) {
                mcx_config.mediabyte = MEDIA_2LABEL_SPLIT;
                auto val = (unsigned char*) buffer.ptr;

                if (mcx_config.vol) {
                    free(mcx_config.vol);
                }

                mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz << 3));
                memcpy(mcx_config.vol, val, (dim_xyz << 3));
            }
        } else {
            mcx_config.mediabyte = 1;

            for (i = 0; i < buffer.size; i++) {
                mcx_config.vol[i] = static_cast<unsigned char*>(buffer.ptr)[i];
            }
        }
    } else if (py::array_t<int16_t>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<int16_t, py::array::f_style>::ensure(volume_handle);
        auto buffer = f_style_volume.request();
        int i = buffer.shape.size() == 4;
        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                          static_cast<unsigned int>(buffer.shape.at(i + 1)),
                          static_cast<unsigned int>(buffer.shape.at(i + 2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));

        if (i == 1) {
            if (buffer.shape.at(0) == 3) {
                mcx_config.mediabyte = MEDIA_2LABEL_MIX;
                auto* val = (unsigned short*) buffer.ptr;
                union {
                    unsigned short h[2];
                    unsigned char c[4];
                    unsigned int i[1];
                } f2bh;

                for (i = 0; i < dim_xyz; i++) {
                    f2bh.c[0] = val[i * 3] & 0xFF;
                    f2bh.c[1] = val[i * 3 + 1] & 0xFF;
                    f2bh.h[1] = val[i * 3 + 2] & 0x7FFF;
                    mcx_config.vol[i] = f2bh.i[0];
                }
            } else if (buffer.shape.at(0) == 2) {
                mcx_config.mediabyte = MEDIA_AS_SHORT;
                memcpy(mcx_config.vol, buffer.ptr, dim_xyz * sizeof(unsigned int));
            }
        } else {
            mcx_config.mediabyte = 2;
            mcx_config.vol = static_cast<unsigned int*>(malloc(buffer.size * sizeof(unsigned int)));

            for (i = 0; i < buffer.size; i++) {
                mcx_config.vol[i] = static_cast<unsigned short*>(buffer.ptr)[i];
            }
        }
    } else if (py::array_t<int32_t>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<int32_t, py::array::f_style>::ensure(volume_handle);
        mcx_config.mediabyte = 4;
        auto buffer = f_style_volume.request();

        if (buffer.shape.size() == 4) {
            throw py::value_error("Invalid volume dims for int32_t volume.");
        }

        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                          static_cast<unsigned int>(buffer.shape.at(1)),
                          static_cast<unsigned int>(buffer.shape.at(2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));
        memcpy(mcx_config.vol, buffer.ptr, buffer.size * sizeof(unsigned int));
    } else if (py::array_t<uint8_t>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<uint8_t, py::array::f_style>::ensure(volume_handle);
        mcx_config.mediabyte = 1;
        auto buffer = f_style_volume.request();

        if (buffer.shape.size() == 4) {
            throw py::value_error("Invalid volume dims for uint8_t volume.");
        }

        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                          static_cast<unsigned int>(buffer.shape.at(1)),
                          static_cast<unsigned int>(buffer.shape.at(2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));

        for (int i = 0; i < buffer.size; i++) {
            mcx_config.vol[i] = static_cast<unsigned char*>(buffer.ptr)[i];
        }
    } else if (py::array_t<uint16_t>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<uint16_t, py::array::f_style>::ensure(volume_handle);
        mcx_config.mediabyte = 2;
        auto buffer = f_style_volume.request();

        if (buffer.shape.size() == 4) {
            throw py::value_error("Invalid volume dims for uint16_t volume.");
        }

        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                          static_cast<unsigned int>(buffer.shape.at(1)),
                          static_cast<unsigned int>(buffer.shape.at(2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));

        for (int i = 0; i < buffer.size; i++) {
            mcx_config.vol[i] = static_cast<unsigned short*>(buffer.ptr)[i];
        }
    } else if (py::array_t<uint32_t>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<uint32_t, py::array::f_style>::ensure(volume_handle);
        mcx_config.mediabyte = 8;
        auto buffer = f_style_volume.request();

        if (buffer.shape.size() == 4) {
            throw py::value_error("Invalid volume dims for uint32_t volume.");
        }

        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                          static_cast<unsigned int>(buffer.shape.at(1)),
                          static_cast<unsigned int>(buffer.shape.at(2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));
        memcpy(mcx_config.vol, buffer.ptr, buffer.size * sizeof(unsigned int));
    } else if (py::array_t<float>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<float, py::array::f_style>::ensure(volume_handle);
        auto buffer = f_style_volume.request();
        int i = buffer.shape.size() == 4;
        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                          static_cast<unsigned int>(buffer.shape.at(i + 1)),
                          static_cast<unsigned int>(buffer.shape.at(i + 2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));

        if (i) {
            switch (buffer.shape.at(0)) {
                case 3: {
                    mcx_config.mediabyte = MEDIA_LABEL_HALF;
                    auto val = (float*) buffer.ptr;
                    union {
                        float f[3];
                        unsigned int i[3];
                        unsigned short h[2];
                        unsigned char c[4];
                    } f2bh;
                    unsigned short tmp;

                    for (i = 0; i < dim_xyz; i++) {
                        f2bh.f[2] = val[i * 3];
                        f2bh.f[1] = val[i * 3 + 1];
                        f2bh.f[0] = val[i * 3 + 2];

                        if (f2bh.f[1] < 0.f || f2bh.f[1] >= 4.f || f2bh.f[0] < 0.f) {
                            throw py::value_error("the 2nd volume must have an integer value between 0 and 3");
                        }

                        f2bh.h[0] = ((((unsigned char) (f2bh.f[1]) & 0x3) << 14) | (unsigned short) (f2bh.f[0]));

                        f2bh.h[1] = (f2bh.i[2] >> 31) << 5;
                        tmp = (f2bh.i[2] >> 23) & 0xff;
                        tmp = (tmp - 0x70) & ((unsigned int) ((int) (0x70 - tmp) >> 4) >> 27);
                        f2bh.h[1] = (f2bh.h[1] | tmp) << 10;
                        f2bh.h[1] |= (f2bh.i[2] >> 13) & 0x3ff;

                        mcx_config.vol[i] = f2bh.i[0];
                    }

                    break;
                }

                case 2: {
                    mcx_config.mediabyte = MEDIA_AS_F2H;
                    auto val = (float*) buffer.ptr;
                    union {
                        float f[2];
                        unsigned int i[2];
                        unsigned short h[2];
                    } f2h;
                    unsigned short tmp, m;

                    for (i = 0; i < dim_xyz; i++) {
                        f2h.f[0] = val[i << 1];
                        f2h.f[1] = val[(i << 1) + 1];

                        if (f2h.f[0] != f2h.f[0]
                                || f2h.f[1] != f2h.f[1]) { /*if one of mua/mus is nan in continuous medium, convert to 0-voxel*/
                            mcx_config.vol[i] = 0;
                            continue;
                        }

                        /**
                        float to half conversion
                        https://stackoverflow.com/questions/3026441/float32-to-float16/5587983#5587983
                        https://gamedev.stackexchange.com/a/17410  (for denorms)
                        */
                        m = ((f2h.i[0] >> 13) & 0x03ff);
                        tmp = (f2h.i[0] >> 23) & 0xff; /*exponent*/
                        tmp = (tmp - 0x70) & ((unsigned int) ((int) (0x70 - tmp) >> 4) >> 27);

                        if (m < 0x10 && tmp == 0) { /*handle denorms - between 2^-24 and 2^-14*/
                            unsigned short sign = (f2h.i[0] >> 16) & 0x8000;
                            tmp = ((f2h.i[0] >> 23) & 0xff);
                            m = (f2h.i[0] >> 12) & 0x07ff;
                            m |= 0x0800u;
                            f2h.h[0] = sign | ((m >> (114 - tmp)) + ((m >> (113 - tmp)) & 1));
                        } else {
                            f2h.h[0] = (f2h.i[0] >> 31) << 5;
                            f2h.h[0] = (f2h.h[0] | tmp) << 10;
                            f2h.h[0] |= (f2h.i[0] >> 13) & 0x3ff;
                        }

                        m = ((f2h.i[1] >> 13) & 0x03ff);
                        tmp = (f2h.i[1] >> 23) & 0xff; /*exponent*/
                        tmp = (tmp - 0x70) & ((unsigned int) ((int) (0x70 - tmp) >> 4) >> 27);

                        if (m < 0x10 && tmp == 0) { /*handle denorms - between 2^-24 and 2^-14*/
                            unsigned short sign = (f2h.i[1] >> 16) & 0x8000;
                            tmp = ((f2h.i[1] >> 23) & 0xff);
                            m = (f2h.i[1] >> 12) & 0x07ff;
                            m |= 0x0800u;
                            f2h.h[1] = sign | ((m >> (114 - tmp)) + ((m >> (113 - tmp)) & 1));
                        } else {
                            f2h.h[1] = (f2h.i[1] >> 31) << 5;
                            f2h.h[1] = (f2h.h[1] | tmp) << 10;
                            f2h.h[1] |= (f2h.i[1] >> 13) & 0x3ff;
                        }

                        if (f2h.i[0] == 0) { /*avoid being detected as a 0-label voxel, setting mus=EPS_fp16*/
                            f2h.i[0] = 0x00010000;
                        }

                        mcx_config.vol[i] = f2h.i[0];
                    }

                    break;
                }

                case 1: {
                    mcx_config.mediabyte = MEDIA_MUA_FLOAT;
                    union {
                        float f;
                        uint i;
                    } f2i;
                    auto* val = (float*) buffer.ptr;

                    for (i = 0; i < dim_xyz; i++) {
                        f2i.f = val[i];

                        if (f2i.i == 0) { /*avoid being detected as a 0-label voxel*/
                            f2i.f = EPS;
                        }

                        if (val[i] != val[i]) { /*if input is nan in continuous medium, convert to 0-voxel*/
                            f2i.i = 0;
                        }

                        mcx_config.vol[i] = f2i.i;
                    }

                    break;
                }

                default:
                    throw py::value_error("Invalid array for vol array.");
            }
        } else {
            mcx_config.mediabyte = 14;
            mcx_config.vol = static_cast<unsigned int*>(malloc(buffer.size * sizeof(unsigned int)));

            for (i = 0; i < buffer.size; i++) {
                mcx_config.vol[i] = static_cast<float*>(buffer.ptr)[i];
            }
        }
    } else if (py::array_t<double>::check_(volume_handle)) {
        auto f_style_volume = py::array_t<double, py::array::f_style>::ensure(volume_handle);
        mcx_config.mediabyte = 4;
        auto buffer = f_style_volume.request();

        if (buffer.shape.size() == 4) {
            throw py::value_error("Invalid volume dims for double volume.");
        }

        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                          static_cast<unsigned int>(buffer.shape.at(1)),
                          static_cast<unsigned int>(buffer.shape.at(2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
        mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));

        for (int i = 0; i < buffer.size; i++) {
            mcx_config.vol[i] = static_cast<double*>(buffer.ptr)[i];
        }
    } else {
        throw py::type_error("Invalid data type for vol array.");
    }
}

void parse_config(const py::dict& user_cfg, Config& mcx_config) {
    mcx_initcfg(&mcx_config);

    mcx_config.flog = stdout;
    GET_SCALAR_FIELD(user_cfg, mcx_config, nphoton, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, nblocksize, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, nthread, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, tstart, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, tstep, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, tend, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, maxdetphoton, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, sradius, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, maxgate, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, respin, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, isreflect, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, isref3, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, isrefint, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, isnormalized, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, isref3, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issrcfrom0, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, autopilot, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, minenergy, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, unitinmm, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, printnum, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, voidtime, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issaveref, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issaveexit, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, ismomentum, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, isspecular, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, replaydet, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, faststep, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, maxvoidstep, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, maxjumpdebug, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, gscatter, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, srcnum, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, omega, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, lambda, py::float_);
    GET_VEC3_FIELD(user_cfg, mcx_config, srcpos, float);
    GET_VEC34_FIELD(user_cfg, mcx_config, srcdir, float);
    GET_VEC3_FIELD(user_cfg, mcx_config, steps, float);
    GET_VEC3_FIELD(user_cfg, mcx_config, crop0, uint);
    GET_VEC3_FIELD(user_cfg, mcx_config, crop1, uint);
    GET_VEC4_FIELD(user_cfg, mcx_config, srcparam1, float);
    GET_VEC4_FIELD(user_cfg, mcx_config, srcparam2, float);
    GET_VEC4_FIELD(user_cfg, mcx_config, srciquv, float);
    parseVolume(user_cfg, mcx_config);

    if (user_cfg.contains("detpos")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["detpos"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid detpos field value");
        }

        auto buffer_info = f_style_volume.request();

        if (buffer_info.shape.at(0) > 0 && buffer_info.shape.at(1) != 4) {
            throw py::value_error("the 'detpos' field must have 4 columns (x,y,z,radius)");
        }

        mcx_config.detnum = buffer_info.shape.at(0);

        if (mcx_config.detpos) {
            free(mcx_config.detpos);
        }

        mcx_config.detpos = (float4*) malloc(mcx_config.detnum * sizeof(float4));
        auto val = static_cast<float*>(buffer_info.ptr);

        for (int j = 0; j < 4; j++)
            for (int i = 0; i < mcx_config.detnum; i++) {
                ((float*) (&mcx_config.detpos[i]))[j] = val[j * mcx_config.detnum + i];
            }
    }

    if (user_cfg.contains("prop")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["prop"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid prop field format");
        }

        auto buffer_info = f_style_volume.request();

        if (buffer_info.shape.at(0) > 0 && buffer_info.shape.at(1) != 4) {
            throw py::value_error("the 'prop' field must have 4 columns (mua,mus,g,n)");
        }

        mcx_config.medianum = buffer_info.shape.at(0);

        if (mcx_config.prop) {
            free(mcx_config.prop);
        }

        mcx_config.prop = (Medium*) malloc(mcx_config.medianum * sizeof(Medium));
        auto val = static_cast<float*>(buffer_info.ptr);

        for (int j = 0; j < 4; j++)
            for (int i = 0; i < mcx_config.medianum; i++) {
                ((float*) (&mcx_config.prop[i]))[j] = val[j * mcx_config.medianum + i];
            }
    }

    if (user_cfg.contains("polprop")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["polprop"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid polprop field value");
        }

        auto buffer_info = f_style_volume.request();

        if (buffer_info.shape.size() != 2) {
            throw py::value_error("the 'polprop' field must a 2D array");
        }

        if (buffer_info.shape.at(0) > 0 && buffer_info.shape.at(1) != 5) {
            throw py::value_error("the 'polprop' field must have 5 columns (mua, radius, rho, n_sph,n_bkg)");
        }

        mcx_config.polmedianum = buffer_info.shape.at(0);

        if (mcx_config.polprop) {
            free(mcx_config.polprop);
        }

        mcx_config.polprop = (POLMedium*) malloc(mcx_config.polmedianum * sizeof(POLMedium));
        auto val = static_cast<float*>(buffer_info.ptr);

        for (int j = 0; j < 5; j++)
            for (int i = 0; i < mcx_config.polmedianum; i++) {
                ((float*) (&mcx_config.polprop[i]))[j] = val[j * mcx_config.polmedianum + i];
            }
    }

    if (user_cfg.contains("session")) {
        std::string session = py::str(user_cfg["session"]);

        if (session.empty()) {
            throw py::value_error("the 'session' field must be a non-empty string");
        }

        if (session.size() > MAX_SESSION_LENGTH) {
            throw py::value_error("the 'session' field is too long");
        }

        strncpy(mcx_config.session, session.c_str(), MAX_SESSION_LENGTH);
    }

    if (user_cfg.contains("srctype")) {
        std::string src_type = py::str(user_cfg["srctype"]);
        const char* srctypeid[] = {"pencil", "isotropic", "cone", "gaussian", "planar",
                                   "pattern", "fourier", "arcsine", "disk", "fourierx", "fourierx2d", "zgaussian",
                                   "line", "slit", "pencilarray", "pattern3d", "hyperboloid", ""
                                  };
        char strtypestr[MAX_SESSION_LENGTH] = {'\0'};

        if (src_type.empty()) {
            throw py::value_error("the 'srctype' field must be a non-empty string");
        }

        if (src_type.size() > MAX_SESSION_LENGTH) {
            throw py::value_error("the 'srctype' field is too long");
        }

        strncpy(strtypestr, src_type.c_str(), MAX_SESSION_LENGTH);
        mcx_config.srctype = mcx_keylookup(strtypestr, srctypeid);

        if (mcx_config.srctype == -1) {
            throw py::value_error("the specified source type is not supported");
        }
    }

    if (user_cfg.contains("outputtype")) {
        std::string output_type_str = py::str(user_cfg["outputtype"]);
        const char* outputtype[] = {"flux", "fluence", "energy", "jacobian", "nscat", "wl", "wp", "wm", "rf", ""};
        char outputstr[MAX_SESSION_LENGTH] = {'\0'};

        if (output_type_str.empty()) {
            throw py::value_error("the 'srctype' field must be a non-empty string");
        }

        if (output_type_str.size() > MAX_SESSION_LENGTH) {
            throw py::value_error("the 'srctype' field is too long");
        }

        strncpy(outputstr, output_type_str.c_str(), MAX_SESSION_LENGTH);
        mcx_config.outputtype = mcx_keylookup(outputstr, outputtype);

        if (mcx_config.outputtype >= 5) { // map wl to jacobian, wp to nscat
            mcx_config.outputtype -= 2;
        }

        if (mcx_config.outputtype == -1) {
            throw py::value_error("the specified output type is not supported");
        }
    }

    if (user_cfg.contains("debuglevel")) {
        std::string debug_level = py::str(user_cfg["debuglevel"]);
        const char debugflag[] = {'R', 'M', 'P', '\0'};
        char debuglevel[MAX_SESSION_LENGTH] = {'\0'};

        if (debug_level.empty()) {
            throw py::value_error("the 'debuglevel' field must be a non-empty string");
        }

        if (debug_level.size() > MAX_SESSION_LENGTH) {
            throw py::value_error("the 'debuglevel' field is too long");
        }

        strncpy(debuglevel, debug_level.c_str(), MAX_SESSION_LENGTH);
        mcx_config.debuglevel = mcx_parsedebugopt(debuglevel, debugflag);

        if (mcx_config.debuglevel == 0) {
            throw py::value_error("the specified debuglevel is not supported");
        }
    }

    if (user_cfg.contains("savedetflag")) {
        std::string save_det_flag = py::str(user_cfg["savedetflag"]);
        const char saveflag[] = {'D', 'S', 'P', 'M', 'X', 'V', 'W', 'I', '\0'};
        char savedetflag[MAX_SESSION_LENGTH] = {'\0'};

        if (save_det_flag.empty()) {
            throw py::value_error("the 'savedetflag' field must be a non-empty string");
        }

        if (save_det_flag.size() > MAX_SESSION_LENGTH) {
            throw py::value_error("the 'savedetflag' field is too long");
        }

        strncpy(savedetflag, save_det_flag.c_str(), MAX_SESSION_LENGTH);
        mcx_config.savedetflag = mcx_parsedebugopt(savedetflag, saveflag);
    }

    if (user_cfg.contains("srcpattern")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["srcpattern"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid srcpattern field value");
        }

        auto buffer_info = f_style_volume.request();

        if (mcx_config.srcpattern) {
            free(mcx_config.srcpattern);
        }

        mcx_config.srcpattern = (float*) malloc(buffer_info.size * sizeof(float));
        auto val = static_cast<float*>(buffer_info.ptr);

        for (int i = 0; i < buffer_info.size; i++) {
            mcx_config.srcpattern[i] = val[i];
        }
    }

    if (user_cfg.contains("invcdf")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["invcdf"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid invcdf field value");
        }

        auto buffer_info = f_style_volume.request();
        unsigned int nphase = buffer_info.shape.size();
        float* val = static_cast<float*>(buffer_info.ptr);
        mcx_config.nphase = nphase + 2;
        mcx_config.nphase += (mcx_config.nphase & 0x1); // make cfg.nphase even number
        mcx_config.invcdf = (float*) calloc(mcx_config.nphase, sizeof(float));

        for (int i = 0; i < nphase; i++) {
            mcx_config.invcdf[i + 1] = val[i];

            if (i > 0 && (val[i] < val[i - 1] || (val[i] > 1.f || val[i] < -1.f)))
                throw py::value_error(
                    "cfg.invcdf contains invalid data; it must be a monotonically increasing vector with all values between -1 and 1");
        }

        mcx_config.invcdf[0] = -1.f;
        mcx_config.invcdf[nphase + 1] = 1.f;
        mcx_config.invcdf[mcx_config.nphase - 1] = 1.f;
    }

    if (user_cfg.contains("shapes")) {
        std::string shapes_string = py::str(user_cfg["shapes"]);

        if (shapes_string.empty()) {
            throw py::value_error("the 'shapes' field must be a non-empty string");
        }

        mcx_config.shapedata = (char*) calloc(shapes_string.size() + 2, 1);
        strncpy(mcx_config.shapedata, shapes_string.c_str(), shapes_string.size() + 1);
    }

    if (user_cfg.contains("bc")) {
        std::string bc_string = py::str(user_cfg["bc"]);

        if (bc_string.empty() || bc_string.size() > 12) {
            throw py::value_error("the 'bc' field must be a non-empty string / have less than 12 characters.");
        }

        strncpy(mcx_config.bc, bc_string.c_str(), bc_string.size() + 1);
        mcx_config.bc[bc_string.size()] = '\0';
    }

    if (user_cfg.contains("detphotons")) {
        auto detphotons = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["detphotons"]);

        if (!detphotons) {
            throw py::value_error("Invalid detphotons field value");
        }

        auto buffer_info = detphotons.request();

        det_ps = static_cast<float*>(buffer_info.ptr);
        dim_det_ps[0] = buffer_info.shape.at(0);
        dim_det_ps[1] = buffer_info.shape.at(1);
    }

    if (user_cfg.contains("seed")) {
        auto seed_value = user_cfg["seed"];

        // If the seed value is scalar (int or float), then assign it directly
        if (py::int_::check_(seed_value)) {
            mcx_config.seed = py::int_(seed_value);
        } else if (py::float_::check_(seed_value)) {
            mcx_config.seed = py::float_(seed_value).cast<int>();
        }
        // Set seed from array
        else {
            auto f_style_array = py::array_t < uint8_t, py::array::f_style | py::array::forcecast >::ensure(seed_value);

            if (!f_style_array) {
                throw py::value_error("Invalid seed field value");
            }

            auto buffer_info = f_style_array.request();
            seed_byte = buffer_info.shape.at(0);

            if (buffer_info.shape.at(0) != sizeof(float) * RAND_WORD_LEN) {
                throw py::value_error("the row number of cfg.seed does not match RNG seed byte-length");
            }

            mcx_config.replay.seed = malloc(buffer_info.size);
            memcpy(mcx_config.replay.seed, buffer_info.ptr, buffer_info.size);
            mcx_config.seed = SEED_FROM_FILE;
            mcx_config.nphoton = buffer_info.shape.at(1);
        }
    }

    if (user_cfg.contains("gpuid")) {
        auto gpu_id_value = user_cfg["gpuid"];

        if (py::int_::check_(gpu_id_value)) {
            mcx_config.gpuid = py::int_(gpu_id_value);
            memset(mcx_config.deviceid, 0, MAX_DEVICE);

            if (mcx_config.gpuid > 0 && mcx_config.gpuid < MAX_DEVICE) {
                memset(mcx_config.deviceid, '0', mcx_config.gpuid - 1);
                mcx_config.deviceid[mcx_config.gpuid - 1] = '1';
            } else {
                throw py::value_error("GPU id must be positive and can not be more than 256");
            }
        } else if (py::str::check_(gpu_id_value)) {
            std::string gpu_id_string_value = py::str(gpu_id_value);

            if (gpu_id_string_value.empty()) {
                throw py::value_error("the 'gpuid' field must be an integer or non-empty string");
            }

            if (gpu_id_string_value.size() > MAX_DEVICE) {
                throw py::value_error("the 'gpuid' field is too long");
            }

            strncpy(mcx_config.deviceid, gpu_id_string_value.c_str(), MAX_DEVICE);
        }

        for (int i = 0; i < MAX_DEVICE; i++)
            if (mcx_config.deviceid[i] == '0') {
                mcx_config.deviceid[i] = '\0';
            }
    }

    if (user_cfg.contains("workload")) {
        auto workload_value = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["workload"]);

        if (!workload_value) {
            throw py::value_error("Invalid workload field value");
        }

        auto buffer_info = workload_value.request();

        if (buffer_info.shape.size() < 2 && buffer_info.size > MAX_DEVICE) {
            throw py::value_error("the workload list can not be longer than 256");
        }

        for (int i = 0; i < buffer_info.size; i++) {
            mcx_config.workload[i] = static_cast<float*>(buffer_info.ptr)[i];
        }
    }

    // Output arguments parsing
    GET_SCALAR_FIELD(user_cfg, mcx_config, issave2pt, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issavedet, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issaveseed, py::bool_);

    // Flush the std::cout and std::cerr
    std::cout.flush();
    std::cerr.flush();
}

/**
 * Function that's called to cleanup any memory/configs allocated by PMCX. It is used in both normal and exceptional
 * termination of the application
 * @param gpu_info reference to an array of MCXGPUInfo data structure
 * @param mcx_config reference to MCXConfig data structure
 */
inline void cleanup_configs(MCXGPUInfo*& gpu_info, MCXConfig& mcx_config) {
    mcx_cleargpuinfo(&gpu_info);
    mcx_clearcfg(&mcx_config);
}


py::dict pmcx_interface(const py::dict& user_cfg) {
    unsigned int partial_data, hostdetreclen;
    Config mcx_config;  /* mcx_config: structure to store all simulation parameters */
    GPUInfo* gpu_info = nullptr;        /** gpuInfo: structure to store GPU information */
    unsigned int active_dev = 0;     /** activeDev: count of total active GPUs to be used */
    int error_flag = 0;
    std::vector<std::string> exception_msgs;
    int thread_id = 0;
    size_t field_dim[6];
    py::dict output;

    try {
        /*
         * To start an MCX simulation, we first create a simulation configuration and set all elements to its default settings.
         */
        det_ps = nullptr;

        parse_config(user_cfg, mcx_config);

        /** The next step, we identify gpu number and query all GPU info */
        if (!(active_dev = mcx_list_gpu(&mcx_config, &gpu_info))) {
            mcx_error(-1, "No GPU device found\n", __FILE__, __LINE__);
        }

        mcx_flush(&mcx_config);

        /** Validate all input fields, and warn incompatible inputs */
        mcx_validatecfg(&mcx_config, det_ps, dim_det_ps, seed_byte);

        partial_data =
            (mcx_config.medianum - 1) * (SAVE_NSCAT(mcx_config.savedetflag) + SAVE_PPATH(mcx_config.savedetflag) +
                                         SAVE_MOM(mcx_config.savedetflag));
        hostdetreclen = partial_data + SAVE_DETID(mcx_config.savedetflag) + 3 * (SAVE_PEXIT(mcx_config.savedetflag) +
                        SAVE_VEXIT(mcx_config.savedetflag)) + SAVE_W0(mcx_config.savedetflag) + 4 * SAVE_IQUV(mcx_config.savedetflag);

        /** One must define the domain and properties */
        if (mcx_config.vol == nullptr || mcx_config.medianum == 0) {
            throw py::value_error("You must define 'vol' and 'prop' field.");
        }

        /** Initialize all buffers necessary to store the output variables */
        if (mcx_config.issave2pt == 1) {
            int field_len =
                static_cast<int>(mcx_config.dim.x) * static_cast<int>(mcx_config.dim.y) * static_cast<int>(mcx_config.dim.z) *
                (int) ((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5) * mcx_config.srcnum;

            if (mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1) {
                field_len *= mcx_config.detnum;
            }

            if (mcx_config.replay.seed != nullptr && mcx_config.outputtype == otRF) {
                field_len *= 2;
            }

            mcx_config.exportfield = (float*) calloc(field_len, sizeof(float));
        }

        if (mcx_config.issavedet == 1) {
            mcx_config.exportdetected = (float*) malloc(hostdetreclen * mcx_config.maxdetphoton * sizeof(float));
        }

        if (mcx_config.issaveseed == 1) {
            mcx_config.seeddata = malloc(mcx_config.maxdetphoton * sizeof(float) * RAND_WORD_LEN);
        }

        if (mcx_config.debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
            mcx_config.exportdebugdata = (float*) malloc(mcx_config.maxjumpdebug * sizeof(float) * MCX_DEBUG_REC_LEN);
        }

        /** Start multiple threads, one thread to run portion of the simulation on one CUDA GPU, all in parallel */
#ifdef _OPENMP
        omp_set_num_threads(active_dev);
        #pragma omp parallel shared(exception_msgs)
        {
            thread_id = omp_get_thread_num();
#endif

            /** Enclose all simulation calls inside a try/catch construct for exception handling */
            try {
                /** Call the main simulation host function to start the simulation */
                mcx_run_simulation(&mcx_config, gpu_info);

            } catch (const char* err) {
                exception_msgs.push_back("Error from thread (" + std::to_string(thread_id) + "): " + err);
            } catch (const std::exception& err) {
                exception_msgs.push_back("C++ Error from thread (" + std::to_string(thread_id) + "): " + err.what());
            } catch (...) {
                exception_msgs.push_back("Unknown Exception from thread (" + std::to_string(thread_id) + ")");
            }

#ifdef _OPENMP
        }
#endif

        /** If error is detected, gracefully terminate the mex and return back to Python */
        if (!exception_msgs.empty()) {
            throw py::runtime_error("PMCX terminated due to an exception!");
        }

        field_dim[4] = 1;
        field_dim[5] = 1;

        if (mcx_config.debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
            field_dim[0] = MCX_DEBUG_REC_LEN;
            field_dim[1] = mcx_config.debugdatalen; // his.savedphoton is for one repetition, should correct
            field_dim[2] = 0;
            field_dim[3] = 0;
            auto photon_traj_data = py::array_t<float, py::array::f_style>({field_dim[0], field_dim[1]});

            if (mcx_config.debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
                memcpy(photon_traj_data.mutable_data(), mcx_config.exportdebugdata, field_dim[0] * field_dim[1] * sizeof(float));
            }

            if (mcx_config.exportdebugdata) {
                free(mcx_config.exportdebugdata);
            }

            mcx_config.exportdebugdata = nullptr;
            output["traj"] = photon_traj_data;
        }

        if (mcx_config.issaveseed == 1) {
            field_dim[0] = (mcx_config.issaveseed > 0) * RAND_WORD_LEN * sizeof(float);
            field_dim[1] = mcx_config.detectedcount; // his.savedphoton is for one repetition, should correct
            field_dim[2] = 0;
            field_dim[3] = 0;
            auto detected_seeds = py::array_t<uint8_t, py::array::f_style>({field_dim[0], field_dim[1]});
            memcpy(detected_seeds.mutable_data(), mcx_config.seeddata, field_dim[0] * field_dim[1]);
            free(mcx_config.seeddata);
            mcx_config.seeddata = nullptr;
            output["seeds"] = detected_seeds;
        }

        if (user_cfg.contains("dumpmask") && py::bool_(user_cfg["dumpmask"]).cast<bool>()) {
            field_dim[0] = mcx_config.dim.x;
            field_dim[1] = mcx_config.dim.y;
            field_dim[2] = mcx_config.dim.z;
            field_dim[3] = 0;

            if (mcx_config.vol) {
                auto detector_vol = py::array_t<uint32_t, py::array::f_style>({field_dim[0], field_dim[1], field_dim[2]});
                memcpy(detector_vol.mutable_data(), mcx_config.vol,
                       field_dim[0] * field_dim[1] * field_dim[2] * sizeof(unsigned int));
                output["vol"] = detector_vol;
            }
        }

        if (mcx_config.issavedet == 1) {
            field_dim[0] = hostdetreclen;
            field_dim[1] = mcx_config.detectedcount;
            field_dim[2] = 0;
            field_dim[3] = 0;

            if (mcx_config.detectedcount > 0) {
                auto partial_path = py::array_t<float, py::array::f_style>(std::initializer_list<size_t>({field_dim[0], mcx_config.detectedcount}));
                memcpy(partial_path.mutable_data(), mcx_config.exportdetected,
                       field_dim[0] * field_dim[1] * sizeof(float));
                output["detp"] = partial_path;
            }

            free(mcx_config.exportdetected);
            mcx_config.exportdetected = NULL;
        }

        if (mcx_config.issave2pt) {
            int field_len;
            field_dim[0] = mcx_config.srcnum * mcx_config.dim.x;
            field_dim[1] = mcx_config.dim.y;
            field_dim[2] = mcx_config.dim.z;
            field_dim[3] = (int) ((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5);

            if (mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1) {
                field_dim[4] = mcx_config.detnum;
            }

            if (mcx_config.replay.seed != nullptr && mcx_config.outputtype == otRF) {
                field_dim[5] = 2;
            }

            field_len = field_dim[0] * field_dim[1] * field_dim[2] * field_dim[3] * field_dim[4] * field_dim[5];
            std::vector<size_t> array_dims;

            if (field_dim[5] > 1)
                array_dims = {field_dim[0], field_dim[1], field_dim[2], field_dim[3], field_dim[4], field_dim[5]};
            else if (field_dim[4] > 1)
                array_dims = {field_dim[0], field_dim[1], field_dim[2], field_dim[3], field_dim[4]};
            else
                array_dims = {field_dim[0], field_dim[1], field_dim[2], field_dim[3]};

            auto dref_array = py::array_t<float, py::array::f_style>(array_dims);

            if (mcx_config.issaveref) {
                auto* dref = static_cast<float*>(dref_array.mutable_data());
                memcpy(dref, mcx_config.exportfield, field_len * sizeof(float));

                for (int i = 0; i < field_len; i++) {
                    if (dref[i] < 0.f) {
                        dref[i] = -dref[i];
                        mcx_config.exportfield[i] = 0.f;
                    } else {
                        dref[i] = 0.f;
                    }
                }

                output["dref"] = dref_array;
            }

            auto data = py::array_t<float, py::array::f_style>(array_dims);
            memcpy(data.mutable_data(), mcx_config.exportfield, field_len * sizeof(float));
            output["flux"] = data;
            free(mcx_config.exportfield);
            mcx_config.exportfield = nullptr;
            // Stat dictionary output
            auto stat_dict = py::dict();
            stat_dict["runtime"] = mcx_config.runtime;
            stat_dict["nphoton"] = mcx_config.nphoton * ((mcx_config.respin > 1) ? (mcx_config.respin) : 1);
            stat_dict["energytot"] = mcx_config.energytot;
            stat_dict["energyabs"] = mcx_config.energyabs;
            stat_dict["normalizer"] = mcx_config.normalizer;
            stat_dict["unitinmm"] = mcx_config.unitinmm;
            py::list workload;

            for (int i = 0; i < active_dev; i++) {
                workload.append(mcx_config.workload[i]);
            }

            stat_dict["workload"] = workload;
            output["stat"] = stat_dict;

            /** return the final optical properties for polarized MCX simulation */
            if (mcx_config.polprop) {
                for (int i = 0; i < mcx_config.polmedianum; i++) {
                    // restore mua and mus values
                    mcx_config.prop[i + 1].mua /= mcx_config.unitinmm;
                    mcx_config.prop[i + 1].mus /= mcx_config.unitinmm;
                }

                auto opt_properties = py::array_t<float, py::array::f_style>({4, int(mcx_config.medianum)});
                memcpy(opt_properties.mutable_data(), mcx_config.prop, mcx_config.medianum * 4 * sizeof(float));
                output["prop"] = opt_properties;
            }
        }
    } catch (const char* err) {
        cleanup_configs(gpu_info, mcx_config);
        throw py::runtime_error(err);
    } catch (const py::type_error& err) {
        cleanup_configs(gpu_info, mcx_config);
        throw err;
    } catch (const py::value_error& err) {
        cleanup_configs(gpu_info, mcx_config);
        throw err;
    } catch (const py::runtime_error& err) {
        cleanup_configs(gpu_info, mcx_config);
        std::string error_msg = err.what();

        for (const auto& m : exception_msgs) {
            error_msg += (m + "\n");
        }

        throw py::runtime_error(error_msg);
    } catch (const std::exception& err) {
        cleanup_configs(gpu_info, mcx_config);
        throw py::runtime_error(std::string("C++ Error: ") + err.what());
    } catch (...) {
        cleanup_configs(gpu_info, mcx_config);
        throw py::runtime_error("Unknown exception occurred");
    }

    /** Clear up simulation data structures by calling the destructors */
    cleanup_configs(gpu_info, mcx_config);
    // return the MCX output dictionary
    return output;
}


/**
 * @brief Error reporting function in PMCX, equivalent to mcx_error in binary mode
 *
 * @param[in] id: a single integer for the types of the error
 * @param[in] msg: the error message string
 * @param[in] filename: the unit file name where this error is raised
 * @param[in] linenum: the line number in the file where this error is raised
 */

int mcx_throw_exception(const int id, const char* msg, const char* filename, const int linenum) {
    throw msg;
    return id;
}

void print_mcx_usage() {
    std::cout
            << "PMCX v2022.10\nUsage:\n    output = pmcx.run(cfg);\n\nRun 'help(pmcx.run)' for more details.\n";
}

/**
 * @brief Force matlab refresh the command window to print all buffered messages
 */

extern "C" void mcx_python_flush() {
    std::cout.flush();
}

py::dict pmcx_interface_wargs(py::args args, const py::kwargs& kwargs) {
    if (py::len(kwargs) == 0) {
        print_mcx_usage();
        return {};
    }

    return pmcx_interface(kwargs);
}

py::list get_GPU_info() {
    Config mcx_config;            /** mcxconfig: structure to store all simulation parameters */
    GPUInfo* gpu_info = nullptr;        /** gpuinfo: structure to store GPU information */
    mcx_initcfg(&mcx_config);
    mcx_config.isgpuinfo = 3;
    py::list output;

    if (!(mcx_list_gpu(&mcx_config, &gpu_info))) {
        std::cerr << "No CUDA-capable device was found." << std::endl;
        return output;
    }

    for (int i = 0; i < gpu_info[0].devcount; i++) {
        py::dict current_device_info;
        current_device_info["name"] = gpu_info[i].name;
        current_device_info["id"] = gpu_info[i].id;
        current_device_info["devcount"] = gpu_info[i].devcount;
        current_device_info["major"] = gpu_info[i].major;
        current_device_info["minor"] = gpu_info[i].minor;
        current_device_info["globalmem"] = gpu_info[i].globalmem;
        current_device_info["constmem"] = gpu_info[i].constmem;
        current_device_info["sharedmem"] = gpu_info[i].sharedmem;
        current_device_info["regcount"] = gpu_info[i].regcount;
        current_device_info["clock"] = gpu_info[i].clock;
        current_device_info["sm"] = gpu_info[i].sm;
        current_device_info["core"] = gpu_info[i].core;
        current_device_info["autoblock"] = gpu_info[i].autoblock;
        current_device_info["autothread"] = gpu_info[i].autothread;
        current_device_info["maxgate"] = gpu_info[i].maxgate;
        output.append(current_device_info);
    }

    mcx_cleargpuinfo(&gpu_info);
    mcx_clearcfg(&mcx_config);
    return output;
}

PYBIND11_MODULE(_pmcx, m) {
    m.doc() = "PMCX: Python bindings for Monte Carlo eXtreme photon transport simulator, http://mcx.space";
    m.def("run", &pmcx_interface, "Runs MCX with the given config.", py::call_guard<py::scoped_ostream_redirect,
          py::scoped_estream_redirect>());
    m.def("run", &pmcx_interface_wargs, "Runs MCX with the given config.", py::call_guard<py::scoped_ostream_redirect,
          py::scoped_estream_redirect>());
    m.def("gpuinfo",
          &get_GPU_info,
          "Prints out the list of CUDA-capable devices attached to this system.",
          py::call_guard<py::scoped_ostream_redirect,
          py::scoped_estream_redirect>());
}


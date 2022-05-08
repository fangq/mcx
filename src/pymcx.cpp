/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2021
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
\file    pymcx.cpp

@brief   Python interface using Pybind11 for MCX
*******************************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_const.h"
#include "mcx_shapes.h"
#include <pybind11/iostream.h>
#include "interface-common.h"

// Python binding for runtime_error exception in Python.
namespace pybind11 {
PYBIND11_RUNTIME_EXCEPTION(runtime_error, PyExc_RuntimeError);
}

namespace py = pybind11;

#if defined(USE_XOROSHIRO128P_RAND)
#define RAND_WORD_LEN 4
#elif defined(USE_LL5_RAND)
#define RAND_WORD_LEN 5
#elif defined(USE_POSIX_RAND)
#define RAND_WORD_LEN 4
#else
#define RAND_WORD_LEN 4       /**< number of Words per RNG state */
#endif

float *det_ps = nullptr;     //! buffer to receive data from cfg.detphotons field
int dim_det_ps[2] = {0, 0};  //! dimensions of the cfg.detphotons array
int seed_byte = 0;

/**
 * Macro to find and extract a scalar property from a source Python dictionary configuration and assign it in a destination
 * MCX Config. The scalar is cast to the python type before assignment.
 */
#define GET_SCALAR_FIELD(src_pydict, dst_mcx_config, property, py_type) if ((src_pydict).contains(#property))\
                                                                        {(dst_mcx_config).property = py::reinterpret_borrow<py_type>((src_pydict)[#property]);\
                                                                        std::cout << #property << ": " << (float) (dst_mcx_config).property << std::endl;}

#define GET_VEC3_FIELD(src, dst, prop, type) if (src.contains(#prop)) {auto list = py::reinterpret_borrow<py::list>(src[#prop]);\
                                             dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>()};\
                                             std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << "]\n";}

#define GET_VEC4_FIELD(src, dst, prop, type) if (src.contains(#prop)) {auto list = py::reinterpret_borrow<py::list>(src[#prop]);\
                                             dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list[3].cast<type>()}; \
                                             std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << ", " << dst.prop.w << "]\n";}

#define GET_VEC34_FIELD(src, dst, prop, type) if (src.contains(#prop)) {auto list = py::reinterpret_borrow<py::list>(src[#prop]);\
                                             dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list.size() == 4 ? list[3].cast<type>() : 1}; \
                                             std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z;\
                                             if (list.size() == 4) std::cout << ", " << dst.prop.w; std::cout << "]\n";}

/**
 * Determines the type of volume passed to the interface and decides how to copy it to MCXConfig.
 * @param userCfg
 * @param mcxConfig
 */
void parseVolume(const py::dict &userCfg, Config &mcxConfig) {
  if (!userCfg.contains("vol"))
    throw py::value_error("Configuration must specify a 2/3/4D volume.");
  auto volumeHandle = userCfg["vol"];
  // Free the volume
  if (mcxConfig.vol) free(mcxConfig.vol);
  unsigned int dim_xyz = 0;
  // Data type-specific logic
  if (py::array_t<int8_t, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<int8_t, py::array::f_style>::ensure(volumeHandle);
    auto buffer = fStyleVolume.request();
    int i = buffer.shape.size() == 4;
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                     static_cast<unsigned int>(buffer.shape.at(i + 1)),
                     static_cast<unsigned int>(buffer.shape.at(i + 2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));
    if (i == 1) {
      if (buffer.shape.at(0) == 4) {
        mcxConfig.mediabyte = MEDIA_ASGN_BYTE;
        memcpy(mcxConfig.vol, buffer.ptr, dim_xyz * sizeof(unsigned int));
      } else if (buffer.shape.at(0) == 8) {
        mcxConfig.mediabyte = MEDIA_2LABEL_SPLIT;
        auto val = (unsigned char *) buffer.ptr;
        if (mcxConfig.vol) free(mcxConfig.vol);
        mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz << 3));
        memcpy(mcxConfig.vol, val, (dim_xyz << 3));
      }
    } else {
      mcxConfig.mediabyte = 1;
      for (i = 0; i < buffer.size; i++)
        mcxConfig.vol[i] = static_cast<unsigned char *>(buffer.ptr)[i];
    }
  }
  else if (py::array_t<int16_t, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<int16_t, py::array::f_style>::ensure(volumeHandle);
    auto buffer = fStyleVolume.request();
    int i = buffer.shape.size() == 4;
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                     static_cast<unsigned int>(buffer.shape.at(i + 1)),
                     static_cast<unsigned int>(buffer.shape.at(i + 2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));
    if (i == 1) {
      if (buffer.shape.at(0) == 3) {
        mcxConfig.mediabyte = MEDIA_2LABEL_MIX;
        auto *val = (unsigned short *) buffer.ptr;
        union {
          unsigned short h[2];
          unsigned char c[4];
          unsigned int i[1];
        } f2bh;
        for (i = 0; i < dim_xyz; i++) {
          f2bh.c[0] = val[i * 3] & 0xFF;
          f2bh.c[1] = val[i * 3 + 1] & 0xFF;
          f2bh.h[1] = val[i * 3 + 2] & 0x7FFF;
          mcxConfig.vol[i] = f2bh.i[0];
        }
      } else if (buffer.shape.at(0) == 2) {
        mcxConfig.mediabyte = MEDIA_AS_SHORT;
        memcpy(mcxConfig.vol, buffer.ptr, dim_xyz * sizeof(unsigned int));
      }
    } else {
      mcxConfig.mediabyte = 2;
      mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
      for (i = 0; i < buffer.size; i++)
        mcxConfig.vol[i] = static_cast<unsigned short *>(buffer.ptr)[i];
    }
  }
  else if (py::array_t<int32_t, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<int32_t, py::array::f_style>::ensure(volumeHandle);
    mcxConfig.mediabyte = 4;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for int32_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));
    memcpy(mcxConfig.vol, buffer.ptr, buffer.size * sizeof(unsigned int));
  }
  else if (py::array_t<u_int8_t, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<u_int8_t, py::array::f_style>::ensure(volumeHandle);
    mcxConfig.mediabyte = 1;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for uint8_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));
    for (int i = 0; i < buffer.size; i++)
      mcxConfig.vol[i] = static_cast<unsigned char *>(buffer.ptr)[i];
  }
  else if (py::array_t<u_int16_t, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<u_int16_t, py::array::f_style>::ensure(volumeHandle);
    mcxConfig.mediabyte = 2;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for u_int16_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));
    for (int i = 0; i < buffer.size; i++)
      mcxConfig.vol[i] = static_cast<unsigned short *>(buffer.ptr)[i];
  }
  else if (py::array_t<u_int32_t, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<u_int32_t, py::array::f_style>::ensure(volumeHandle);
    mcxConfig.mediabyte = 8;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for u_int32_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));
    memcpy(mcxConfig.vol, buffer.ptr, buffer.size * sizeof(unsigned int));
  }
  else if (py::array_t<float, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<float, py::array::f_style>::ensure(volumeHandle);
    auto buffer = fStyleVolume.request();
    int i = buffer.shape.size() == 4;
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                     static_cast<unsigned int>(buffer.shape.at(i + 1)),
                     static_cast<unsigned int>(buffer.shape.at(i + 2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));

    if (i) {
      switch (buffer.shape.at(0)) {
        case 3: {
          mcxConfig.mediabyte = MEDIA_LABEL_HALF;
          auto val = (float *) buffer.ptr;
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

            if (f2bh.f[1] < 0.f || f2bh.f[1] >= 4.f || f2bh.f[0] < 0.f)
              throw py::value_error("the 2nd volume must have an integer value between 0 and 3");

            f2bh.h[0] = ((((unsigned char) (f2bh.f[1]) & 0x3) << 14) | (unsigned short) (f2bh.f[0]));

            f2bh.h[1] = (f2bh.i[2] >> 31) << 5;
            tmp = (f2bh.i[2] >> 23) & 0xff;
            tmp = (tmp - 0x70) & ((unsigned int) ((int) (0x70 - tmp) >> 4) >> 27);
            f2bh.h[1] = (f2bh.h[1] | tmp) << 10;
            f2bh.h[1] |= (f2bh.i[2] >> 13) & 0x3ff;

            mcxConfig.vol[i] = f2bh.i[0];
          }
          break;
        }
        case 2: {
          mcxConfig.mediabyte = MEDIA_AS_F2H;
          auto val = (float *) buffer.ptr;
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
              mcxConfig.vol[i] = 0;
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

            if (f2h.i[0] == 0) /*avoid being detected as a 0-label voxel, setting mus=EPS_fp16*/
              f2h.i[0] = 0x00010000;

            mcxConfig.vol[i] = f2h.i[0];
          }
          break;
        }
        case 1: {
          mcxConfig.mediabyte = MEDIA_MUA_FLOAT;
          union {
            float f;
            uint i;
          } f2i;
          auto *val = (float *) buffer.ptr;
          for (i = 0; i < dim_xyz; i++) {
            f2i.f = val[i];
            if (f2i.i == 0) /*avoid being detected as a 0-label voxel*/
              f2i.f = EPS;
            if (val[i] != val[i]) /*if input is nan in continuous medium, convert to 0-voxel*/
              f2i.i = 0;
            mcxConfig.vol[i] = f2i.i;
          }
          break;
        }
        default:throw py::value_error("Invalid array for vol array.");
      }
    } else {
      mcxConfig.mediabyte = 14;
      mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
      for (i = 0; i < buffer.size; i++)
        mcxConfig.vol[i] = static_cast<float *>(buffer.ptr)[i];
    }
  }
  else if (py::array_t<double, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<double, py::array::f_style>::ensure(volumeHandle);
    mcxConfig.mediabyte = 4;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for double volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    dim_xyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
    mcxConfig.vol = static_cast<unsigned int *>(malloc(dim_xyz * sizeof(unsigned int)));
    for (int i = 0; i < buffer.size; i++)
      mcxConfig.vol[i] = static_cast<double *>(buffer.ptr)[i];
  }
  else
    throw py::type_error("Invalid data type for vol array.");
}

void parse_config(const py::dict &user_cfg, Config &mcx_config) {
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
  GET_SCALAR_FIELD(user_cfg, mcx_config, issaveseed, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, issaveref, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, issaveexit, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, ismomentum, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, isspecular, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, replaydet, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, faststep, py::int_);
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
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(user_cfg["detpos"]);
    auto bufferInfo = fStyleVolume.request();
    if (bufferInfo.shape.at(0) > 0 && bufferInfo.shape.at(1) != 4)
      throw py::value_error("the 'detpos' field must have 4 columns (x,y,z,radius)");
    mcx_config.detnum = bufferInfo.shape.at(0);
    if (mcx_config.detpos) free(mcx_config.detpos);
    mcx_config.detpos = (float4 *) malloc(mcx_config.detnum * sizeof(float4));
    auto val = static_cast<float *>(bufferInfo.ptr);
    for (int j = 0; j < 4; j++)
      for (int i = 0; i < mcx_config.detnum; i++)
        ((float *) (&mcx_config.detpos[i]))[j] = val[j * mcx_config.detnum + i];
  }
  if (user_cfg.contains("prop")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(user_cfg["prop"]);
    auto bufferInfo = fStyleVolume.request();
    if (bufferInfo.shape.at(0) > 0 && bufferInfo.shape.at(1) != 4)
      throw py::value_error("the 'prop' field must have 4 columns (mua,mus,g,n)");
    mcx_config.medianum = bufferInfo.shape.at(0);
    if (mcx_config.prop) free(mcx_config.prop);
    mcx_config.prop = (Medium *) malloc(mcx_config.medianum * sizeof(Medium));
    auto val = static_cast<float *>(bufferInfo.ptr);
    for (int j = 0; j < 4; j++)
      for (int i = 0; i < mcx_config.medianum; i++)
        ((float *) (&mcx_config.prop[i]))[j] = val[j * mcx_config.medianum + i];
  }
  if (user_cfg.contains("polprop")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(user_cfg["polprop"]);
    auto bufferInfo = fStyleVolume.request();
    if (bufferInfo.shape.size() != 2)
      throw py::value_error("the 'polprop' field must a 2D array");
    if (bufferInfo.shape.at(0) > 0 && bufferInfo.shape.at(1) != 5)
      throw py::value_error("the 'polprop' field must have 5 columns (mua, radius, rho, n_sph,n_bkg)");
    mcx_config.polmedianum = bufferInfo.shape.at(0);
    if (mcx_config.polprop) free(mcx_config.polprop);
    mcx_config.polprop = (POLMedium *) malloc(mcx_config.polmedianum * sizeof(POLMedium));
    auto val = static_cast<float *>(bufferInfo.ptr);
    for (int j = 0; j < 5; j++)
      for (int i = 0; i < mcx_config.polmedianum; i++)
        ((float *) (&mcx_config.polprop[i]))[j] = val[j * mcx_config.polmedianum + i];
  }
  if (user_cfg.contains("session")) {
    std::string session = py::str(user_cfg["session"]);
    if (session.empty())
      throw py::value_error("the 'session' field must be a non-empty string");
    if (session.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'session' field is too long");
    strncpy(mcx_config.session, session.c_str(), MAX_SESSION_LENGTH);
  }
  if (user_cfg.contains("srctype")) {
    std::string srcType = py::str(user_cfg["srctype"]);
    const char *srctypeid[] = {"pencil", "isotropic", "cone", "gaussian", "planar",
                               "pattern", "fourier", "arcsine", "disk", "fourierx", "fourierx2d", "zgaussian",
                               "line", "slit", "pencilarray", "pattern3d", "hyperboloid", ""};
    char strtypestr[MAX_SESSION_LENGTH] = {'\0'};

    if (srcType.empty())
      throw py::value_error("the 'srctype' field must be a non-empty string");
    if (srcType.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'srctype' field is too long");
    strncpy(strtypestr, srcType.c_str(), MAX_SESSION_LENGTH);
    mcx_config.srctype = mcx_keylookup(strtypestr, srctypeid);
    if (mcx_config.srctype == -1)
      throw py::value_error("the specified source type is not supported");
  }
  if (user_cfg.contains("outputtype")) {
    std::string outputType = py::str(user_cfg["outputtype"]);
    const char *outputtype[] = {"flux", "fluence", "energy", "jacobian", "nscat", "wl", "wp", "wm", "rf", ""};
    char outputstr[MAX_SESSION_LENGTH] = {'\0'};
    if (outputType.empty())
      throw py::value_error("the 'srctype' field must be a non-empty string");
    if (outputType.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'srctype' field is too long");
    strncpy(outputstr, outputType.c_str(), MAX_SESSION_LENGTH);
    mcx_config.outputtype = mcx_keylookup(outputstr, outputtype);
    if (mcx_config.outputtype >= 5) // map wl to jacobian, wp to nscat
      mcx_config.outputtype -= 2;
    if (mcx_config.outputtype == -1)
      throw py::value_error("the specified output type is not supported");
  }
  if (user_cfg.contains("debuglevel")) {
    std::string debugLevel = py::str(user_cfg["debuglevel"]);
    const char debugflag[] = {'R', 'M', 'P', '\0'};
    char debuglevel[MAX_SESSION_LENGTH] = {'\0'};
    if (debugLevel.empty())
      throw py::value_error("the 'debuglevel' field must be a non-empty string");
    if (debugLevel.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'debuglevel' field is too long");
    strncpy(debuglevel, debugLevel.c_str(), MAX_SESSION_LENGTH);
    mcx_config.debuglevel = mcx_parsedebugopt(debuglevel, debugflag);
    if (mcx_config.debuglevel == 0)
      throw py::value_error("the specified debuglevel is not supported");
  }
  if (user_cfg.contains("savedetflag")) {
    std::string saveDetFlag = py::str(user_cfg["savedetflag"]);
    const char saveflag[] = {'D', 'S', 'P', 'M', 'X', 'V', 'W', 'I', '\0'};
    char savedetflag[MAX_SESSION_LENGTH] = {'\0'};
    if (saveDetFlag.empty())
      throw py::value_error("the 'savedetflag' field must be a non-empty string");
    if (saveDetFlag.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'savedetflag' field is too long");
    strncpy(savedetflag, saveDetFlag.c_str(), MAX_SESSION_LENGTH);
    mcx_config.savedetflag = mcx_parsedebugopt(savedetflag, saveflag);
  }
  if (user_cfg.contains("srcpattern")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(user_cfg["srcpattern"]);
    auto bufferInfo = fStyleVolume.request();
    if (mcx_config.srcpattern) free(mcx_config.srcpattern);
    mcx_config.srcpattern = (float*) malloc(bufferInfo.size * sizeof(float));
    auto val = static_cast<float*>(bufferInfo.ptr);
    for(int i = 0; i < bufferInfo.size; i++)
      mcx_config.srcpattern[i] = val[i];
  }
  if (user_cfg.contains("invcdf")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(user_cfg["invcdf"]);
    auto bufferInfo = fStyleVolume.request();
    unsigned int nphase = bufferInfo.shape.size();
    float *val = static_cast<float *>(bufferInfo.ptr);
    mcx_config.nphase = nphase + 2;
    mcx_config.nphase += (mcx_config.nphase & 0x1); // make cfg.nphase even number
    mcx_config.invcdf = (float *) calloc(mcx_config.nphase, sizeof(float));
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
    std::string shapesString = py::str(user_cfg["shapes"]);
    if (shapesString.empty())
      throw py::value_error("the 'shapes' field must be a non-empty string");
    mcx_config.shapedata = (char *) calloc(shapesString.size() + 2, 1);
    strncpy(mcx_config.shapedata, shapesString.c_str(), shapesString.size() + 1);
  }
  if (user_cfg.contains("bc")) {
    std::string bcString = py::str(user_cfg["bc"]);
    if (bcString.empty() || bcString.size() > 12)
      throw py::value_error("the 'bc' field must be a non-empty string / have less than 12 characters.");
    strncpy(mcx_config.bc, bcString.c_str(), bcString.size() + 1);
    mcx_config.bc[bcString.size()] = '\0';
  }
  if (user_cfg.contains("seed")) {
    auto seedValue = user_cfg["seed"];
    // If the seed value is scalar (int or float), then assign it directly
    if (py::int_::check_(seedValue))
      mcx_config.seed = py::int_(seedValue);
    else if (py::float_::check_(seedValue))
      mcx_config.seed = py::float_(seedValue).cast<int>();
      // Set seed from array
    else {
      auto fStyleArray = py::array_t<uint8_t, py::array::f_style | py::array::forcecast>::ensure(seedValue);
      auto bufferInfo = fStyleArray.request();
      seed_byte = bufferInfo.shape.at(0);
      if (bufferInfo.shape.at(0) != sizeof(float) * RAND_WORD_LEN)
        throw py::value_error("the row number of cfg.seed does not match RNG seed byte-length");
      mcx_config.replay.seed = malloc(bufferInfo.size);
      memcpy(mcx_config.replay.seed, bufferInfo.ptr, bufferInfo.size);
      mcx_config.seed = SEED_FROM_FILE;
      mcx_config.nphoton = bufferInfo.shape.at(1);
    }
  }
  if (user_cfg.contains("gpuid")) {
    auto gpuIdValue = user_cfg["gpuid"];
    if (py::int_::check_(gpuIdValue)) {
      mcx_config.gpuid = py::int_(gpuIdValue);
      memset(mcx_config.deviceid, 0, MAX_DEVICE);
      if (mcx_config.gpuid > 0 && mcx_config.gpuid < MAX_DEVICE) {
        memset(mcx_config.deviceid, '0', mcx_config.gpuid - 1);
        mcx_config.deviceid[mcx_config.gpuid - 1] = '1';
      } else
        throw py::value_error("GPU id must be positive and can not be more than 256");
    } else if (py::str::check_(gpuIdValue)) {
      std::string gpuIdStringValue = py::str(gpuIdValue);
      if (gpuIdStringValue.empty())
        throw py::value_error("the 'gpuid' field must be an integer or non-empty string");
      if (gpuIdStringValue.size() > MAX_DEVICE)
        throw py::value_error("the 'gpuid' field is too long");
      strncpy(mcx_config.deviceid, gpuIdStringValue.c_str(), MAX_DEVICE);
    }
    for (int i = 0; i < MAX_DEVICE; i++)
      if (mcx_config.deviceid[i] == '0')
        mcx_config.deviceid[i] = '\0';
  }
  if (user_cfg.contains("workload")) {
    auto workloadValue = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(user_cfg["workload"]);
    auto bufferInfo = workloadValue.request();
    if (bufferInfo.shape.size() < 2 && bufferInfo.size > MAX_DEVICE)
      throw py::value_error("the workload list can not be longer than 256");
    for (int i = 0; i < bufferInfo.size; i++)
      mcx_config.workload[i] = static_cast<float *>(bufferInfo.ptr)[i];
  }
  // Output arguments parsing
  GET_SCALAR_FIELD(user_cfg, mcx_config, issave2pt, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, issavedet, py::int_);
  GET_SCALAR_FIELD(user_cfg, mcx_config, issaveseed, py::int_);

  // Flush the std::cout and std::cerr
  std::cout.flush();
  std::cerr.flush();
}


py::dict py_mcx_interface(const py::dict &user_cfg) {
  unsigned int partial_data, hostdetreclen;
  Config mcx_config;  /* mcx_config: structure to store all simulation parameters */
  GPUInfo *gpu_info = nullptr;        /** gpuInfo: structure to store GPU information */
  unsigned int active_dev = 0;     /** activeDev: count of total active GPUs to be used */
  int error_flag = 0;
  int threadid = 0;
  size_t fielddim[6];
  py::dict output;
  try {
    /*
     * To start an MCX simulation, we first create a simulation configuration and set all elements to its default settings.
     */
    parse_config(user_cfg, mcx_config);

    /** The next step, we identify gpu number and query all GPU info */
    if (!(active_dev = mcx_list_gpu(&mcx_config, &gpu_info))) {
      mcx_error(-1, "No GPU device found\n", __FILE__, __LINE__);
    }
    det_ps = nullptr;

    mcx_flush(&mcx_config);

    /** Validate all input fields, and warn incompatible inputs */
    validate_config(&mcx_config, det_ps, dim_det_ps, seed_byte, [](const char *msg) { throw py::value_error(msg); });

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
      int fieldlen =
          static_cast<int>(mcx_config.dim.x) * static_cast<int>(mcx_config.dim.y) * static_cast<int>(mcx_config.dim.z) *
              (int) ((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5) * mcx_config.srcnum;
      if (mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1)
        fieldlen *= mcx_config.detnum;
      if (mcx_config.replay.seed != NULL && mcx_config.outputtype == otRF)
        fieldlen *= 2;
      mcx_config.exportfield = (float *) calloc(fieldlen, sizeof(float));
    }
    if (mcx_config.issavedet == 1) {
      mcx_config.exportdetected = (float *) malloc(hostdetreclen * mcx_config.maxdetphoton * sizeof(float));
    }
    if (mcx_config.issaveseed == 1) {
      mcx_config.seeddata = malloc(mcx_config.maxdetphoton * sizeof(float) * RAND_WORD_LEN);
    }
    if (mcx_config.debuglevel != 0) {
      mcx_config.exportdebugdata = (float *) malloc(mcx_config.maxjumpdebug * sizeof(float) * MCX_DEBUG_REC_LEN);
      mcx_config.debuglevel |= MCX_DEBUG_MOVE;
    }

    /** Start multiple threads, one thread to run portion of the simulation on one CUDA GPU, all in parallel */
#ifdef _OPENMP
    omp_set_num_threads(active_dev);
#pragma omp parallel shared(error_flag)
    {
      threadid = omp_get_thread_num();
#endif
      /** Enclose all simulation calls inside a try/catch construct for exception handling */
      try {
        /** Call the main simulation host function to start the simulation */
        mcx_run_simulation(&mcx_config, gpu_info);

      } catch (const char *err) {
        std::cerr << "Error from thread (" << threadid << "): " << err << std::endl;
        error_flag++;
      } catch (const std::exception &err) {
        std::cerr << "C++ Error from thread (" << threadid << "): " << err.what() << std::endl;
        error_flag++;
      } catch (...) {
        std::cerr << "Unknown Exception from thread (" << threadid << ")" << std::endl;
        error_flag++;
      }
#ifdef _OPENMP
    }
#endif
    /** If error is detected, gracefully terminate the mex and return back to Python */
    if (error_flag)
      throw py::runtime_error("PyMCX Terminated due to an exception!");
    fielddim[4] = 1;
    fielddim[5] = 1;

    if (mcx_config.debuglevel != 0) {
      fielddim[0] = MCX_DEBUG_REC_LEN;
      fielddim[1] = mcx_config.debugdatalen; // his.savedphoton is for one repetition, should correct
      fielddim[2] = 0;
      fielddim[3] = 0;
      auto photonTrajData = py::array_t<float, py::array::f_style>({fielddim[0], fielddim[1]});
      if (mcx_config.debuglevel & MCX_DEBUG_MOVE)
        memcpy(photonTrajData.mutable_data(), mcx_config.exportdebugdata, fielddim[0] * fielddim[1] * sizeof(float));
      if (mcx_config.exportdebugdata)
        free(mcx_config.exportdebugdata);
      mcx_config.exportdebugdata = nullptr;
      output["photontraj"] = photonTrajData;
    }
    if (mcx_config.issaveseed == 1) {
      fielddim[0] = (mcx_config.issaveseed > 0) * RAND_WORD_LEN * sizeof(float);
      fielddim[1] = mcx_config.detectedcount; // his.savedphoton is for one repetition, should correct
      fielddim[2] = 0;
      fielddim[3] = 0;
      auto detectedSeeds = py::array_t<uint8_t, py::array::f_style>({fielddim[0], fielddim[1]});
      memcpy(detectedSeeds.mutable_data(), mcx_config.seeddata, fielddim[0] * fielddim[1]);
      free(mcx_config.seeddata);
      mcx_config.seeddata = nullptr;
      output["detectedseeds"] = detectedSeeds;
    }
    if (user_cfg.contains("dumpmask") && py::reinterpret_borrow<py::bool_>(user_cfg["dumpmask"]).cast<bool>()) {
      fielddim[0] = mcx_config.dim.x;
      fielddim[1] = mcx_config.dim.y;
      fielddim[2] = mcx_config.dim.z;
      fielddim[3] = 0;
      if (mcx_config.vol) {
        auto detectorVol = py::array_t<uint32_t, py::array::f_style>({fielddim[0], fielddim[1], fielddim[2]});
        memcpy(detectorVol.mutable_data(), mcx_config.vol,
               fielddim[0] * fielddim[1] * fielddim[2] * sizeof(unsigned int));
        output["detector"] = detectorVol;
      }
    }
    if (mcx_config.issavedet == 1) {
      fielddim[0] = hostdetreclen;
      fielddim[1] = mcx_config.detectedcount;
      fielddim[2] = 0;
      fielddim[3] = 0;
      if (mcx_config.detectedcount > 0) {
        auto partialPath = py::array_t<float, py::array::f_style>({fielddim[0], mcx_config.detectedcount});
        memcpy(partialPath.mutable_data(), mcx_config.exportdetected,
               fielddim[0] * fielddim[1] * sizeof(float));
        output["partialpath"] = partialPath;
      }
      free(mcx_config.exportdetected);
      mcx_config.exportdetected = NULL;
    }
    if (mcx_config.issave2pt) {
      int fieldlen;
      fielddim[0] = mcx_config.srcnum * mcx_config.dim.x;
      fielddim[1] = mcx_config.dim.y;
      fielddim[2] = mcx_config.dim.z;
      fielddim[3] = (int) ((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5);
      if (mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1)
        fielddim[4] = mcx_config.detnum;
      if (mcx_config.replay.seed != nullptr && mcx_config.outputtype == otRF)
        fielddim[5] = 2;
      fieldlen = fielddim[0] * fielddim[1] * fielddim[2] * fielddim[3] * fielddim[4] * fielddim[5];
      py::detail::any_container<ssize_t> arrayDims;
      if (fielddim[5] > 1)
        arrayDims = {fielddim[0], fielddim[1], fielddim[2], fielddim[3], fielddim[4], fielddim[5]};
      else if (fielddim[4] > 1)
        arrayDims = {fielddim[0], fielddim[1], fielddim[2], fielddim[3], fielddim[4]};
      else
        arrayDims = {fielddim[0], fielddim[1], fielddim[2], fielddim[3]};
      auto drefArray = py::array_t<float, py::array::f_style>(arrayDims);
      if (mcx_config.issaveref) {
        auto *dref = static_cast<float *>(drefArray.mutable_data());
        memcpy(dref, mcx_config.exportfield, fieldlen * sizeof(float));
        for (int i = 0; i < fieldlen; i++) {
          if (dref[i] < 0.f) {
            dref[i] = -dref[i];
            mcx_config.exportfield[i] = 0.f;
          } else
            dref[i] = 0.f;
        }
        output["dref"] = drefArray;
      }
      auto data = py::array_t<float, py::array::f_style>(arrayDims);
      memcpy(data.mutable_data(), mcx_config.exportfield, fieldlen * sizeof(float));
      output["data"] = data;
      free(mcx_config.exportfield);
      mcx_config.exportfield = nullptr;
      output["runtime"] = mcx_config.runtime;
      output["nphoton"] = mcx_config.nphoton * ((mcx_config.respin > 1) ? (mcx_config.respin) : 1);
      output["energytot"] = mcx_config.energytot;
      output["energyabs"] = mcx_config.energyabs;
      output["normalizer"] = mcx_config.normalizer;
      output["unitinmm"] = mcx_config.normalizer;
      py::list workload;
      for (int i = 0; i < active_dev; i++)
        workload.append(mcx_config.workload[i]);
      output["workload"] = workload;

      /** return the final optical properties for polarized MCX simulation */
      if (mcx_config.polprop) {
        for (int i = 0; i < mcx_config.polmedianum; i++) {
          // restore mua and mus values
          mcx_config.prop[i + 1].mua /= mcx_config.unitinmm;
          mcx_config.prop[i + 1].mus /= mcx_config.unitinmm;
        }
        auto optProperties = py::array_t<float, py::array::f_style>({4, int(mcx_config.medianum)});
        memcpy(optProperties.mutable_data(), mcx_config.prop, mcx_config.medianum * 4 * sizeof(float));
        output["opticalprops"] = optProperties;
      }
    }
  } catch (const char *err) {
    std::cerr << "Error: " << err << std::endl;
  } catch (const std::exception &err) {
    std::cerr << "C++ Error: " << err.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown Exception" << std::endl;
  }

  /** Clear up simulation data structures by calling the destructors */
  if (det_ps)
    free(det_ps);
  mcx_cleargpuinfo(&gpu_info);
  mcx_clearcfg(&mcx_config);
  // return a pointer to the MCX output, wrapped in a std::vector
  return output;
}

void printMCXUsage() {
  std::cout
      << "PyMCX v2021.2\nUsage:\n    output = pymcx.mcx(cfg);\n\nRun 'help(pymcx.mcx)' for more details.\n";
}

py::dict py_mcx_interface_wargs(py::args args, const py::kwargs &kwargs) {
  if (py::len(kwargs) == 0) {
    printMCXUsage();
    return {};
  }
  return py_mcx_interface(kwargs);
}

py::list get_GPU_info() {
  Config mcxConfig;            /** mcxconfig: structure to store all simulation parameters */
  GPUInfo *gpuInfo = nullptr;        /** gpuinfo: structure to store GPU information */
  mcx_initcfg(&mcxConfig);
  mcxConfig.isgpuinfo = 3;
  py::list output;
  if (!(mcx_list_gpu(&mcxConfig, &gpuInfo))) {
    std::cerr << "No CUDA-capable device was found." << std::endl;
    return output;
  }

  for (int i = 0; i < gpuInfo[0].devcount; i++) {
    py::dict currentDeviceInfo;
    currentDeviceInfo["name"] = gpuInfo[i].name;
    currentDeviceInfo["id"] = gpuInfo[i].id;
    currentDeviceInfo["devcount"] = gpuInfo[i].devcount;
    currentDeviceInfo["major"] = gpuInfo[i].major;
    currentDeviceInfo["minor"] = gpuInfo[i].minor;
    currentDeviceInfo["globalmem"] = gpuInfo[i].globalmem;
    currentDeviceInfo["constmem"] = gpuInfo[i].constmem;
    currentDeviceInfo["sharedmem"] = gpuInfo[i].sharedmem;
    currentDeviceInfo["regcount"] = gpuInfo[i].regcount;
    currentDeviceInfo["clock"] = gpuInfo[i].clock;
    currentDeviceInfo["sm"] = gpuInfo[i].sm;
    currentDeviceInfo["core"] = gpuInfo[i].core;
    currentDeviceInfo["autoblock"] = gpuInfo[i].autoblock;
    currentDeviceInfo["autothread"] = gpuInfo[i].autothread;
    currentDeviceInfo["maxgate"] = gpuInfo[i].maxgate;
    output.append(currentDeviceInfo);
  }
  mcx_cleargpuinfo(&gpuInfo);
  mcx_clearcfg(&mcxConfig);
  return output;
}

PYBIND11_MODULE(pymcx, m) {
  m.doc() = "PyMCX: Monte Carlo eXtreme Python Interface, www.mcx.space.";
  m.def("mcx", &py_mcx_interface, "Runs MCX with the given config.", py::call_guard<py::scoped_ostream_redirect,
                                                                                    py::scoped_estream_redirect>());
  m.def("mcx", &py_mcx_interface_wargs, "Runs MCX with the given config.", py::call_guard<py::scoped_ostream_redirect,
                                                                                          py::scoped_estream_redirect>());
  m.def("gpu_info",
        &get_GPU_info,
        "Prints out the list of CUDA-capable devices attached to this system.",
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>());
}


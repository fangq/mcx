/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Matin Raayai Ardakani <raayaiardakani.m at northeastern.edu>
**  \copyright Matin Raayai Ardakani, 2022
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

/***************************************************************************//**
\file    pmcx.cpp

@brief   Python interface using Pybind11 for MCX
*******************************************************************************/
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <cerrno>
#include <cstring>
#include <vector>
#include <cstdarg>
#include <streambuf>
#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <complex>
#include <string>
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_const.h"
#include "mcx_shapes.h"
#include "mcx_lang.h"
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
            pyecho() << #property << ": " << (float) (dst_mcx_config).property << std::endl;} \
        catch (const std::runtime_error &err)\
        {throw py::type_error(std::string("Failed to assign MCX property " + std::string(#property) + ". Reason: " + err.what()));}\
    }

#define GET_VEC3_FIELD(src, dst, prop, type) if (src.contains(#prop)) {try {auto list = py::list(src[#prop]);\
            dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>()};\
            pyecho() << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << "]\n";} \
        catch (const std::runtime_error &err ) {throw py::type_error(std::string("Failed to assign MCX property " + std::string(#prop) + ". Reason: " + err.what()));}}

#define GET_VEC4_FIELD(src, dst, prop, type) if (src.contains(#prop)) {try {auto list = py::list(src[#prop]);\
            dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list[3].cast<type>()}; \
            pyecho() << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << ", " << dst.prop.w << "]\n";} \
        catch (const std::runtime_error &err ) {throw py::type_error(std::string("Failed to assign MCX property " + std::string(#prop) + ". Reason: " + err.what()));}}

#define GET_VEC34_FIELD(src, dst, prop, type) if (src.contains(#prop)) {try {auto list = py::list(src[#prop]);\
            dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list.size() == 4 ? list[3].cast<type>() : dst.prop.w}; \
            pyecho() << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << ", " << dst.prop.w << "]\n";} \
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

    size_t dim_xyz = 0;

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
        int i = buffer.shape.size() == 4;

        /* a 4-D uint8 array is a 4-byte (asgn_byte) or 8-byte (svmc, e.g. from pmcx.svmc) record per
           voxel, the same as an int8 array and as uint8 arrays in mcxlab */
        if (i == 1 && buffer.shape.at(0) != 4 && buffer.shape.at(0) != 8) {
            throw py::value_error("Invalid volume dims for uint8_t volume: the 1st dimension of a 4-D array must be 4 or 8.");
        }

        mcx_config.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                          static_cast<unsigned int>(buffer.shape.at(i + 1)),
                          static_cast<unsigned int>(buffer.shape.at(i + 2))
                         };
        dim_xyz = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;

        if (i == 1) {
            mcx_config.mediabyte = (buffer.shape.at(0) == 4) ? MEDIA_ASGN_BYTE : MEDIA_2LABEL_SPLIT;
            mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * buffer.shape.at(0)));
            memcpy(mcx_config.vol, buffer.ptr, dim_xyz * buffer.shape.at(0));
        } else {
            mcx_config.vol = static_cast<unsigned int*>(malloc(dim_xyz * sizeof(unsigned int)));

            for (size_t k = 0; k < (size_t)buffer.size; k++) {
                mcx_config.vol[k] = static_cast<unsigned char*>(buffer.ptr)[k];
            }
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
        mcx_config.mediabyte = 4;
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
                        f2bh.f[2] = val[i * 3];        // mua/mus/g or n float-value, depending on f[1]
                        f2bh.f[1] = val[i * 3 + 1];    // 0-3: f[2] replaces the mua (0), mus (1), g (2) or n (3) of the label, as the kernel decodes it
                        f2bh.f[0] = val[i * 3 + 2];    // voxel label

                        if (f2bh.f[1] < 0.f || f2bh.f[1] >= 4.f || f2bh.f[0] < 0.f) {
                            throw py::value_error("the 2nd volume must have an integer value between 0 and 3");
                        }

                        if (f2bh.f[1] < 2.f) { // mua (0) or mus (1) in 1/mm: scale by unitinmm; not g (2) or n (3)
                            f2bh.f[2] *= mcx_config.unitinmm;
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

                case 2:
                case 4: {
                    mcx_config.mediabyte = (buffer.shape.at(0) == 2 ? MEDIA_AS_F2H : MEDIA_ASGN_F2H);
                    auto val = (float*) buffer.ptr;

                    float f2h[2];
                    int offset = (mcx_config.mediabyte == MEDIA_ASGN_F2H);

                    if (mcx_config.mediabyte == MEDIA_ASGN_F2H) {
                        mcx_config.vol = static_cast<unsigned int*>(realloc(mcx_config.vol, dim_xyz * 2 * sizeof(unsigned int)));
                    }

                    for (i = 0; i < dim_xyz; i++) {
                        f2h[0] = val[i << (1 + offset)] * mcx_config.unitinmm;        // mua
                        f2h[1] = val[(i << (1 + offset)) + 1] * mcx_config.unitinmm; // mus

                        if (f2h[0] != f2h[0]
                                || f2h[1] != f2h[1]) { /*if one of mua/mus is nan in continuous medium, convert to 0-voxel*/
                            mcx_config.vol[i] = 0;
                            continue;
                        }

                        if (mcx_config.mediabyte == MEDIA_ASGN_F2H) {
                            mcx_config.vol[i] = mcx_float2half2(f2h);
                            f2h[0] = val[(i << 2) + 2];   // g
                            f2h[1] = val[(i << 2) + 3];   // n
                            mcx_config.vol[i + dim_xyz] = mcx_float2half2(f2h);
                        } else {
                            mcx_config.vol[i] = mcx_float2half2(f2h);
                        }
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
                        f2i.f = val[i] * mcx_config.unitinmm; // mua

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
            mcx_config.mediabyte = 4;  /* float32 3D labels are stored as 32bit integer labels */

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

/**
 * The echo of the cfg fields (and the warnings printed while reading them) goes where cfg.flog sends
 * the simulation log: Python's sys.stdout (std::cout, redirected by pybind11) for 2 (default), or
 * the stdout of the process for 1 and 0, the log file, or nowhere for os.devnull
 */
/**
 * 'NUL' (any case) and '/dev/null' name the null device on every OS: cfg.flog='NUL' on Linux
 * or macOS wrote, or failed to write, a file of that name (fangq/mcx#280)
 */
static const char* mcx_logpath(const char* path) {
#ifdef _WIN32
    const char* nulldev = "NUL";
#else
    const char* nulldev = "/dev/null";
#endif

    if (strcmp(path, "/dev/null") == 0 || ((path[0] == 'N' || path[0] == 'n') && (path[1] == 'U' || path[1] == 'u')
                                           && (path[2] == 'L' || path[2] == 'l') && path[3] == '\0')) {
        return nulldev;
    }

    return path;
}

static int pyecho_lognull = 0;     /**< 1: cfg['flog'] is the null device, which also silences the progress bar */
static FILE* pyecho_logfp = NULL;   /**< NULL or stderr: std::cout, i.e. sys.stdout */

class PyEchoFileBuf : public std::streambuf {
  public:
    FILE* fp = NULL;
  protected:
    int overflow(int c) override {
        return (c == EOF || fputc(c, fp) == EOF) ? EOF : c;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        return (std::streamsize)fwrite(s, 1, (size_t)n, fp);
    }
};

static std::ostream& pyecho() {
    static PyEchoFileBuf filebuf;
    static std::ostream fileout(&filebuf);

    if (pyecho_logfp == NULL || pyecho_logfp == stderr) {
        return std::cout;
    }

    filebuf.fp = pyecho_logfp;
    return fileout;
}

void parse_config(const py::dict& user_cfg, Config& mcx_config) {
    mcx_initcfg(&mcx_config);

    mcx_config.flog = stderr;

    /* cfg.flog first: the echo of the other fields goes where it sends the log */
    pyecho_lognull = 0;

    if (user_cfg.contains("flog")) {
        auto logfile_id_value = user_cfg["flog"];

        if (py::int_::check_(logfile_id_value)) {
            int logid = py::int_(logfile_id_value);
            mcx_config.flog = (logid >= 2 ? stderr : (logid == 1 ? stdout : (mcx_config.printnum = -1, stdout)));
        } else if (py::str::check_(logfile_id_value)) {
            std::string logfile_id_string_value = py::str(logfile_id_value);

            if (logfile_id_string_value.empty()) {
                throw py::value_error("the 'flog' field must be an integer or non-empty string");
            }

            pyecho_lognull = (mcx_logpath(logfile_id_string_value.c_str()) != logfile_id_string_value.c_str());
            mcx_config.flog = fopen(mcx_logpath(logfile_id_string_value.c_str()), "a+");

            if (mcx_config.flog == NULL) {
                throw py::value_error("the log file '" + logfile_id_string_value + "' of cfg['flog'] can not be written: " + strerror(errno));
            }
        }
    }

    pyecho_logfp = mcx_config.flog;
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
    GET_SCALAR_FIELD(user_cfg, mcx_config, isgpuinfo, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, printnum, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, voidtime, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issaveref, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issaveexit, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, ismomentum, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, isspecular, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, istrajstokes, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, replaydet, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, faststep, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, maxvoidstep, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, maxjumpdebug, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, gscatter, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, srcnum, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, srcid, py::int_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, omega, py::float_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, lambda, py::float_);
    GET_VEC3_FIELD(user_cfg, mcx_config, steps, float);
    GET_VEC3_FIELD(user_cfg, mcx_config, crop0, uint);
    GET_VEC3_FIELD(user_cfg, mcx_config, crop1, uint);
    GET_VEC4_FIELD(user_cfg, mcx_config, srciquv, float);
    parseVolume(user_cfg, mcx_config);

    if (user_cfg.contains("srcpos")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["srcpos"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid srcpos field value");
        }

        auto buffer_info = f_style_volume.request();

        size_t arraydim[2] = {0};

        if (buffer_info.shape.size() == 1) {
            arraydim[0] = 1;
            arraydim[1] = buffer_info.shape.at(0);
        } else if (buffer_info.shape.size() > 1) {
            arraydim[0] = buffer_info.shape.at(0);
            arraydim[1] = buffer_info.shape.at(1);
        }

        if (arraydim[0] == 0 || arraydim[1] < 3 || arraydim[1] > 4) {
            throw py::value_error("the 'srcpos' field must have 3 or 4 columns (x,y,z,w0)");
        }

        auto val = static_cast<float*>(buffer_info.ptr);

        for (int i = 0; i < arraydim[1]; i++) {
            ((float*)(&mcx_config.srcpos.x))[i] = val[i * arraydim[0]];
        }

        if (arraydim[0] > 1 || mcx_config.extrasrclen > 0) {
            if (mcx_config.extrasrclen && mcx_config.extrasrclen != arraydim[0] - 1) {
                throw py::value_error("Length of sub-elements of srcpos/srcdir/srcparam1/srcparam2 must match");
            } else {
                mcx_config.extrasrclen = arraydim[0] - 1;
            }

            if (mcx_config.srcdata == NULL) {
                mcx_config.srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), mcx_config.extrasrclen);
            }

            for (int j = 0; j < arraydim[1]; j++)
                for (int i = 0; i < mcx_config.extrasrclen; i++) {
                    ((float*)(&mcx_config.srcdata[i].srcpos.x))[j] = val[j * arraydim[0] + i + 1];
                }
        }
    }

    if (user_cfg.contains("srcdir")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["srcdir"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid srcdir field value");
        }

        auto buffer_info = f_style_volume.request();

        size_t arraydim[2] = {0};

        if (buffer_info.shape.size() == 1) {
            arraydim[0] = 1;
            arraydim[1] = buffer_info.shape.at(0);
        } else if (buffer_info.shape.size() > 1) {
            arraydim[0] = buffer_info.shape.at(0);
            arraydim[1] = buffer_info.shape.at(1);
        }

        if (arraydim[0] == 0 || arraydim[1] < 3 || arraydim[1] > 4) {
            throw py::value_error("the 'srcdir' field must have 3 or 4 columns (vx,vy,vz,focallength)");
        }

        auto val = static_cast<float*>(buffer_info.ptr);

        for (int i = 0; i < arraydim[1]; i++) {
            ((float*)(&mcx_config.srcdir.x))[i] = val[i * arraydim[0]];
        }

        if (arraydim[0] > 1 || mcx_config.extrasrclen > 0) {
            if (mcx_config.extrasrclen && mcx_config.extrasrclen != arraydim[0] - 1) {
                throw py::value_error("Length of sub-elements of srcpos/srcdir/srcparam1/srcparam2 must match");
            } else {
                mcx_config.extrasrclen = arraydim[0] - 1;
            }

            if (mcx_config.srcdata == NULL) {
                mcx_config.srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), mcx_config.extrasrclen);
            }

            for (int j = 0; j < arraydim[1]; j++)
                for (int i = 0; i < mcx_config.extrasrclen; i++) {
                    ((float*)(&mcx_config.srcdata[i].srcdir.x))[j] = val[j * arraydim[0] + i + 1];
                }
        }
    }

    if (user_cfg.contains("srcparam1")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["srcparam1"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid srcparam1 field value");
        }

        auto buffer_info = f_style_volume.request();

        size_t arraydim[2] = {0};

        if (buffer_info.shape.size() == 1) {
            arraydim[0] = 1;
            arraydim[1] = buffer_info.shape.at(0);
        } else if (buffer_info.shape.size() > 1) {
            arraydim[0] = buffer_info.shape.at(0);
            arraydim[1] = buffer_info.shape.at(1);
        }

        if (arraydim[0] == 0 || (arraydim[1] == 0 && arraydim[1] > 4)) {
            throw py::value_error("the 'srcparam1' field must have 1-4 columns");
        }

        auto val = static_cast<float*>(buffer_info.ptr);

        for (int i = 0; i < arraydim[1]; i++) {
            ((float*)(&mcx_config.srcparam1.x))[i] = val[i * arraydim[0]];
        }

        if (arraydim[0] > 1 || mcx_config.extrasrclen > 0) {
            if (mcx_config.extrasrclen && mcx_config.extrasrclen != arraydim[0] - 1) {
                throw py::value_error("Length of sub-elements of srcpos/srcdir/srcparam1/srcparam2 must match");
            } else {
                mcx_config.extrasrclen = arraydim[0] - 1;
            }

            if (mcx_config.srcdata == NULL) {
                mcx_config.srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), mcx_config.extrasrclen);
            }

            for (int j = 0; j < arraydim[1]; j++)
                for (int i = 0; i < mcx_config.extrasrclen; i++) {
                    ((float*)(&mcx_config.srcdata[i].srcparam1.x))[j] = val[j * arraydim[0] + i + 1];
                }
        }
    }

    if (user_cfg.contains("srcparam2")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["srcparam2"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid srcparam2 field value");
        }

        auto buffer_info = f_style_volume.request();

        size_t arraydim[2] = {0};

        if (buffer_info.shape.size() == 1) {
            arraydim[0] = 1;
            arraydim[1] = buffer_info.shape.at(0);
        } else if (buffer_info.shape.size() > 1) {
            arraydim[0] = buffer_info.shape.at(0);
            arraydim[1] = buffer_info.shape.at(1);
        }

        if (arraydim[0] == 0 || (arraydim[1] == 0 && arraydim[1] > 4)) {
            throw py::value_error("the 'srcparam2' field must have 1-4 columns");
        }

        auto val = static_cast<float*>(buffer_info.ptr);

        for (int i = 0; i < arraydim[1]; i++) {
            ((float*)(&mcx_config.srcparam2.x))[i] = val[i * arraydim[0]];
        }

        if (arraydim[0] > 1 || mcx_config.extrasrclen > 0) {
            if (mcx_config.extrasrclen && mcx_config.extrasrclen != arraydim[0] - 1) {
                throw py::value_error("Length of sub-elements of srcpos/srcdir/srcparam1/srcparam2 must match");
            } else {
                mcx_config.extrasrclen = arraydim[0] - 1;
            }

            if (mcx_config.srcdata == NULL) {
                mcx_config.srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), mcx_config.extrasrclen);
            }

            for (int j = 0; j < arraydim[1]; j++)
                for (int i = 0; i < mcx_config.extrasrclen; i++) {
                    ((float*)(&mcx_config.srcdata[i].srcparam2.x))[j] = val[j * arraydim[0] + i + 1];
                }
        }
    }

    if (user_cfg.contains("detpos")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["detpos"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid detpos field value");
        }

        auto buffer_info = f_style_volume.request();

        if ((buffer_info.shape.size() > 1 && buffer_info.shape.at(0) > 0 && buffer_info.shape.at(1) != 4) || (buffer_info.shape.size() == 1 && buffer_info.shape.at(0) != 4)) {
            throw py::value_error("the 'detpos' field must have 4 columns (x,y,z,radius)");
        }

        mcx_config.detnum = (buffer_info.shape.size() == 1) ? 1 : buffer_info.shape.at(0);

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

    if (user_cfg.contains("detdir")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["detdir"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid detdir field value");
        }

        auto buffer_info = f_style_volume.request();

        /* [vx,vy,vz] or [vx,vy,vz,focallength] per detector; a missing focal length stays 0: no focusing */
        int detdir_cols = (buffer_info.shape.size() == 1) ? buffer_info.shape.at(0) : ((buffer_info.shape.size() > 1) ? buffer_info.shape.at(1) : 0);

        if ((buffer_info.shape.size() < 2 || buffer_info.shape.at(0) > 0) && (detdir_cols < 3 || detdir_cols > 4)) {
            throw py::value_error("the 'detdir' field must have 3 or 4 columns (vx,vy,vz,focallength)");
        }

        int detdir_count = (buffer_info.shape.size() == 1) ? 1 : buffer_info.shape.at(0);

        if (mcx_config.detnum > 0 && detdir_count != mcx_config.detnum) {
            throw py::value_error("the number of rows in 'detdir' must match 'detpos'");
        }

        if (mcx_config.detdir) {
            free(mcx_config.detdir);
        }

        mcx_config.detdir = (float4*) malloc(detdir_count * sizeof(float4));
        memset(mcx_config.detdir, 0, detdir_count * sizeof(float4));
        auto val = static_cast<float*>(buffer_info.ptr);

        for (int j = 0; j < detdir_cols; j++)
            for (int ii = 0; ii < detdir_count; ii++) {
                ((float*) (&mcx_config.detdir[ii]))[j] = val[j * detdir_count + ii];
            }
    }

    if (user_cfg.contains("sdmap")) {
        auto f_style_volume = py::array_t < double, py::array::f_style | py::array::forcecast >::ensure(user_cfg["sdmap"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid sdmap field value");
        }

        auto buffer_info = f_style_volume.request();

        if (mcx_config.sdmap) {
            free(mcx_config.sdmap);
            mcx_config.sdmap = nullptr;
        }

        mcx_config.sdmaplen = 0;

        if (buffer_info.size > 0) {
            if ((buffer_info.shape.size() > 1 && buffer_info.shape.at(1) < 2) || (buffer_info.shape.size() == 1 && buffer_info.shape.at(0) < 2) || buffer_info.shape.size() > 2) {
                throw py::value_error("the 'sdmap' field must have 2 or more columns (srcid,detid,...)");
            }

            unsigned int nch = (buffer_info.shape.size() == 1) ? 1 : buffer_info.shape.at(0);
            auto val = static_cast<double*>(buffer_info.ptr);

            mcx_config.sdmap = (unsigned int*) malloc(sizeof(unsigned int) * 2 * nch);
            mcx_config.sdmaplen = nch;

            for (unsigned int ii = 0; ii < nch; ii++) {
                if (val[ii] < 1 || val[nch + ii] < 1) {
                    throw py::value_error("the source and detector indices in 'sdmap' start from 1");
                }

                mcx_config.sdmap[ii << 1] = (unsigned int)val[ii] - 1;
                mcx_config.sdmap[(ii << 1) + 1] = (unsigned int)val[nch + ii] - 1;
            }
        }
    }

    if (user_cfg.contains("prop")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["prop"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid prop field format");
        }

        auto buffer_info = f_style_volume.request();

        if ((buffer_info.shape.size() > 1 && buffer_info.shape.at(0) > 0 && buffer_info.shape.at(1) != 4) || buffer_info.shape.size() == 1 && buffer_info.shape.at(0) != 4) {
            throw py::value_error("the 'prop' field must have 4 columns (mua,mus,g,n)");
        }

        mcx_config.medianum = (buffer_info.shape.size() == 1) ? 1 : buffer_info.shape.at(0);

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

        if ((buffer_info.shape.size() > 1 && buffer_info.shape.at(0) > 0 && buffer_info.shape.at(1) != 5) || (buffer_info.shape.size() == 1 && buffer_info.shape.at(0) != 5)) {
            throw py::value_error("the 'polprop' field must have 5 columns (mua, radius, rho, n_sph,n_bkg)");
        }

        mcx_config.polmedianum = (buffer_info.shape.size() == 1) ? 1 : buffer_info.shape.at(0);

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
                                   "line", "slit", "pencilarray", "pattern3d", "hyperboloid", "ring", ""
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
        const char* outputtype[] = {"flux", "fluence", "energy", "jacobian", "nscat", "wl", "wp", "wm", "rf", "length", "rfmus", "wltof", "wptof", "adjoint",
                                    "adjoint_dcoeff", "adjoint_mus", "adjoint_musp", "adjoint_mua_d", "adjoint_mua_musp", nullptr
                                   };
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
        const char debugflag[] = {'R', 'M', 'P', 'T', '\0'};  /* as the command line and mcxlab */
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

    /* the progress bar is written to the terminal, not to the log: drop it when the log is the
       null device, so that cfg['flog']='/dev/null' or 'NUL' silences everything */
    if (pyecho_lognull) {
        mcx_config.debuglevel &= ~MCX_DEBUG_PROGRESS;
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
        unsigned int nphase = buffer_info.size;
        float* val = static_cast<float*>(buffer_info.ptr);
        mcx_config.nphase = nphase + 2;
        mcx_config.invcdf = (float*) calloc(mcx_config.nphase, sizeof(float));

        for (int i = 0; i < nphase; i++) {
            mcx_config.invcdf[i + 1] = val[i];

            if ((i > 0 && val[i] < val[i - 1]) || val[i] > 1.f || val[i] < -1.f)
                throw py::value_error(
                    "cfg.invcdf contains invalid data; it must be a monotonically increasing vector with all values between -1 and 1");
        }

        mcx_config.invcdf[0] = -1.f;
        mcx_config.invcdf[mcx_config.nphase - 1] = 1.f;
    }

    if (user_cfg.contains("angleinvcdf")) {
        auto f_style_volume = py::array_t < float, py::array::f_style | py::array::forcecast >::ensure(user_cfg["angleinvcdf"]);

        if (!f_style_volume) {
            throw py::value_error("Invalid angleinvcdf field value");
        }

        auto buffer_info = f_style_volume.request();
        unsigned int nangle = buffer_info.size;
        float* val = static_cast<float*>(buffer_info.ptr);
        mcx_config.nangle = nangle;
        mcx_config.angleinvcdf = (float*) calloc(mcx_config.nangle, sizeof(float));

        for (int i = 0; i < nangle; i++) {
            mcx_config.angleinvcdf[i] = val[i];

            if ((i > 0 && val[i] < val[i - 1]) || val[i] > 1.f || val[i] < 0.f)
                throw py::value_error(
                    "cfg.angleinvcdf contains invalid data; it must be a monotonically increasing vector with all values between 0 and 1");
        }
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
            throw py::value_error("the 'bc' field must be a non-empty string / have no more than 12 characters.");
        }

        strncpy(mcx_config.bc, bc_string.c_str(), bc_string.size() + 1); // copy the string, plus the ending NULL, max 13 char
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


    if (user_cfg.contains("lang")) {
        std::string langid = py::str(user_cfg["lang"]);

        if (langid.empty()) {
            throw py::value_error("the 'lang' field must be a non-empty string");
        }

        int idx = mcx_keylookup((char*)langid.c_str(), languagename);

        if (idx == -1) {
            throw py::value_error("Unsupported language");
        }

        mcx_lang = cJSON_Parse(translations[idx]);
    }

    // Output arguments parsing
    GET_SCALAR_FIELD(user_cfg, mcx_config, issave2pt, py::bool_);
    GET_SCALAR_FIELD(user_cfg, mcx_config, issavedet, py::int_);
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

    if (mcx_config.flog && mcx_config.flog != stdout && mcx_config.flog != stderr) {
        fclose(mcx_config.flog);
        mcx_config.flog = NULL;
    }

    pyecho_logfp = NULL;

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
    unsigned int debuglen = 0;  /** number of float numbers per photon trajectory position */

    try {
        /*
         * To start an MCX simulation, we first create a simulation configuration and set all elements to its default settings.
         */
        det_ps = nullptr;

        parse_config(user_cfg, mcx_config);

        debuglen = MCX_DEBUG_REC_LEN + (mcx_config.istrajstokes << 2);

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

        /** For adjoint mode: append detectors as disk-shaped extra sources before exportfield allocation */
        if ((MCX_IS_ADJOINT_TYPE(mcx_config.outputtype) || mcx_config.srcid == -2) && mcx_config.seed != SEED_FROM_FILE && mcx_config.detnum > 0 && mcx_config.detdir != nullptr) {
            unsigned int origextrasrclen = mcx_config.extrasrclen;
            mcx_config.srcdata = (ExtraSrc*)realloc(mcx_config.srcdata, (mcx_config.extrasrclen + mcx_config.detnum) * sizeof(ExtraSrc));
            memset(mcx_config.srcdata + mcx_config.extrasrclen, 0, mcx_config.detnum * sizeof(ExtraSrc));

            for (unsigned int adi = 0; adi < mcx_config.detnum; adi++) {
                mcx_config.srcdata[origextrasrclen + adi].srcpos.x = mcx_config.detpos[adi].x;
                mcx_config.srcdata[origextrasrclen + adi].srcpos.y = mcx_config.detpos[adi].y;
                mcx_config.srcdata[origextrasrclen + adi].srcpos.z = mcx_config.detpos[adi].z;
                mcx_config.srcdata[origextrasrclen + adi].srcpos.w = 1.f;
                mcx_config.srcdata[origextrasrclen + adi].srcdir   = mcx_config.detdir[adi];
                mcx_config.srcdata[origextrasrclen + adi].srcparam1.x = mcx_config.detpos[adi].w;
            }

            mcx_config.extrasrclen += mcx_config.detnum;

            if (MCX_IS_ADJOINT_TYPE(mcx_config.outputtype)) {
                mcx_config.srcid = -1;
            }
        }

        /** Initialize all buffers necessary to store the output variables */
        if (mcx_config.issave2pt == 1) {
            size_t field_len =
                static_cast<int>(mcx_config.dim.x) * static_cast<int>(mcx_config.dim.y) * static_cast<int>(mcx_config.dim.z) *
                (size_t) ((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5) * mcx_config.srcnum;

            if (mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1) {
                field_len *= mcx_config.detnum;
            }

            if (((mcx_config.replay.seed != nullptr && (mcx_config.outputtype == otRF || mcx_config.outputtype == otRFmus)) || (mcx_config.omega > 0.f && mcx_config.seed != SEED_FROM_FILE))) {
                field_len *= 2;
            }

            if (mcx_config.extrasrclen && mcx_config.srcid < 0) {
                field_len *= mcx_fieldslots(&mcx_config);  /* Ns+Nd, or only the slots cfg['sdmap'] uses */
            }

            mcx_config.exportfield = (float*) calloc(field_len, sizeof(float));
        }

        if (mcx_config.issavedet >= 1) {
            mcx_config.exportdetected = (float*) malloc(hostdetreclen * mcx_config.maxdetphoton * sizeof(float));
        }

        if (mcx_config.issaveseed == 1) {
            mcx_config.seeddata = malloc(mcx_config.maxdetphoton * sizeof(float) * RAND_WORD_LEN);
        }

        if (mcx_config.debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
            mcx_config.exportdebugdata = (float*) malloc(mcx_config.maxjumpdebug * sizeof(float) * debuglen);
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
            field_dim[0] = debuglen;
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

        if (mcx_config.issavedet >= 1) {
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
            size_t field_len;
            field_dim[0] = mcx_config.srcnum * mcx_config.dim.x;
            field_dim[1] = mcx_config.dim.y;
            field_dim[2] = mcx_config.dim.z;
            field_dim[3] = (int) ((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5);

            if (mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1) {
                field_dim[4] = mcx_config.detnum;
            }

            if (((mcx_config.replay.seed != nullptr && (mcx_config.outputtype == otRF || mcx_config.outputtype == otRFmus)) || (mcx_config.omega > 0.f && mcx_config.seed != SEED_FROM_FILE))) {
                field_dim[5] = 2;
            }

            if (mcx_config.extrasrclen && mcx_config.srcid < 0) {
                field_dim[5] *= mcx_fieldslots(&mcx_config);
            }

            /** adjoint forward fluence: reshape to [Nx, Ny, Nz, maxgate, Ns+Nd] (5D); with cfg['sdmap'], only the slots it uses */
            if (MCX_IS_ADJOINT_TYPE(mcx_config.outputtype) && mcx_config.seed != SEED_FROM_FILE && mcx_config.detdir != nullptr) {
                int nsrcslots = mcx_fieldslots(&mcx_config);
                int isrf = (mcx_config.omega > 0.f) ? 1 : 0;
                field_dim[0] = mcx_config.dim.x;
                field_dim[4] = nsrcslots;
                field_dim[5] = isrf ? 2 : 1;
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
                int highdim = field_dim[3] * field_dim[4] * field_dim[5];
                int voxellen = mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z;
                auto* dref = static_cast<float*>(dref_array.mutable_data());
                memcpy(dref, mcx_config.exportfield, field_len * sizeof(float));

                for (int voxelid = 0; voxelid < voxellen; voxelid++) {
                    if (mcx_config.vol[voxelid]) {
                        for (int gate = 0; gate < highdim; gate++)
                            for (int srcid = 0; srcid < mcx_config.srcnum; srcid++) {
                                dref[(gate * voxellen + voxelid) * mcx_config.srcnum + srcid] = 0.f;
                            }
                    } else {
                        for (int gate = 0; gate < highdim; gate++)
                            for (int srcid = 0; srcid < mcx_config.srcnum; srcid++) {
                                dref[(gate * voxellen + voxelid) * mcx_config.srcnum + srcid] = -dref[(gate * voxellen + voxelid) * mcx_config.srcnum + srcid];
                                mcx_config.exportfield[(gate * voxellen + voxelid) * mcx_config.srcnum + srcid] = 0.f;
                            }
                    }
                }

                output["dref"] = dref_array;
            }

            if (mcx_config.omega > 0.f && mcx_config.seed != SEED_FROM_FILE && mcx_config.exportfield) {
                /** RF forward/adjoint: convert [Re, Im] halves to complex numpy array */
                size_t halflen = field_len / 2;
                std::vector<size_t> cdims;

                for (int i = 0; i < 4; i++) {
                    cdims.push_back(field_dim[i]);
                }

                if (field_dim[4] > 1) {
                    cdims.push_back(field_dim[4]);
                }

                auto cdata = py::array_t<std::complex<float>, py::array::f_style>(cdims);
                auto* cptr = (std::complex<float>*)cdata.mutable_data();

                for (size_t i = 0; i < halflen; i++) {
                    cptr[i] = std::complex<float>(mcx_config.exportfield[i], mcx_config.exportfield[i + halflen]);
                }

                output["flux"] = cdata;
            } else {
                auto data = py::array_t<float, py::array::f_style>(array_dims);
                memcpy(data.mutable_data(), mcx_config.exportfield, field_len * sizeof(float));
                output["flux"] = data;
            }

            free(mcx_config.exportfield);
            mcx_config.exportfield = nullptr;

            /** adjoint Jacobian: return in separate named output keys (jmua, jd, jmus, etc.) */
            if (MCX_IS_ADJOINT_TYPE(mcx_config.outputtype) && mcx_config.exportjacob) {
                int nsrcpairs = (int)mcx_jacchannels(&mcx_config); /* Ns*Nd, or the rows of cfg['sdmap'] */
                int isdual    = MCX_IS_DUAL_ADJOINT_TYPE(mcx_config.outputtype);
                int isrf      = (mcx_config.omega > 0.f) ? 1 : 0;
                size_t adjlen = (size_t)mcx_config.dim.x * mcx_config.dim.y * mcx_config.dim.z * nsrcpairs;
                std::vector<size_t> jdims = {(size_t)mcx_config.dim.x, (size_t)mcx_config.dim.y,
                                             (size_t)mcx_config.dim.z, (size_t)nsrcpairs
                                            };

                const char* jname1 = "jmua";
                const char* jname2 = "jd";

                switch (mcx_config.outputtype) {
                    case otAdjoint:
                        jname1 = "jmua";
                        break;

                    case otAdjointDcoeff:
                        jname1 = "jd";
                        break;

                    case otAdjointMus:
                        jname1 = "jmus";
                        break;

                    case otAdjointMusp:
                        jname1 = "jmusp";
                        break;

                    case otAdjointMuaD:
                        jname1 = "jmua";
                        jname2 = "jd";
                        break;

                    case otAdjointMuaMusp:
                        jname1 = "jmua";
                        jname2 = "jmusp";
                        break;

                    default:
                        break;
                }

                /* exportjacob layout: CW non-dual:[re1]; RF non-dual:[re1,im1];
                 * CW dual:[re1,re2]; RF dual:[re1,re2,im1,im2] each of size adjlen */
                float* re1 = mcx_config.exportjacob;
                float* re2 = isdual           ? mcx_config.exportjacob + adjlen             : NULL;
                float* im1 = isrf             ? mcx_config.exportjacob + (isdual ? 2 : 1) * adjlen : NULL;
                float* im2 = (isrf && isdual) ? mcx_config.exportjacob + 3 * adjlen         : NULL;

                auto add_jac_field = [&](const char* fname, float * re_data, float * im_data) {
                    if (!isrf) {
                        auto jfield = py::array_t<float, py::array::f_style>(jdims);
                        memcpy(jfield.mutable_data(), re_data, adjlen * sizeof(float));
                        output[fname] = jfield;
                    } else {
                        auto jfield = py::array_t<std::complex<float>, py::array::f_style>(jdims);
                        auto* cptr = static_cast<std::complex<float>*>(jfield.mutable_data());

                        for (size_t ii = 0; ii < adjlen; ii++) {
                            cptr[ii] = std::complex<float>(re_data[ii], im_data[ii]);
                        }

                        output[fname] = jfield;
                    }
                };

                add_jac_field(jname1, re1, im1);

                if (isdual) {
                    add_jac_field(jname2, re2, im2);
                }

                free(mcx_config.exportjacob);
                mcx_config.exportjacob = nullptr;
            }

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

        // Stat dictionary output, returned even when the fluence is not saved (issave2pt=0), as mcxlab does
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
            << "PMCX (" MCX_VERSION ")\nUsage:\n    output = pmcx.run(cfg);\n\nRun 'help(pmcx.run)' for more details.\n";
}

/**
 * @brief Force matlab refresh the command window to print all buffered messages
 */

extern "C" void mcx_python_flush() {
    std::cout.flush();
}

/** prints the log of cfg.flog=2 (stderr, the default) to Python's sys.stdout, as mexPrintf does in
    MATLAB: the log then shows in Jupyter, in order with Python's own output. Called by the thread
    that holds the GIL (OpenMP thread 0, see MCX_FPRINTF); without a Python stdout it uses stderr */
extern "C" int mcx_python_printf(const char* format, ...) {
    va_list args, args2;
    va_start(args, format);
    va_copy(args2, args);
    int len = vsnprintf(NULL, 0, format, args);
    va_end(args);

    if (len < 0) {
        va_end(args2);
        return len;
    }

    std::vector<char> buf(len + 1);
    vsnprintf(buf.data(), buf.size(), format, args2);
    va_end(args2);

    try {
        py::gil_scoped_acquire gil;
        py::object out = py::module_::import("sys").attr("stdout");

        if (out.is_none()) {
            throw std::runtime_error("no sys.stdout");
        }

        out.attr("write")(py::str(buf.data(), len));
        out.attr("flush")();
    } catch (...) {
        fputs(buf.data(), stderr);
    }

    return len;
}

py::dict pmcx_interface_wargs(py::args args, const py::kwargs& kwargs) {
    if (py::len(kwargs) == 0) {
        print_mcx_usage();
        return {};
    }

    return pmcx_interface(kwargs);
}

py::str print_version() {
    Config mcx_config;            /** mcxconfig: structure to store all simulation parameters */
    mcx_initcfg(&mcx_config);
    mcx_printheader(&mcx_config);
    mcx_clearcfg(&mcx_config);
    return py::str(MCX_VERSION);
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

/**
 * The docstring of run(), one literal per line and joined on first use: as one literal, it would
 * exceed the 4095 characters -pedantic accepts and the 16 KB limit of MSVC
 */
static const char* pmcx_run_doc_lines[] = {
    "Run an MCX simulation on CUDA GPUs.\n",
    "\n",
    "    res = pmcx.run(cfg)\n",
    "    res = pmcx.run(nphoton=1000000, vol=vol, prop=prop, ...)\n",
    "\n",
    "The settings are given as a dict, or as keyword arguments. Array values may be\n",
    "NumPy arrays (C or Fortran order) or nested lists; their shapes follow mcxlab, so\n",
    "cfg['vol'][x, y, z] is the voxel at (x, y, z) and cfg['prop'] has one row per\n",
    "medium. pmcx.mcxlab(cfg) runs this function and post-processes the outputs as\n",
    "mcxlab does (named detected-photon fields, sorted trajectories).\n",
    "\n",
    "Parameters\n",
    "----------\n",
    "cfg : dict\n",
    "    The simulation settings. Keys marked with * are required; values in [] are the\n",
    "    defaults. Lengths are in voxels unless stated otherwise; cfg['unitinmm'] gives\n",
    "    the voxel size in mm.\n",
    "\n",
    "    Domain and optical properties\n",
    "\n",
    "    * nphoton : int\n",
    "        Total number of photons to simulate (up to 2**63-1).\n",
    "    * vol : ndarray\n",
    "        A 3-D array (Nx, Ny, Nz) of tissue labels (an integer type, float32 or\n",
    "        float64), label 0 being the background. A singleton x or y dimension makes a 2-D\n",
    "        simulation. A 4-D array, whose first dimension holds per-voxel data,\n",
    "        describes continuously varying media:\n",
    "          (1, Nx, Ny, Nz) float32: per-voxel mua; mus/g/n from prop[1]\n",
    "          (2, Nx, Ny, Nz) float32: per-voxel mua and mus (stored as half floats)\n",
    "          (2, Nx, Ny, Nz) uint16: mua/mus as 0-65535 levels between prop[1], prop[2]\n",
    "          (3, Nx, Ny, Nz) float32: \"labelplus\", [value, property index 0-3, label]\n",
    "          (3, Nx, Ny, Nz) int16/uint16: \"mixlabel\", [label1, label2, label1 share]\n",
    "          (4, Nx, Ny, Nz) uint8: mua/mus/g/n as 0-255 levels between prop[1], prop[2]\n",
    "          (4, Nx, Ny, Nz) float32: per-voxel mua/mus/g/n\n",
    "          (8, Nx, Ny, Nz) uint8: split-voxel MC (SVMC), see pmcx.svmc()\n",
    "    * prop : ndarray, shape (N, 4)\n",
    "        One row per medium, [mua (1/mm), mus (1/mm), g, n]; row 0 is label 0 (the\n",
    "        background, normally [0, 0, 1, 1]), row 1 is label 1, and so on.\n",
    "    polprop : ndarray, shape (N, 5)\n",
    "        Polarized MC: one row per medium from label 1, [mua (1/mm), sphere radius\n",
    "        (micron), number density (1/micron^3), sphere index, ambient index]; needs\n",
    "        lambda.\n",
    "    unitinmm : float\n",
    "        Voxel edge length in mm [1.0].\n",
    "    shapes : str\n",
    "        A JSON string of shapes (spheres, boxes, layers ...) drawn into vol, see\n",
    "        https://mcx.space/wiki/?Shapes\n",
    "    issrcfrom0 : int\n",
    "        1: the first voxel spans [0, 1] in each axis; [0]: [1, 2].\n",
    "\n",
    "    Time gates\n",
    "\n",
    "    * tend : float\n",
    "        End of the simulated time window (s).\n",
    "    tstart : float\n",
    "        Start of the time window (s) [0].\n",
    "    tstep : float\n",
    "        Width of each time gate (s) [tend - tstart, one gate].\n",
    "    maxgate : int\n",
    "        The number of time gates simulated per run [0: all that fit in GPU memory].\n",
    "        When fewer than the gates, the photons are rerun for each group of gates,\n",
    "        and pmcx returns the raw output (isnormalized is turned off, with a warning).\n",
    "\n",
    "    Source\n",
    "\n",
    "    * srcpos : array_like\n",
    "        [x, y, z] position of the source; an optional 4th value is the initial\n",
    "        photon weight [1]. Several sources: one row per source.\n",
    "    * srcdir : array_like\n",
    "        [vx, vy, vz] unit launch direction; an optional 4th value is the focal\n",
    "        length of the area sources: 0 collimated, >0 converging, <0 diverging; or,\n",
    "        for every source type, nan for an isotropic or -inf for a Lambertian launch.\n",
    "    srctype : str\n",
    "        ['pencil'], 'isotropic', 'arcsine', 'cone', 'zgaussian', 'gaussian',\n",
    "        'hyperboloid', 'disk', 'ring', 'planar', 'pattern', 'pattern3d', 'fourier',\n",
    "        'fourierx', 'fourierx2d', 'pencilarray', 'line' or 'slit'; see the mcxlab\n",
    "        help (help mcxlab in MATLAB) for the meaning of srcparam1/srcparam2 of each.\n",
    "    srcparam1, srcparam2 : array_like\n",
    "        Four parameters each, see srctype.\n",
    "    srcpattern : ndarray\n",
    "        The intensity pattern of 'pattern' (srcparam1[3] x srcparam2[3] pixels, or\n",
    "        (srcnum, Nx, Ny) for several) and 'pattern3d' sources.\n",
    "    srcnum : int\n",
    "        The number of patterns simulated together by a 'pattern' source [1].\n",
    "    srcid : int\n",
    "        With several sources: [0] all in one output, -1 each in its own output,\n",
    "        N>0 only the N-th source.\n",
    "    angleinvcdf : array_like\n",
    "        A user-defined launch angle distribution: the inverse CDF of theta/pi,\n",
    "        increasing within [0, 1].\n",
    "    voidtime : int\n",
    "        For a source outside of the domain, [1]: start the clock at launch, 0: when\n",
    "        the photon enters the domain.\n",
    "    maxvoidstep : int\n",
    "        The farthest (voxels) a photon launched outside travels to the domain [1000].\n",
    "    isspecular : int\n",
    "        1: apply the specular reflection at entry [0].\n",
    "    srciquv : array_like\n",
    "        [I, Q, U, V] Stokes vector of the source for polarized MC.\n",
    "    lambda : float\n",
    "        Wavelength (nm) for polarized MC.\n",
    "    omega : float\n",
    "        Modulation angular frequency 2*pi*f (rad/s) for frequency-domain (RF)\n",
    "        outputs [0].\n",
    "\n",
    "    Detectors\n",
    "\n",
    "    detpos : ndarray, shape (N, 4)\n",
    "        One row per disk detector, [x, y, z, radius].\n",
    "    detdir : ndarray, shape (N, 3) or (N, 4)\n",
    "        The inward direction (and focal length) of each detector, required by the\n",
    "        adjoint outputs.\n",
    "    sdmap : ndarray, shape (Nch, 2)\n",
    "        [srcid, detid] rows (1-based) selecting the channels of the adjoint\n",
    "        Jacobians [all pairs, detector index fastest].\n",
    "    bc : str\n",
    "        Per-face boundary conditions, up to 12 letters; letters 1-6 are the -x, -y,\n",
    "        -z, +x, +y, +z faces: '_' isreflect, 'r' Fresnel, 'a' absorbing, 'm'\n",
    "        mirror, 'c' cyclic; letters 7-12, '0' or '1', make the faces detectors.\n",
    "    maxdetphoton : int\n",
    "        The most detected photons kept [1000000].\n",
    "    savedetflag : str\n",
    "        The data saved per detected photon ['dp']: 'd' detector ID, 's' scattering\n",
    "        counts, 'p' partial paths, 'm' momentum transfer, 'x' exit position, 'v'\n",
    "        exit direction, 'w' initial weight, 'i' Stokes vector.\n",
    "    issavedet : int\n",
    "        [1]: save the detected photons (when detpos is given); 0: do not; 3: stop\n",
    "        once maxdetphoton photons are detected.\n",
    "    issaveexit : int\n",
    "        1: also save the exit position and direction [0].\n",
    "    ismomentum : int\n",
    "        1: also save the momentum transfer [0].\n",
    "    issaveseed : int\n",
    "        1: save the RNG seed of each detected photon, for replay [0].\n",
    "\n",
    "    Monte Carlo settings\n",
    "\n",
    "    seed : int or ndarray\n",
    "        RNG seed [1648335518]; <=0 seeds from the clock. A uint8 array (the\n",
    "        'seeds' output of an earlier run) replays those photons.\n",
    "    respin : int\n",
    "        Repeat the simulation this many times and add up the results [1].\n",
    "    isreflect : int\n",
    "        [1]: Fresnel reflection at the domain boundary; 0: index-matched.\n",
    "    isnormalized : int\n",
    "        [1]: normalize the output to a unit source; 0: raw; 2: replay without the\n",
    "        normalization by the detected weight.\n",
    "    minenergy : float\n",
    "        Weight below which Russian roulette starts [0].\n",
    "    gscatter : int\n",
    "        Scatter isotropically after this many events [1e9].\n",
    "    invcdf : array_like\n",
    "        A user-defined phase function: the inverse CDF of cos(theta), increasing\n",
    "        within [-1, 1].\n",
    "\n",
    "    Output\n",
    "\n",
    "    outputtype : str\n",
    "        What res['flux'] holds: ['flux'] fluence rate (1/mm^2/s), 'fluence',\n",
    "        'energy', 'length'; replay: 'jacobian' (or 'wl'), 'nscat' (or 'wp'), 'wm',\n",
    "        'rf', 'rfmus', 'wltof', 'wptof'; forward with omega>0: 'rf'; adjoint\n",
    "        Jacobians (need detpos and detdir): 'adjoint', 'adjoint_dcoeff',\n",
    "        'adjoint_mus', 'adjoint_musp', 'adjoint_mua_d', 'adjoint_mua_musp'.\n",
    "    issave2pt : int\n",
    "        [1]: compute res['flux']; 0: skip it.\n",
    "    issaveref : int\n",
    "        1: save the diffuse reflectance in res['dref'] [0]; it is recorded in the\n",
    "        background voxels next to the domain.\n",
    "    dumpmask : bool\n",
    "        True: return the preprocessed label volume in res['vol'] [False].\n",
    "    session : str\n",
    "        A name for the output files.\n",
    "    flog : int or str\n",
    "        Where the messages go: [2] sys.stdout, 1 the stdout of the process, 0 the\n",
    "        same without the banner, or a file name (appended); '/dev/null' or 'NUL'\n",
    "        (either, on any OS) silences everything, the progress bar included. Errors\n",
    "        are always raised.\n",
    "    lang : str\n",
    "        Language of the log: zh_CN, zh_TW, ja_JP, fr_CA, es_MX, de_DE, ko_KR, hi_IN\n",
    "        or pt_BR.\n",
    "\n",
    "    Replay\n",
    "\n",
    "    detphotons : ndarray\n",
    "        The 'detp' output of the baseline run.\n",
    "    replaydet : int\n",
    "        [0]: replay all detectors into one output; N>0: only detector N; -1: each\n",
    "        detector in its own output.\n",
    "\n",
    "    GPU\n",
    "\n",
    "    gpuid : int or str\n",
    "        The GPU to use, from 1 [1]; a string of 0s and 1s selects several, e.g.\n",
    "        '1101'. pmcx.gpuinfo() lists them.\n",
    "    workload : array_like\n",
    "        The relative share of photons for each selected GPU.\n",
    "    autopilot : int\n",
    "        [1]: choose the thread and block counts automatically; 0: use nthread and\n",
    "        nblocksize, which makes the results reproducible.\n",
    "    nthread, nblocksize : int\n",
    "        Total threads [16384] and threads per block [64] with autopilot=0.\n",
    "    isgpuinfo : int\n",
    "        1: print the GPU information before running [0].\n",
    "\n",
    "    Debugging\n",
    "\n",
    "    debuglevel : str\n",
    "        Debug flags: 'P' progress bar, 'M' record photon trajectories, 'T'\n",
    "        trajectories only (no flux, detp or seeds), 'R' test the RNG.\n",
    "    maxjumpdebug : int\n",
    "        The most trajectory positions recorded [10000000].\n",
    "    istrajstokes : int\n",
    "        1: also record the Stokes vector along trajectories [0].\n",
    "\n",
    "Returns\n",
    "-------\n",
    "res : dict\n",
    "    flux : ndarray, float32\n",
    "        The quantity of outputtype, (Nx, Ny, Nz, Ngate), normalized as set by\n",
    "        isnormalized; an extra last dimension for several sources (srcid=-1),\n",
    "        detectors (replaydet=-1) or adjoint slots; complex64 for RF outputs. With\n",
    "        srcnum>1, the patterns are interleaved in the first axis, (srcnum*Nx, ...);\n",
    "        pmcx.mcxlab moves them to the last dimension.\n",
    "    dref : ndarray\n",
    "        The diffuse reflectance, the same shape as flux (issaveref=1).\n",
    "    jmua, jd, jmus, jmusp : ndarray\n",
    "        The adjoint Jacobians, (Nx, Ny, Nz, Nch), complex64 for RF.\n",
    "    detp : ndarray, float32, shape (ncol, ndetected)\n",
    "        The detected photons (issavedet=1 and detpos given), one column per\n",
    "        photon, with the rows of savedetflag; pmcx.detphoton() splits them into\n",
    "        named fields.\n",
    "    seeds : ndarray, uint8, shape (seedbyte, ndetected)\n",
    "        The RNG seed of each detected photon (issaveseed=1).\n",
    "    traj : ndarray, float32, shape (ncol, npos)\n",
    "        The photon trajectories (debuglevel containing 'M'), one column per\n",
    "        position, [id, x, y, z, weight, srcid, ...], the id stored as uint32 bits.\n",
    "    vol : ndarray, uint32\n",
    "        The preprocessed label volume (dumpmask=True).\n",
    "    prop : ndarray\n",
    "        The optical properties computed from polprop (polarized MC).\n",
    "    stat : dict\n",
    "        runtime (ms), nphoton, energytot, energyabs, normalizer, unitinmm and\n",
    "        workload.\n",
    "\n",
    "Raises\n",
    "------\n",
    "ValueError\n",
    "    If a setting is missing or invalid.\n",
    "RuntimeError\n",
    "    If the simulation fails.\n",
    "\n",
    "Examples\n",
    "--------\n",
    ">>> import numpy as np, pmcx\n",
    ">>> res = pmcx.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype=\"uint8\"),\n",
    "...                tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30, 30, 0],\n",
    "...                srcdir=[0, 0, 1], prop=[[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]])\n",
    ">>> res[\"flux\"].shape\n",
    "(60, 60, 60, 1)\n",
    nullptr
};

static const char* pmcx_run_doc() {
    static std::string doc;

    if (doc.empty()) {
        for (const char** line = pmcx_run_doc_lines; *line; line++) {
            doc += *line;
        }
    }

    return doc.c_str();
}

PYBIND11_MODULE(_pmcx, m) {
    m.doc() = "PMCX (" MCX_VERSION "): Python bindings for Monte Carlo eXtreme photon transport simulator, https://mcx.space";
    m.def("run", &pmcx_interface, pmcx_run_doc(), py::call_guard<py::scoped_ostream_redirect,
          py::scoped_estream_redirect>());
    m.def("run", &pmcx_interface_wargs, "The same as run(cfg), with the settings given as keyword arguments.", py::call_guard<py::scoped_ostream_redirect,
          py::scoped_estream_redirect>());
    m.def("gpuinfo",
          &get_GPU_info,
          "List the CUDA GPUs.\n\n"
          "Returns\n-------\nlist of dict\n"
          "    One dict per device: name, id, devcount, major, minor, globalmem, constmem, sharedmem,\n"
          "    regcount, clock, sm, core, autoblock, autothread and maxgate.",
          py::call_guard<py::scoped_ostream_redirect,
          py::scoped_estream_redirect>());
    m.def("version",
          &print_version,
          "Print the MCX banner and return the version string.\n\n"
          "Returns\n-------\nstr\n    The version, e.g. 'v2026'.",
          py::call_guard<py::scoped_ostream_redirect,
          py::scoped_estream_redirect>());
}

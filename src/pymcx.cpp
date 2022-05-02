#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_const.h"
#include "mcx_shapes.h"
#include <pybind11/common.h>
#include <pybind11/iostream.h>
#include "interface-common.h"

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

float *detps = nullptr;         //! buffer to receive data from cfg.detphotons field
int    dimdetps[2] = {0,0};  //! dimensions of the cfg.detphotons array
int    seedbyte = 0;


#define GET_SCALAR_FIELD(src, dst, prop, type) if (src.contains(#prop)) {dst.prop = py::reinterpret_borrow<type>(src[#prop]); std::cout << #prop << ": " << dst.prop << std::endl;}

#define GET_VEC3_FIELD(src, dst, prop, type) if (src.contains(#prop)) {auto list = py::reinterpret_borrow<py::list>(src[#prop]);\
                                             dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>()};\
                                             std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << "]\n";}

#define GET_VEC4_FIELD(src, dst, prop, type) if (src.contains(#prop)) {auto list = py::reinterpret_borrow<py::list>(src[#prop]);\
                                             dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list[3].cast<type>()}; \
                                             std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z << ", " << dst.prop.w << "]\n";}


#define GET_VEC34_FIELD(src, dst, prop, type) if (src.contains(#prop)) {auto list = py::reinterpret_borrow<py::list>(src[#prop]);\
                                             dst.prop = {list[0].cast<type>(), list[1].cast<type>(), list[2].cast<type>(), list.size() == 4 ? list[3].cast<type>() : 1}; \
                                             std::cout << #prop << ": [" << dst.prop.x << ", " << dst.prop.y << ", " << dst.prop.z;\
                                             (list.size() == 4 ? (std::cout << ", " << dst.prop.w << "]") :  std::cout << "]") << "]\n";}


void parseVolume(const py::dict& userCfg, Config& mcxConfig) {
  if (!userCfg.contains("vol"))
    throw py::value_error("Configuration must specify a 2/3/4D volume.");
  auto volumeHandle = userCfg["vol"];
  if (py::array_t<int8_t, py::array::c_style>::check_(volumeHandle)) {
    auto fStyleVolume = py::array_t<int8_t, py::array::f_style>::ensure(volumeHandle);
    auto buffer = fStyleVolume.request();
    int i = buffer.shape.size() == 4;
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                     static_cast<unsigned int>(buffer.shape.at(i + 1)),
                     static_cast<unsigned int>(buffer.shape.at(i + 2))};
    if (i == 1)
      mcxConfig.mediabyte = buffer.shape.at(0) == 4 ? MEDIA_ASGN_BYTE : MEDIA_2LABEL_SPLIT;
    else {
      mcxConfig.mediabyte = 1;
      mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
      for (i = 0; i < buffer.size; i++)
        mcxConfig.vol[i] = static_cast<unsigned char *>(buffer.ptr)[i];
    }
  }
  else if (py::array_t<int16_t, py::array::c_style>::check_(userCfg["vol"])) {
    auto fStyleVolume = py::array_t<int16_t, py::array::f_style>::ensure(userCfg["vol"]);
    auto buffer = fStyleVolume.request();
    int i = buffer.shape.size() == 4;
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                     static_cast<unsigned int>(buffer.shape.at(i + 1)),
                     static_cast<unsigned int>(buffer.shape.at(i + 2))};
    if (i == 1)
      mcxConfig.mediabyte = buffer.shape.at(0) == 3 ? MEDIA_2LABEL_MIX : MEDIA_AS_SHORT;
    else {
      mcxConfig.mediabyte = 2;
      mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
      for (i = 0; i < buffer.size; i++)
        mcxConfig.vol[i] = static_cast<unsigned short *>(buffer.ptr)[i];
    }
  }
  else if (py::array_t<int32_t, py::array::c_style>::check_(userCfg["vol"])) {
    auto fStyleVolume = py::array_t<int32_t, py::array::f_style>::ensure(userCfg["vol"]);
    mcxConfig.mediabyte = 4;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for int32_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
    memcpy(mcxConfig.vol, buffer.ptr, buffer.size * sizeof(unsigned int));
  }
  else if (py::array_t<u_int8_t, py::array::c_style>::check_(userCfg["vol"])) {
    auto fStyleVolume = py::array_t<u_int8_t, py::array::f_style>::ensure(userCfg["vol"]);
    mcxConfig.mediabyte = 1;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for uint8_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
    for (int i = 0; i < buffer.size; i++)
      mcxConfig.vol[i] = static_cast<unsigned char *>(buffer.ptr)[i];
  }
  else if (py::array_t<u_int16_t, py::array::c_style>::check_(userCfg["vol"])) {
    auto fStyleVolume = py::array_t<u_int16_t, py::array::f_style>::ensure(userCfg["vol"]);
    mcxConfig.mediabyte = 2;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for u_int16_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
    for (int i = 0; i < buffer.size; i++)
      mcxConfig.vol[i] = static_cast<unsigned short *>(buffer.ptr)[i];
  }
  else if (py::array_t<u_int32_t, py::array::c_style>::check_(userCfg["vol"])) {
    auto fStyleVolume = py::array_t<u_int32_t, py::array::f_style>::ensure(userCfg["vol"]);
    mcxConfig.mediabyte = 8;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for u_int32_t volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
    memcpy(mcxConfig.vol, buffer.ptr, buffer.size * sizeof(unsigned int));
  }
  else if (py::array_t<float, py::array::c_style>::check_(userCfg["vol"])) {
    auto fStyleVolume = py::array_t<float, py::array::f_style>::ensure(userCfg["vol"]);
    auto buffer = fStyleVolume.request();
    int i = buffer.shape.size() == 4;
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(i)),
                     static_cast<unsigned int>(buffer.shape.at(i + 1)),
                     static_cast<unsigned int>(buffer.shape.at(i + 2))};
    if (i) {
      switch (buffer.shape.at(0)) {
        case 3:
          mcxConfig.mediabyte = MEDIA_LABEL_HALF;
          break;
        case 2:
          mcxConfig.mediabyte = MEDIA_AS_F2H;
          break;
        case 1:
          mcxConfig.mediabyte = MEDIA_MUA_FLOAT;
          break;
        default:
          throw py::value_error("Invalid array for vol array.");
      }
    }
    else {
      mcxConfig.mediabyte = 14;
      mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
      for (i = 0; i < buffer.size; i++)
        mcxConfig.vol[i] = static_cast<float *>(buffer.ptr)[i];
    }
  }
  else if (py::array_t<double, py::array::c_style>::check_(userCfg["vol"])) {
    auto fStyleVolume = py::array_t<double, py::array::f_style>::ensure(userCfg["vol"]);
    mcxConfig.mediabyte = 4;
    auto buffer = fStyleVolume.request();
    if (buffer.shape.size() == 4)
      throw py::value_error("Invalid volume dims for double volume.");
    mcxConfig.dim = {static_cast<unsigned int>(buffer.shape.at(0)),
                     static_cast<unsigned int>(buffer.shape.at(1)),
                     static_cast<unsigned int>(buffer.shape.at(2))};
    mcxConfig.vol = static_cast<unsigned int *>(malloc(buffer.size * sizeof(unsigned int)));
    for (int i = 0; i < buffer.size; i++)
      mcxConfig.vol[i] = static_cast<double *>(buffer.ptr)[i];
  }
  else
    throw py::value_error("Invalid data type for vol array.");
  std::cout << "End of volume assignment" << std::endl;
}


void parseConfig(const py::dict& userCfg, Config& mcxConfig) {
  mcx_initcfg(&mcxConfig);

  mcxConfig.flog = stderr;
  GET_SCALAR_FIELD(userCfg, mcxConfig, nphoton, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, nblocksize, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, nthread, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, tstart, py::float_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, tstep, py::float_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, tend, py::float_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, maxdetphoton, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, sradius, py::float_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, maxgate, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, respin, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, isreflect, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, isref3, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, isrefint, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, isnormalized, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, isref3, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, issrcfrom0, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, autopilot, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, minenergy, py::float_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, unitinmm, py::float_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, printnum, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, voidtime, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, issaveseed, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, issaveref, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, issaveexit, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, ismomentum, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, isspecular, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, replaydet, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, faststep, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, maxvoidstep, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, maxjumpdebug, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, gscatter, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, srcnum, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, omega, py::float_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, issave2pt, py::int_);
  GET_SCALAR_FIELD(userCfg, mcxConfig, lambda, py::float_);
  GET_VEC3_FIELD(userCfg, mcxConfig, srcpos, float);
  GET_VEC34_FIELD(userCfg, mcxConfig, srcdir, float);
  GET_VEC3_FIELD(userCfg, mcxConfig, steps, float);
  GET_VEC3_FIELD(userCfg, mcxConfig, crop0, uint);
  GET_VEC3_FIELD(userCfg, mcxConfig, crop1, uint);
  GET_VEC4_FIELD(userCfg, mcxConfig, srcparam1, float);
  GET_VEC4_FIELD(userCfg, mcxConfig, srcparam2, float);
  GET_VEC4_FIELD(userCfg, mcxConfig, srciquv, float);
  std::cout << "Parsing volume:" << std::endl;
  parseVolume(userCfg, mcxConfig);
  std::cout << "Parsed volume" << std::endl;

  if (userCfg.contains("detpos")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(userCfg["detpos"]);
    auto bufferInfo = fStyleVolume.request();
    if (bufferInfo.shape.at(0) > 0 && bufferInfo.shape.at(1) != 4)
      throw py::value_error("the 'detpos' field must have 4 columns (x,y,z,radius)");
    mcxConfig.detnum = bufferInfo.shape.at(0);
    if(mcxConfig.detpos) free(mcxConfig.detpos);
    mcxConfig.detpos = (float4 *)malloc(mcxConfig.detnum * sizeof(float4));
    auto val = static_cast<float *>(bufferInfo.ptr);
    for(int j = 0; j < 4; j++)
      for(int i = 0; i < mcxConfig.detnum; i++)
        ((float *)(&mcxConfig.detpos[i]))[j] = val[j * mcxConfig.detnum + i];
  }
  if (userCfg.contains("prop")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(userCfg["prop"]);
    auto bufferInfo = fStyleVolume.request();
    if (bufferInfo.shape.at(0) > 0 && bufferInfo.shape.at(1) != 4)
      throw py::value_error("the 'prop' field must have 4 columns (mua,mus,g,n)");
    mcxConfig.medianum = bufferInfo.shape.at(0);
    if(mcxConfig.prop) free(mcxConfig.prop);
    mcxConfig.prop = (Medium *)malloc(mcxConfig.medianum * sizeof(Medium));
    auto val = static_cast<float*>(bufferInfo.ptr);
    for(int j = 0; j < 4; j++)
      for(int i = 0; i < mcxConfig.medianum; i++)
        ((float *)(&mcxConfig.prop[i]))[j]=val[j * mcxConfig.medianum + i];
  }
  if (userCfg.contains("polprop")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(userCfg["polprop"]);
    auto bufferInfo = fStyleVolume.request();
    if(bufferInfo.shape.size() != 2)
      throw py::value_error("the 'polprop' field must a 2D array");
    if(bufferInfo.shape.at(0) > 0 && bufferInfo.shape.at(1) != 5)
      throw py::value_error("the 'polprop' field must have 5 columns (mua,radius,rho,n_sph,n_bkg");
    mcxConfig.polmedianum = bufferInfo.shape.at(0);
    if(mcxConfig.polprop) free(mcxConfig.polprop);
    mcxConfig.polprop = (POLMedium *)malloc(mcxConfig.polmedianum * sizeof(POLMedium));
    auto val = static_cast<float*>(bufferInfo.ptr);
    for(int j = 0; j < 5;j++)
      for(int i = 0; i < mcxConfig.polmedianum; i++)
        ((float *)(&mcxConfig.polprop[i]))[j] = val[j *  mcxConfig.polmedianum + i];
  }
  if(userCfg.contains("session")) {
    std::string session = py::str(userCfg["session"]);
    if (session.empty())
      throw py::value_error("the 'session' field must be a non-empty string");
    if (session.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'session' field is too long");
    strncpy(mcxConfig.session, session.c_str(), MAX_SESSION_LENGTH);
  }
  if(userCfg.contains("srctype")) {
    std::string srcType = py::str(userCfg["srctype"]);
    const char *srctypeid[]={"pencil","isotropic","cone","gaussian","planar",
                             "pattern","fourier","arcsine","disk","fourierx","fourierx2d","zgaussian",
                             "line","slit","pencilarray","pattern3d","hyperboloid",""};
    char strtypestr[MAX_SESSION_LENGTH]={'\0'};

    if(srcType.empty())
      throw py::value_error("the 'srctype' field must be a non-empty string");
    if (srcType.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'srctype' field is too long");
    strncpy(strtypestr, srcType.c_str(), MAX_SESSION_LENGTH);
    mcxConfig.srctype = mcx_keylookup(strtypestr, srctypeid);
    if(mcxConfig.srctype==-1)
      throw py::value_error("the specified source type is not supported");
  }
  if(userCfg.contains("outputtype")) {
    std::string outputType = py::str(userCfg["outputtype"]);
    const char *outputtype[]={"flux","fluence","energy","jacobian","nscat","wl","wp","wm","rf",""};
    char outputstr[MAX_SESSION_LENGTH]={'\0'};
    if(outputType.empty())
      throw py::value_error("the 'srctype' field must be a non-empty string");
    if (outputType.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'srctype' field is too long");
    strncpy(outputstr, outputType.c_str(), MAX_SESSION_LENGTH);
    mcxConfig.outputtype = mcx_keylookup(outputstr, outputtype);
    if(mcxConfig.outputtype >= 5) // map wl to jacobian, wp to nscat
      mcxConfig.outputtype -= 2;
    if(mcxConfig.outputtype==-1)
      throw py::value_error("the specified output type is not supported");
  }
  if (userCfg.contains("debuglevel")) {
    std::string debugLevel = py::str(userCfg["debuglevel"]);
    const char debugflag[]={'R','M','P','\0'};
    char debuglevel[MAX_SESSION_LENGTH]={'\0'};
    if (debugLevel.empty())
      throw py::value_error("the 'debuglevel' field must be a non-empty string");
    if (debugLevel.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'debuglevel' field is too long");
    strncpy(debuglevel, debugLevel.c_str(), MAX_SESSION_LENGTH);
    mcxConfig.debuglevel = mcx_parsedebugopt(debuglevel, debugflag);
    if(mcxConfig.debuglevel==0)
      throw py::value_error("the specified debuglevel is not supported");
  }
  if(userCfg.contains("savedetflag")) {
    std::string saveDetFlag = py::str(userCfg["savedetflag"]);
    const char saveflag[]={'D','S','P','M','X','V','W','I','\0'};
    char savedetflag[MAX_SESSION_LENGTH]={'\0'};
    if (saveDetFlag.empty())
      throw py::value_error("the 'savedetflag' field must be a non-empty string");
    if (saveDetFlag.size() > MAX_SESSION_LENGTH)
      throw py::value_error("the 'savedetflag' field is too long");
    strncpy(savedetflag, saveDetFlag.c_str(), MAX_SESSION_LENGTH);
    mcxConfig.savedetflag = mcx_parsedebugopt(savedetflag, saveflag);
  }
  if(userCfg.contains("srcpattern")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(userCfg["srcpattern"]);
    auto bufferInfo = fStyleVolume.request();
    mcxConfig.srcpattern = static_cast<float*>(bufferInfo.ptr);
  }
  if(userCfg.contains("invcdf")) {
    auto fStyleVolume = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(userCfg["invcdf"]);
    auto bufferInfo = fStyleVolume.request();
    unsigned int nphase = bufferInfo.shape.size();
    float* val = static_cast<float*>(bufferInfo.ptr);
    mcxConfig.nphase = nphase + 2;
    mcxConfig.nphase += (mcxConfig.nphase & 0x1); // make cfg.nphase even number
    mcxConfig.invcdf = (float*)calloc(mcxConfig.nphase, sizeof(float));
    for(int i = 0; i < nphase; i++){
      mcxConfig.invcdf[i + 1] = val[i];
      if(i > 0 && (val[i] < val[i - 1] || (val[i] > 1.f || val[i] < -1.f)))
        throw py::value_error("cfg.invcdf contains invalid data; it must be a monotonically increasing vector with all values between -1 and 1");
    }
    mcxConfig.invcdf[0] = -1.f;
    mcxConfig.invcdf[nphase + 1] = 1.f;
    mcxConfig.invcdf[mcxConfig.nphase - 1] = 1.f;
  }
  if (userCfg.contains("shapes")) {
    std::string shapesString = py::str(userCfg["shapes"]);
    if (shapesString.empty())
      throw py::value_error("the 'shapes' field must be a non-empty string");
    mcxConfig.shapedata = (char *)calloc(shapesString.size() + 2, 1);
    strncpy(mcxConfig.shapedata, shapesString.c_str(), shapesString.size() + 1);
  }
  if(userCfg.contains("bc")) {
    std::string bcString = py::str(userCfg["bc"]);
    if(bcString.empty() || bcString.size() > 12)
      throw py::value_error("the 'bc' field must be a non-empty string / have less than 12 characters.");
    strncpy(mcxConfig.bc, bcString.c_str(), bcString.size() + 1);
    mcxConfig.bc[bcString.size()] = '\0';
  }
  if(userCfg.contains("seed")) {
    auto seedValue = userCfg["seed"];
    // If the seed value is scalar (int or float), then assign it directly
    if (py::int_::check_(seedValue))
      mcxConfig.seed = py::int_(seedValue);
    else if (py::float_::check_(seedValue))
      mcxConfig.seed = py::float_(seedValue).cast<int>();
    // Set seed from array
    else {
      auto fStyleArray = py::array_t<uint8_t, py::array::f_style | py::array::forcecast>::ensure(seedValue);
      auto bufferInfo = fStyleArray.request();
      seedbyte = bufferInfo.shape.at(0);
      if(bufferInfo.shape.at(0) != sizeof(float) * RAND_WORD_LEN)
        throw py::value_error("the row number of cfg.seed does not match RNG seed byte-length");
      mcxConfig.replay.seed = malloc(bufferInfo.size);
      memcpy(mcxConfig.replay.seed, bufferInfo.ptr, bufferInfo.size);
      mcxConfig.seed = SEED_FROM_FILE;
      mcxConfig.nphoton = bufferInfo.shape.at(1);
    }
  }
  if(userCfg.contains("gpuid")) {
    auto gpuIdValue = userCfg["gpuid"];
    if (py::int_::check_(gpuIdValue)) {
      mcxConfig.gpuid = py::int_(gpuIdValue);
      memset(mcxConfig.deviceid, 0, MAX_DEVICE);
      if (mcxConfig.gpuid > 0 && mcxConfig.gpuid < MAX_DEVICE) {
        memset(mcxConfig.deviceid, '0', mcxConfig.gpuid - 1);
        mcxConfig.deviceid[mcxConfig.gpuid - 1] = '1';
      } else
        throw py::value_error("GPU id must be positive and can not be more than 256");
    } else if (py::str::check_(gpuIdValue)) {
      std::string gpuIdStringValue = py::str(gpuIdValue);
      if (gpuIdStringValue.empty())
        throw py::value_error("the 'gpuid' field must be an integer or non-empty string");
      if (gpuIdStringValue.size() > MAX_DEVICE)
        throw py::value_error("the 'gpuid' field is too long");
      strncpy(mcxConfig.deviceid, gpuIdStringValue.c_str(), MAX_DEVICE);
    }
    for (int i = 0; i < MAX_DEVICE; i++)
      if (mcxConfig.deviceid[i] == '0')
        mcxConfig.deviceid[i] = '\0';
  }
  if(userCfg.contains("workload")) {
    auto workloadValue = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(userCfg["workload"]);
    auto bufferInfo = workloadValue.request();
    if (bufferInfo.shape.size() < 2 && bufferInfo.size > MAX_DEVICE)
      throw py::value_error("the workload list can not be longer than 256");
    for(int i = 0; i < bufferInfo.size; i++)
      mcxConfig.workload[i] = static_cast<float*>(bufferInfo.ptr)[i];
  }
}

//  unsigned int dimxyz = mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z;
//
//  if(mcxConfig.mediabyte==4 || (mcxConfig.mediabyte>100 && mcxConfig.mediabyte!=MEDIA_MUA_FLOAT))
//      memcpy(mcxConfig.vol, mxGetData(item),dimxyz*sizeof(unsigned int));
//  else {
//      if(cfg->mediabyte==1){
//        unsigned char *val=(unsigned char *)mxGetPr(item);
//        for(i=0;i<dimxyz;i++)
//          cfg->vol[i]=val[i];
//      }else if(cfg->mediabyte==2){
//        unsigned short *val=(unsigned short *)mxGetPr(item);
//        for(i=0;i<dimxyz;i++)
//          cfg->vol[i]=val[i];
//      }else if(cfg->mediabyte==8){
//        double *val=(double *)mxGetPr(item);
//        for(i=0;i<dimxyz;i++)
//          cfg->vol[i]=val[i];
//        cfg->mediabyte=4;
//      }else if(cfg->mediabyte==14){
//        float *val=(float *)mxGetPr(item);
//        for(i=0;i<dimxyz;i++)
//          cfg->vol[i]=val[i];
//      }else if(cfg->mediabyte==MEDIA_MUA_FLOAT){
//        union{
//          float f;
//          uint  i;
//        } f2i;
//        float *val=(float *)mxGetPr(item);
//        for(i=0;i<dimxyz;i++){
//          f2i.f=val[i];
//          if(f2i.i==0) /*avoid being detected as a 0-label voxel*/
//            f2i.f=EPS;
//          if(val[i]!=val[i]) /*if input is nan in continuous medium, convert to 0-voxel*/
//            f2i.i=0;
//          cfg->vol[i]=f2i.i;
//        }
//      }else if(cfg->mediabyte==MEDIA_AS_F2H){
//        float *val=(float *)mxGetPr(item);
//        union{
//          float f[2];
//          unsigned int i[2];
//          unsigned short h[2];
//        } f2h;
//        unsigned short tmp,m;
//        for(i=0;i<dimxyz;i++){
//          f2h.f[0]=val[i<<1];
//          f2h.f[1]=val[(i<<1)+1];
//          if(f2h.f[0]!=f2h.f[0] || f2h.f[1]!=f2h.f[1]){ /*if one of mua/mus is nan in continuous medium, convert to 0-voxel*/
//            cfg->vol[i]=0;
//            continue;
//          }
//          /**
//          float to half conversion
//          https://stackoverflow.com/questions/3026441/float32-to-float16/5587983#5587983
//          https://gamedev.stackexchange.com/a/17410  (for denorms)
//          */
//          m = ((f2h.i[0] >> 13) & 0x03ff);
//          tmp = (f2h.i[0] >> 23) & 0xff; /*exponent*/
//          tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
//          if(m<0x10 && tmp==0){ /*handle denorms - between 2^-24 and 2^-14*/
//            unsigned short sign = (f2h.i[0] >> 16) & 0x8000;
//            tmp = ((f2h.i[0] >> 23) & 0xff);
//            m = (f2h.i[0] >> 12) & 0x07ff;
//            m |= 0x0800u;
//            f2h.h[0] = sign | ((m >> (114 - tmp)) + ((m >> (113 - tmp)) & 1));
//          }else{
//            f2h.h[0] = (f2h.i[0] >> 31) << 5;
//            f2h.h[0] = (f2h.h[0] | tmp) << 10;
//            f2h.h[0] |= (f2h.i[0] >> 13) & 0x3ff;
//          }
//
//          m = ((f2h.i[1] >> 13) & 0x03ff);
//          tmp = (f2h.i[1] >> 23) & 0xff; /*exponent*/
//          tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
//          if(m<0x10 && tmp==0){ /*handle denorms - between 2^-24 and 2^-14*/
//            unsigned short sign = (f2h.i[1] >> 16) & 0x8000;
//            tmp = ((f2h.i[1] >> 23) & 0xff);
//            m = (f2h.i[1] >> 12) & 0x07ff;
//            m |= 0x0800u;
//            f2h.h[1] = sign | ((m >> (114 - tmp)) + ((m >> (113 - tmp)) & 1));
//          }else{
//            f2h.h[1] = (f2h.i[1] >> 31) << 5;
//            f2h.h[1] = (f2h.h[1] | tmp) << 10;
//            f2h.h[1] |= (f2h.i[1] >> 13) & 0x3ff;
//          }
//
//          if(f2h.i[0]==0) /*avoid being detected as a 0-label voxel, setting mus=EPS_fp16*/
//            f2h.i[0]=0x00010000;
//
//          cfg->vol[i]=f2h.i[0];
//        }
//      }else if(cfg->mediabyte==MEDIA_LABEL_HALF){
//        float *val=(float *)mxGetPr(item);
//        union{
//          float f[3];
//          unsigned int i[3];
//          unsigned short h[2];
//          unsigned char c[4];
//        } f2bh;
//        unsigned short tmp;
//        for(i=0;i<dimxyz;i++){
//          f2bh.f[2]=val[i*3];
//          f2bh.f[1]=val[i*3+1];
//          f2bh.f[0]=val[i*3+2];
//
//          if(f2bh.f[1]<0.f || f2bh.f[1]>=4.f || f2bh.f[0]<0.f )
//            mexErrMsgTxt("the 2nd volume must have an integer value between 0 and 3");
//
//          f2bh.h[0]=( (((unsigned char)(f2bh.f[1]) & 0x3) << 14) | (unsigned short)(f2bh.f[0]) );
//
//          f2bh.h[1] = (f2bh.i[2] >> 31) << 5;
//          tmp = (f2bh.i[2] >> 23) & 0xff;
//          tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
//          f2bh.h[1] = (f2bh.h[1] | tmp) << 10;
//          f2bh.h[1] |= (f2bh.i[2] >> 13) & 0x3ff;
//
//          cfg->vol[i]=f2bh.i[0];
//        }
//      }else if(cfg->mediabyte==MEDIA_2LABEL_MIX){
//        unsigned short *val=(unsigned short *)mxGetPr(item);
//        union{
//          unsigned short h[2];
//          unsigned char  c[4];
//          unsigned int   i[1];
//        } f2bh;
//        unsigned short tmp;
//        for(i=0;i<dimxyz;i++){
//          f2bh.c[0]=val[i*3]   & 0xFF;
//          f2bh.c[1]=val[i*3+1] & 0xFF;
//          f2bh.h[1]=val[i*3+2] & 0x7FFF;
//          cfg->vol[i]=f2bh.i[0];
//        }
//      }else if(cfg->mediabyte==MEDIA_2LABEL_SPLIT){
//        unsigned char *val=(unsigned char *)mxGetPr(item);
//        if(cfg->vol)
//          free(cfg->vol);
//        cfg->vol=static_cast<unsigned int *>(malloc(dimxyz<<3));
//        memcpy(cfg->vol, val, (dimxyz<<3));
//      }
//    }
//};






/**
 * @brief Validate all input fields, and warn incompatible inputs
 *
 * Perform self-checking and raise exceptions or warnings when input error is detected
 *
 * @param[in,out] cfg: the simulation configuration structure
 */

void validateConfig(Config *cfg){
  int i,gates,idx1d, isbcdet=0;
  const char boundarycond[]={'_','r','a','m','c','\0'};
  const char boundarydetflag[]={'0','1','\0'};
  unsigned int partialdata=(cfg->medianum-1)*(SAVE_NSCAT(cfg->savedetflag)+SAVE_PPATH(cfg->savedetflag)+SAVE_MOM(cfg->savedetflag));
  unsigned int hostdetreclen=partialdata+SAVE_DETID(cfg->savedetflag)+3*(SAVE_PEXIT(cfg->savedetflag)+SAVE_VEXIT(cfg->savedetflag))+SAVE_W0(cfg->savedetflag);
  hostdetreclen+=cfg->polmedianum?(4*SAVE_IQUV(cfg->savedetflag)):0; // for polarized photon simulation

  if(!cfg->issrcfrom0){
    cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
  }
  if(cfg->tstart>cfg->tend || cfg->tstep==0.f){
    std::cerr << "incorrect time gate settings" << std::endl;
  }
  if(ABS(cfg->srcdir.x*cfg->srcdir.x+cfg->srcdir.y*cfg->srcdir.y+cfg->srcdir.z*cfg->srcdir.z - 1.f)>1e-5)
    std::cerr << "field 'srcdir' must be a unitary vector" << std::endl;
  if(cfg->steps.x==0.f || cfg->steps.y==0.f || cfg->steps.z==0.f)
    std::cerr << "field 'steps' can not have zero elements" << std::endl;
  if(cfg->tend<=cfg->tstart)
    std::cerr << "field 'tend' must be greater than field 'tstart'" << std::endl;
  gates = (int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
  if(cfg->maxgate>gates)
    cfg->maxgate=gates;
  if(cfg->sradius>0.f){
    cfg->crop0.x=MAX((int)(cfg->srcpos.x-cfg->sradius),0);
    cfg->crop0.y=MAX((int)(cfg->srcpos.y-cfg->sradius),0);
    cfg->crop0.z=MAX((int)(cfg->srcpos.z-cfg->sradius),0);
    cfg->crop1.x=MIN((int)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
    cfg->crop1.y=MIN((int)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
    cfg->crop1.z=MIN((int)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
  }else if(cfg->sradius==0.f){
    memset(&(cfg->crop0),0,sizeof(uint3));
    memset(&(cfg->crop1),0,sizeof(uint3));
  }else{
    /*
        if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
    */
    if(!cfg->issrcfrom0){
      cfg->crop0.x--;cfg->crop0.y--;cfg->crop0.z--;  /*convert to C index*/
      cfg->crop1.x--;cfg->crop1.y--;cfg->crop1.z--;
    }
  }

  if(cfg->seed<0 && cfg->seed!=SEED_FROM_FILE) cfg->seed=time(NULL);
  if((cfg->outputtype==otJacobian || cfg->outputtype==otWP || cfg->outputtype==otDCS || cfg->outputtype==otRF) && cfg->seed!=SEED_FROM_FILE)
    std::cerr << "Jacobian output is only valid in the reply mode. Please define cfg.seed" << std::endl;
  for(i=0;i<cfg->detnum;i++){
    if(!cfg->issrcfrom0){
      cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
    }
  }
  if(cfg->shapedata && strstr(cfg->shapedata,":")!=NULL){
    if(cfg->mediabyte>4){
      std::cerr << "rasterization of shapes must be used with label-based mediatype" << std::endl;
    }
    Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},0};
    if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
    int status=mcx_parse_shapestring(&grid,cfg->shapedata);
    if(status){
      std::cerr << mcx_last_shapeerror() << std::endl;
    }
  }
  mcx_preprocess(cfg);

  cfg->his.maxmedia=cfg->medianum-1; /*skip medium 0*/
  cfg->his.detnum=cfg->detnum;
  cfg->his.srcnum=cfg->srcnum;
  cfg->his.colcount=hostdetreclen; /*column count=maxmedia+2*/
  cfg->his.savedetflag=cfg->savedetflag;
  mcx_replay_prep(cfg, detps, dimdetps, seedbyte, [](const char* msg){throw py::value_error(msg);});
}


py::dict pyMcxInterface(const py::dict& userCfg) {
  unsigned int partial_data, hostdetreclen;
  Config  mcx_config;  /* mcx_config: structure to store all simulation parameters */
  GPUInfo *gpu_info = nullptr;        /** gpuInfo: structure to store GPU information */
  unsigned int active_dev = 0;     /** activeDev: count of total active GPUs to be used */
  int     error_flag = 0;
  int     threadid = 0;
  size_t fielddim[6];
  py::dict output;
  try {
    /*
     * To start an MCX simulation, we first create a simulation configuration and set all elements to its default settings.
     */
    parseConfig(userCfg, mcx_config);

    /** The next step, we identify gpu number and query all GPU info */
    if(!(active_dev = mcx_list_gpu(&mcx_config, &gpu_info))) {
      mcx_error(-1, "No GPU device found\n", __FILE__, __LINE__);
    }
    detps = nullptr;

    mcx_flush(&mcx_config);

    /*
     * Number of output arguments has to be explicitly specified, unlike Matlab.
    */
    if (!userCfg.contains("nlhs"))
      throw py::value_error("Number of output arguments must be specified.");
    if (!py::int_::check_(userCfg["nlhs"]))
      throw py::value_error("Number of output arguments must be int.");
    int nlhs = py::int_(userCfg["nlhs"]);
    if (nlhs < 0)
      throw py::value_error("Number of output arguments must be greater than zero.");

    /** Overwrite the output flags using the number of output present */
    if(nlhs < 1)
      mcx_config.issave2pt = 0; /** issave2pt default is 1, but allow users to manually disable, auto disable only if there is no output */
    mcx_config.issavedet = nlhs >= 2 ? 1 : 0;  /** save detected photon data to the 2nd output if present */
    mcx_config.issaveseed = nlhs >= 4 ? 1 : 0; /** save detected photon seeds to the 4th output if present */
    /** Validate all input fields, and warn incompatible inputs */
    validateConfig(&mcx_config);

    partial_data = (mcx_config.medianum - 1) * (SAVE_NSCAT(mcx_config.savedetflag) + SAVE_PPATH(mcx_config.savedetflag) +
        SAVE_MOM(mcx_config.savedetflag));
    hostdetreclen = partial_data + SAVE_DETID(mcx_config.savedetflag) + 3 * (SAVE_PEXIT(mcx_config.savedetflag) +
        SAVE_VEXIT(mcx_config.savedetflag)) + SAVE_W0(mcx_config.savedetflag) + 4 * SAVE_IQUV(mcx_config.savedetflag);

    /** One must define the domain and properties */
    if(mcx_config.vol == nullptr || mcx_config.medianum == 0){
      throw py::value_error("You must define 'vol' and 'prop' field.");
    }

    /** Initialize all buffers necessary to store the output variables */
    if(nlhs >= 1) {
      int fieldlen = static_cast<int>(mcx_config.dim.x) * static_cast<int>(mcx_config.dim.y) * static_cast<int>(mcx_config.dim.z) *
          (int)((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5) * mcx_config.srcnum;
      if(mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1)
        fieldlen *= mcx_config.detnum;
      if(mcx_config.replay.seed != NULL && mcx_config.outputtype == otRF)
        fieldlen *= 2;
      mcx_config.exportfield = (float*)calloc(fieldlen, sizeof(float));
    }
    if(nlhs >= 2){
      mcx_config.exportdetected = (float*)malloc(hostdetreclen * mcx_config.maxdetphoton*sizeof(float));
    }
    if(nlhs >= 4){
      mcx_config.seeddata = malloc(mcx_config.maxdetphoton * sizeof(float) * RAND_WORD_LEN);
    }
    if(nlhs >= 5) {
      mcx_config.exportdebugdata = (float*)malloc(mcx_config.maxjumpdebug * sizeof(float) * MCX_DEBUG_REC_LEN);
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
        try{
          /** Call the main simulation host function to start the simulation */
          mcx_run_simulation(&mcx_config, gpu_info);

        }catch(const char *err){
          std::cerr << "Error from thread (" << threadid << "): " << err << std::endl;
          error_flag++;
        }catch(const std::exception &err){
          std::cerr << "C++ Error from thread (" << threadid << "): " << err.what() << std::endl;
          error_flag++;
        }catch(...){
          std::cerr << "Unknown Exception from thread (" << threadid << ")" << std::endl;
          error_flag++;
        }
#ifdef _OPENMP
      }
#endif
    /** If error is detected, gracefully terminate the mex and return back to MATLAB */
    if(error_flag)
      throw py::runtime_error("PyMCX Terminated due to an exception!");
    fielddim[4] = 1;
    fielddim[5] = 1;

    /** if 5th output presents, output the photon trajectory data */
    if(nlhs >= 5) {
      fielddim[0] = MCX_DEBUG_REC_LEN;
      fielddim[1] = mcx_config.debugdatalen; // his.savedphoton is for one repetition, should correct
      fielddim[2] = 0;
      fielddim[3] = 0;
      auto photonTrajData = py::array_t<float, py::array::f_style>({fielddim[0], fielddim[1]});
      if(mcx_config.debuglevel & MCX_DEBUG_MOVE)
        memcpy(photonTrajData.mutable_data(), mcx_config.exportdebugdata, fielddim[0] * fielddim[1] * sizeof(float));
      if(mcx_config.exportdebugdata)
        free(mcx_config.exportdebugdata);
      mcx_config.exportdebugdata = nullptr;
      output["photontraj"] = photonTrajData;
    }
      /** if the 4th output presents, output the detected photon seeds */
      if(nlhs >= 4) {
        fielddim[0] = (mcx_config.issaveseed > 0) * RAND_WORD_LEN * sizeof(float);
        fielddim[1]= mcx_config.detectedcount; // his.savedphoton is for one repetition, should correct
        fielddim[2] = 0; fielddim[3] = 0;
        auto detectedSeeds = py::array_t<uint8_t, py::array::f_style>({fielddim[0], fielddim[1]});
        memcpy(detectedSeeds.mutable_data(), mcx_config.seeddata, fielddim[0]*fielddim[1]);
        free(mcx_config.seeddata);
        mcx_config.seeddata=NULL;
        output["detectedseeds"] = detectedSeeds;
      }
      /** if the 3rd output presents, output the detector-masked medium volume, similar to the --dumpmask flag */
      if(nlhs >= 3){
        fielddim[0] = mcx_config.dim.x;
        fielddim[1] = mcx_config.dim.y;
        fielddim[2] = mcx_config.dim.z;
        fielddim[3] = 0;
        if(mcx_config.vol) {
          auto detectorVol = py::array_t<uint32_t, py::array::f_style>({fielddim[0], fielddim[1], fielddim[2]});
          memcpy(detectorVol.mutable_data(), mcx_config.vol,
                 fielddim[0] * fielddim[1] * fielddim[2] * sizeof(unsigned int));
          output["detector"] = detectorVol;
        }
      }
      /** if the 2nd output presents, output the detected photon partialpath data */
      if(nlhs >= 2){
        fielddim[0] = hostdetreclen;
        fielddim[1] = mcx_config.detectedcount;
        fielddim[2] = 0;
        fielddim[3] = 0;
        if(mcx_config.detectedcount > 0){
          auto partialPath = py::array_t<float, py::array::f_style>({fielddim[0], mcx_config.detectedcount});
          memcpy(partialPath.mutable_data(), mcx_config.exportdetected,
                 fielddim[0]*fielddim[1]*sizeof(float));
          output["partialpath"] = partialPath;
        }
        free(mcx_config.exportdetected);
        mcx_config.exportdetected = NULL;
      }
      /** if the 1st output presents, output the fluence/energy-deposit volume data */
      if(nlhs >= 1) {
        int fieldlen;
        fielddim[0] = mcx_config.srcnum * mcx_config.dim.x;
        fielddim[1] = mcx_config.dim.y;
        fielddim[2] = mcx_config.dim.z;
        fielddim[3] = (int)((mcx_config.tend - mcx_config.tstart) / mcx_config.tstep + 0.5);
        if(mcx_config.replay.seed != nullptr && mcx_config.replaydet == -1)
          fielddim[4] = mcx_config.detnum;
        if(mcx_config.replay.seed != nullptr && mcx_config.outputtype == otRF)
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
        if(mcx_config.issaveref) {
          auto* dref = static_cast<float*>(drefArray.mutable_data());
          memcpy(dref, mcx_config.exportfield, fieldlen * sizeof(float));
          for(int i = 0; i < fieldlen; i++) {
            if(dref[i]<0.f){
              dref[i]=-dref[i];
              mcx_config.exportfield[i]=0.f;
            }else
              dref[i]=0.f;
          }
          output["dref"] = drefArray;
        }
        if(mcx_config.issave2pt) {
          auto data = py::array_t<float, py::array::f_style>(arrayDims);
          memcpy(data.mutable_data(), mcx_config.exportfield, fieldlen * sizeof(float));
          output["data"] = data;
        }
        free(mcx_config.exportfield);
        mcx_config.exportfield = nullptr;
        output["runtime"] = mcx_config.runtime;
        output["nphoton"] = mcx_config.nphoton * ((mcx_config.respin > 1) ? (mcx_config.respin) : 1);
        output["energytot"] = mcx_config.energytot;
        output["energyabs"] = mcx_config.energyabs;
        output["normalizer"] = mcx_config.normalizer;
        output["unitinmm"] = mcx_config.normalizer;
        py::list workload;
        for(int i = 0; i < active_dev; i++)
          workload.append(mcx_config.workload[i]);
        output["workload"] = workload;

        /** return the final optical properties for polarized MCX simulation */
        if(mcx_config.polprop) {
          for(int i = 0; i < mcx_config.polmedianum; i++){
            // restore mua and mus values
            mcx_config.prop[i+1].mua /= mcx_config.unitinmm;
            mcx_config.prop[i+1].mus /= mcx_config.unitinmm;
          }
          auto optProperties = py::array_t<float, py::array::f_style>({4, int(mcx_config.medianum)});
          memcpy(optProperties.mutable_data(), mcx_config.prop, mcx_config.medianum * 4 * sizeof(float));
          output["opticalprops"] = optProperties;
        }
      }
    }catch(const char *err){
      std::cerr << "Error: " << err << std::endl;
    }catch(const std::exception &err){
      std::cerr << "C++ Error: " << err.what() << std::endl;
    }catch(...){
      std::cerr << "Unknown Exception" << std::endl;
    }

    /** Clear up simulation data structures by calling the destructors */
    if(detps)
      free(detps);
    mcx_cleargpuinfo(&gpu_info);
    mcx_clearcfg(&mcx_config);
  // return a pointer to the MCX output, wrapped in a std::vector
  return output;
}


py::dict pyMcxInterfaceWargs(py::args args, const py::kwargs& kwargs) {
  return pyMcxInterface(kwargs);
}


void printMCXUsage() {
  std::cout << "PyMCX v2021.2\nUsage:\n    flux, detphoton, vol, seeds = pymcx.mcx(cfg);\n\nRun 'help(pymcx.mcx)' for more details.\n";
}


py::list getGPUInfo() {
  Config  mcxConfig;            /** mcxconfig: structure to store all simulation parameters */
  GPUInfo *gpuInfo = nullptr;        /** gpuinfo: structure to store GPU information */
  mcx_initcfg(&mcxConfig);
  mcxConfig.isgpuinfo = 3;
  py::list output;
  if(!(mcx_list_gpu(&mcxConfig, &gpuInfo))){
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
  m.def("mcx", &pyMcxInterface, "Runs MCX", py::call_guard<py::scoped_ostream_redirect,
                                                           py::scoped_estream_redirect>());
  m.def("mcx", &pyMcxInterfaceWargs, "Runs MCX", py::call_guard<py::scoped_ostream_redirect,
                                                                py::scoped_estream_redirect>());
  m.def("mcx", &printMCXUsage, "", py::call_guard<py::scoped_ostream_redirect,
                                                  py::scoped_estream_redirect>());
  m.def("gpu_info", &getGPUInfo, "Prints out the list of CUDA-capable devices attached to this system.", py::call_guard<py::scoped_ostream_redirect,
                                                                                                                        py::scoped_estream_redirect>());
}


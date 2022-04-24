#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include <iostream>
#include <string>
#include "mcx_utils.h"
#include "mcx_core.h"

namespace py = pybind11;


void pyMcxInterface(const py::dict& userCfg) {
  Config  mcxConfig;            /** mcxconfig: structure to store all simulation parameters */
  GPUInfo *gpuInfo = nullptr;        /** gpuinfo: structure to store GPU information */
  unsigned int activeDev = 0;     /** activedev: count of total active GPUs to be used */
  /** To start an MCX simulation, we first create a simulation configuration and set all elements to its
   * default settings.
   */
  mcx_initcfg(&mcxConfig);
  for (auto item : userCfg) {
    // TODO: Add error checking for key
    std::string itemKey = std::string(py::str(item.first));
    if (itemKey == "nphoton") {
      mcxConfig.nphoton = py::reinterpret_borrow<py::int_>(item.second);
    }
    if (itemKey == "nblocksize") {
      mcxConfig.nblocksize = py::reinterpret_borrow<py::int_>(item.second);
    }
    if (itemKey == "nthread") {
      mcxConfig.nthread = py::reinterpret_borrow<py::int_>(item.second);
    }
    if (itemKey == "seed") {
      mcxConfig.seed = py::reinterpret_borrow<py::int_>(item.second);
    }
    if (itemKey == "srcpos") {
      auto srcPosArray = py::reinterpret_borrow<py::list>(item.second);
      mcxConfig.srcpos = {srcPosArray[0].cast<float>(),
                          srcPosArray[1].cast<float>(),
                          srcPosArray[2].cast<float>(),
                          srcPosArray[3].cast<float>()
      };
    }
    if (itemKey == "srcdir") {
      auto srcPosDir = py::reinterpret_borrow<py::list>(item.second);
      mcxConfig.srcdir = {srcPosDir[0].cast<float>(),
                          srcPosDir[1].cast<float>(),
                          srcPosDir[2].cast<float>(),
                          srcPosDir[3].cast<float>()
      };
    }
    if (itemKey == "vol") {
      auto fStyleVolume = py::array_t<unsigned int, py::array::f_style | py::array::forcecast>::ensure(item.second);
      auto bufferInfo = fStyleVolume.request();
      mcxConfig.vol = static_cast<unsigned int*>(bufferInfo.ptr);
      mcxConfig.dim = {static_cast<unsigned int>(bufferInfo.shape.at(0)),
                       static_cast<unsigned int>(bufferInfo.shape.at(1)),
                       static_cast<unsigned int>(bufferInfo.shape.at(2))
                       };
    }

  }
  /** The next step, we identify gpu number and query all GPU info */
  if(!(activeDev = mcx_list_gpu(&mcxConfig, &gpuInfo))) {
    mcx_error(-1,"No GPU device found\n",__FILE__,__LINE__);
  }
  std::cout << gpuInfo->name << std::endl;

//#ifdef _OPENMP
//  /**
//     Now we are ready to launch one thread for each involked GPU to run the simulation
//  */
//  omp_set_num_threads(int(activeDev));
//  #pragma omp parallel {
//#endif

  /**
     This line runs the main MCX simulation for each GPU inside each thread
  */
  mcx_run_simulation(&mcxConfig,gpuInfo);

//  #ifdef _OPENMP
//  }
//  #endif

  /**
   Once simulation is complete, we clean up the allocated memory in config and gpuinfo, and exit
  **/
  mcx_cleargpuinfo(&gpuInfo);
  mcx_clearcfg(&mcxConfig);
  // return a pointer to the MCX output, wrapped in a std::vector
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
  if(!(mcx_list_gpu(&mcxConfig,&gpuInfo))){
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

  m.def("mcx", &pyMcxInterface, "Runs MCX");
  m.def("mcx", &printMCXUsage, "");
  m.def("gpu_info", &getGPUInfo, "Prints out the list of CUDA-capable devices attached to this system.");
}


//int main (int argc, char *argv[]) {
//  /*! structure to store all simulation parameters
//   */
//  Config  mcxconfig;            /** mcxconfig: structure to store all simulation parameters */
//  GPUInfo *gpuinfo=NULL;        /** gpuinfo: structure to store GPU information */
//  unsigned int activedev=0;     /** activedev: count of total active GPUs to be used */
//
//  /**
//     To start an MCX simulation, we first create a simulation configuration and
// set all elements to its default settings.
//   */
//  mcx_initcfg(&mcxconfig);
//
//  /**
//     Then, we parse the full command line parameters and set user specified settings
//   */
//  mcx_parsecmd(argc,argv,&mcxconfig);
//
//  /** The next step, we identify gpu number and query all GPU info */
//  if(!(activedev=mcx_list_gpu(&mcxconfig,&gpuinfo))){
//    mcx_error(-1,"No GPU device found\n",__FILE__,__LINE__);
//  }
//
//#ifdef _OPENMP
//  /**
//     Now we are ready to launch one thread for each involked GPU to run the simulation
//   */
//  omp_set_num_threads(activedev);
//     #pragma omp parallel
//  {
//#endif
//
//    /**
//       This line runs the main MCX simulation for each GPU inside each thread
//     */
//    mcx_run_simulation(&mcxconfig,gpuinfo);
//
//#ifdef _OPENMP
//  }
//#endif
//
///**
//   Once simulation is complete, we clean up the allocated memory in config and gpuinfo, and exit
// */
//mcx_cleargpuinfo(&gpuinfo);
//mcx_clearcfg(&mcxconfig);
//return 0;
//}

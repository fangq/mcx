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
    if (itemKey == "vol") {
      auto cStyleVolume = py::array_t<unsigned int, py::array::c_style | py::array::forcecast>::ensure(item.second);
      if (!cStyleVolume) {
        std::cout << "Failed to cast to c!" << std::endl;
        auto fStyleVolume = py::array_t<unsigned int, py::array::f_style | py::array::forcecast>::ensure(item.second);
        auto bufferInfo = fStyleVolume.request();
        mcxConfig.vol = static_cast<unsigned int*>(bufferInfo.ptr);
        mcxConfig.dim = {static_cast<unsigned int>(bufferInfo.shape.at(0)),
                         static_cast<unsigned int>(bufferInfo.shape.at(1)),
                         static_cast<unsigned int>(bufferInfo.shape.at(2))};
      }
      else {
        std::cout << "Works like a charm converting to f!" << std::endl;
        auto fStyleVolume = py::array_t<unsigned int, py::array::f_style | py::array::forcecast>(cStyleVolume);
        mcxConfig.vol = static_cast<unsigned int*>(fStyleVolume.request().ptr);
        auto bufferInfo = fStyleVolume.request();
        mcxConfig.dim = {static_cast<unsigned int>(bufferInfo.shape.at(0)),
                         static_cast<unsigned int>(bufferInfo.shape.at(1)),
                         static_cast<unsigned int>(bufferInfo.shape.at(2))};
      }
      for (int i = 0; i < mcxConfig.dim.x * mcxConfig.dim.y * mcxConfig.dim.z; i++) {
        std::cout << mcxConfig.vol[i] << ", " << std::endl;
      }
    }
  }
  /** The next step, we identify gpu number and query all GPU info */
  if(!(activeDev = mcx_list_gpu(&mcxConfig, &gpuInfo))) {
    mcx_error(-1,"No GPU device found\n",__FILE__,__LINE__);
  }

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

PYBIND11_MODULE(pymcx, m) {
  m.doc() = "Monte Carlo eXtreme Python Interface www.mcx.space"; // optional module docstring

  m.def("mcx", &pyMcxInterface, "Runs MCX");
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

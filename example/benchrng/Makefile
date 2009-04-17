#################################################################
#  Makefile for Monte-Carlo Extreme (MCX) RNG Benchmark
#  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
#  2009/04/16
#################################################################

CUDACC=nvcc
CCC=g++
SOURCE=rngspeed
OUTPUT_DIR=../../bin
OPT=-DUSE_MT_RAND
CPPOPT=-I/usr/local/cuda/include -I../../src

all mt:     OPT=-DUSE_MT_RAND
logistic:   OPT=
mtw:        OPT=-DUSE_MT_RAND -DGLOBAL_WRITE
logisticw:  OPT=-DGLOBAL_WRITE
debug:      OPT=-DUSE_MT_RAND -DGLOBAL_WRITE
debuglog:   OPT=-DGLOBAL_WRITE

all mt mtw logistic logisticw: 
	$(CUDACC) $(OPT) $(SOURCE).cu $(CPPOPT) -o $(OUTPUT_DIR)/$(SOURCE)
	
debug debuglog: 
	$(CUDACC) -deviceemu $(OPT) $(SOURCE).cu $(CPPOPT) -o $(OUTPUT_DIR)/$(SOURCE)

clean:
	-rm -f $(SOURCE).o $(SOURCE_UTILS).o $(OUTPUT_DIR)/$(SOURCE)

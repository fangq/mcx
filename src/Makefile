#################################################################
#  Makefile for Monte-Carlo Extreme (MCX)
#  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
#  2009/04/02
#################################################################

CUDACC=nvcc
CCC=g++
SOURCE=mcextreme
SOURCE_UTILS=mcx_utils
OUTPUT_DIR=../bin
OPT=-DUSE_MT_RAND
LINKOPT=-L/usr/local/cuda/lib -lcudart 
CPPOPT=-I/usr/local/cuda/include

all logfast:OPT=-DFAST_MATH
mt:         OPT=-DUSE_MT_RAND
fast:       OPT=-DUSE_MT_RAND -DFAST_MATH
log:        OPT=
logcache:   OPT=-DCACHE_MEDIA -DFAST_MATH
cache:      OPT=-DUSE_MT_RAND -DCACHE_MEDIA
fastcache:  OPT=-DUSE_MT_RAND -DCACHE_MEDIA -DFAST_MATH
debug:      OPT=-DUSE_MT_RAND
debuglog:   OPT=

all mt fast log logfast cache fastcache logcache: 
	$(CUDACC) -c $(OPT) $(SOURCE).cu
	$(CCC) $(CPPOPT) -c $(OPT) $(SOURCE_UTILS).c
	$(CCC) $(LINKOPT) $(SOURCE).o $(SOURCE_UTILS).o -o $(OUTPUT_DIR)/$(SOURCE)
	
debug debuglog: 
	$(CUDACC) -deviceemu -c $(OPT) $(SOURCE).cu
	$(CCC) $(CPPOPT) -c $(OPT) $(SOURCE_UTILS).c
	$(CCC) $(LINKOPT) $(SOURCE).o $(SOURCE_UTILS).o -o $(OUTPUT_DIR)/$(SOURCE)
clean:
	-rm -f $(SOURCE).o $(SOURCE_UTILS).o $(OUTPUT_DIR)/$(SOURCE)

#################################################################
#  Makefile for Monte Carlo Extreme (MCX)
#  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
#  2009/04/02
#################################################################

CUDACC=nvcc
CCC=g++
SOURCE=mcextreme
BINARY=mcx
SOURCE_UTILS=mcx_utils
OUTPUT_DIR=../bin
CUOPT= -g  #-maxrregcount=40 
OPT=-DUSE_MT_RAND
LINKOPT=-L/usr/local/cuda/lib -lcudart 
CPPOPT=-I/usr/local/cuda/include -g

ARCH = $(shell uname -m)
ifeq ($(findstring x86_64,$(ARCH)), x86_64)
   ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
      LINKOPT=-L/usr/local/cuda/lib64 -lcudart 
   endif
endif

all logfast:OPT=-use_fast_math
mt:         OPT=-DUSE_MT_RAND
fast:       OPT=-DUSE_MT_RAND -use_fast_math
log:        OPT=
logcache:   OPT=-DCACHE_MEDIA -use_fast_math
cache:      OPT=-DUSE_MT_RAND -DCACHE_MEDIA
fastcache:  OPT=-DUSE_MT_RAND -DCACHE_MEDIA -use_fast_math
debugmt:    OPT=-DUSE_MT_RAND
debuglog:   OPT=
racing:     OPT=-DTEST_RACING
mtatomic:   OPT=-DUSE_MT_RAND -DUSE_ATOMIC -use_fast_math
mtatomic:   CUOPT+= -arch compute_11
logatomic:  OPT=-DUSE_ATOMIC -use_fast_math
logatomic:  CUOPT+= -arch compute_11
mtbox:      OPT+= -DCACHE_NON_ATOMIC
logbox:	    OPT+= -DCACHE_NON_ATOMIC


all mt fast log logfast cache fastcache logcache racing mtatomic logatomic mtbox logbox: cudasdk
	$(CUDACC) -c $(CUOPT) $(OPT) $(SOURCE).cu
	$(CCC) $(CPPOPT) -c $(OPT) $(SOURCE_UTILS).c
	$(CCC) $(LINKOPT) $(SOURCE).o $(SOURCE_UTILS).o -o $(OUTPUT_DIR)/$(BINARY)

debugmt debuglog: cudasdk
	$(CUDACC) -deviceemu -c $(CUOPT) $(OPT) $(SOURCE).cu
	$(CCC) $(CPPOPT) -c $(OPT) $(SOURCE_UTILS).c
	$(CCC) $(LINKOPT) $(SOURCE).o $(SOURCE_UTILS).o -o $(OUTPUT_DIR)/$(BINARY)
clean:
	-rm -f $(SOURCE).o $(SOURCE_UTILS).o $(OUTPUT_DIR)/$(BINARY)
cudasdk:
	@if [ -z `which ${CUDACC}` ]; then \
	   echo "Please first install CUDA SDK or add path to nvcc to your PATH environment variable."; exit 1;\
	fi

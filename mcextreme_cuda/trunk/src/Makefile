#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
#  2009/04/02
#################################################################

CUDACC=nvcc
CCC=gcc
BINARY=mcx
OUTPUT_DIR=../bin
INCLUDEDIRS=-I/usr/local/cuda/include
LINKOPT=-L/usr/local/cuda/lib -lcudart 

CUCCOPT=#-arch compute_11 #--maxrregcount 32
CPPOPT=-g -Wall -O3

OBJSUFFIX=.o
EXESUFFIX=

FILES=mcx_core mcx_utils tictoc mcextreme

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)
ifeq ($(findstring MINGW32,$(PLATFORM)), MINGW32)
  CCC=nvcc
  LINKOPT=-L/c/CUDA/lib -lcudart --link
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c  -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
else
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        LINKOPT=-L/usr/local/cuda/lib64 -lcudart 
     endif
  endif
endif
  
all logfast:CUCCOPT+=-use_fast_math
mt:         CUCCOPT+=-DUSE_MT_RAND
fast:       CUCCOPT+=-DUSE_MT_RAND -use_fast_math
log:        CUCCOPT+=
debugmt:    CUCCOPT+=-DUSE_MT_RAND
debuglog:   CUCCOPT+=
racing:     CUCCOPT+=-DTEST_RACING
mtatomic:   CUCCOPT+=-DUSE_MT_RAND -DUSE_ATOMIC -use_fast_math -arch compute_11 -DMCX_TARGET_NAME='"MT Atomic MCX"'
logatomic:  CUCCOPT+=-DUSE_ATOMIC -use_fast_math -arch compute_11 -DMCX_TARGET_NAME='"LL5 Atomic MCX"'
mtbox logbox:		CUCCOPT+=-DUSE_CACHEBOX -use_fast_math -arch compute_12 -DMCX_TARGET_NAME='"Cachebox MCX"'
debugmt debuglog:	CUCCOPT+=-deviceemu
mtatomic logatomic:	BINARY:=$(BINARY)_atomic
det:        CUCCOPT+=-DSAVE_DETECTORS -use_fast_math -arch compute_11 -DMCX_TARGET_NAME='"Detective MCX"'
detbox:     CUCCOPT+=-DSAVE_DETECTORS -DUSE_CACHEBOX -use_fast_math -arch compute_12 -DMCX_TARGET_NAME='"Cached Detective MCX"'
det detbox: BINARY:=$(BINARY)_det

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all mt fast log logfast racing mtatomic logatomic mtbox logbox debugmt debuglog det detbox: cudasdk $(OUTPUT_DIR)/$(BINARY)

$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(CCC) $(LINKOPT) $(OBJS) -o $(OUTPUT_DIR)/$(BINARY)

%$(OBJSUFFIX): %.c
	$(CCC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cu
	$(CUDACC) -c $(CUCCOPT) -o $@  $<

clean:
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_det$(EXESUFFIX)
cudasdk:
	@if [ -z `which ${CUDACC}` ]; then \
	   echo "Please first install CUDA SDK and add the path to nvcc to your PATH environment variable."; exit 1;\
	fi

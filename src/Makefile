#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
#  2009/04/02
#################################################################

BACKEND ?= cuda

ifeq ($(BACKEND),ocelot)
  CC?=g++
endif

CUDACC=nvcc
AR=$(CC)

BINARY=mcx
OUTPUT_DIR=../bin

INCLUDEDIRS=-I/usr/local/cuda/include

ifeq ($(BACKEND),ocelot)
  LINKOPT=-L/usr/local/lib `OcelotConfig -l`
  CUCCOPT=-arch compute_20 #--maxrregcount 32
else
  LINKOPT=-L/usr/local/cuda/lib -lcudart
  CUCCOPT=#-arch compute_11 #--maxrregcount 32
endif

DLLFLAG=-fPIC

CPPOPT=-g -Wall -O3 -std=c99

OBJSUFFIX=.o
EXESUFFIX=

FILES=mcx_core mcx_utils mcx_shapes tictoc mcextreme cjson/cJSON

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)
ifeq ($(findstring MINGW32,$(PLATFORM)), MINGW32)
  CC=nvcc
  LINKOPT=-L/c/CUDA/lib -lcudart --link
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c  -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  DLLFLAG=
else
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        ifeq ($(BACKEND),cuda)
           LINKOPT=-L/usr/local/cuda/lib64 -lcudart
        endif
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
fermi fermimex fermioct:      CUCCOPT+=-DUSE_ATOMIC -use_fast_math
mtbox logbox:		CUCCOPT+=-DUSE_CACHEBOX -use_fast_math -arch compute_12 -DMCX_TARGET_NAME='"Cachebox MCX"'
debugmt debuglog:	CUCCOPT+=-deviceemu
mtatomic logatomic:	BINARY:=$(BINARY)_atomic
det mex oct:    CUCCOPT+=-DSAVE_DETECTORS -use_fast_math -arch compute_11 -DMCX_TARGET_NAME='"Detective MCX"'
detbox mexbox octbox fermi fermimex fermioct:     CUCCOPT+=-DSAVE_DETECTORS -DUSE_CACHEBOX -use_fast_math -arch compute_12
detbox mexbox octbox: CUCCOPT+=-DMCX_TARGET_NAME='"Cached Detective MCX"'
fermi fermimex fermioct: CUCCOPT+=-DMCX_TARGET_NAME='"Fermi MCX"'
det detbox: BINARY:=$(BINARY)_det
logbox detbox:          BINARY:=$(BINARY)_cached

mexbox mex fermimex:        AR=mex
mexbox mex fermimex:        LINKOPT+= CXXFLAGS='$$CXXFLAGS -DSAVE_DETECTORS -DUSE_CACHEBOX'
mexbox mex oct octbox fermimex:    OUTPUT_DIR=../mcxlab
mex:               BINARY=mcxlab
mexbox:            BINARY=mcxlab_atom
mexbox mex oct octbox fermimex fermioct:    CUCCOPT+=--compiler-options "$(DLLFLAG) -Wall" -DMCX_CONTAINER
mexbox mex oct octbox fermimex fermioct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER
mexbox mex fermimex:        LINKOPT+=mcxlab.cpp -cxx -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

octbox oct fermioct:        AR= CXXFLAGS='-DSAVE_DETECTORS -DUSE_CACHEBOX' mkoctfile
oct:               BINARY=mcxlab.mex
octbox:            BINARY=mcxlab_atom.mex
octbox oct fermioct:        LINKOPT+=--mex mcxlab.cpp $(INCLUDEDIRS)

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all mt fast log logfast racing mtatomic logatomic mtbox logbox debugmt debuglog \
 det detbox fermi mex oct mexbox octbox fermimex fermioct: cudasdk $(OUTPUT_DIR)/$(BINARY)

$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(AR) $(OBJS) -o $(OUTPUT_DIR)/$(BINARY) $(LINKOPT)

%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cu
	$(CUDACC) -c $(CUCCOPT) -o $@  $<

clean:
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_det$(EXESUFFIX)
cudasdk:
	@if [ -z `which ${CUDACC}` ]; then \
	   echo "Please first install CUDA SDK and add the path to nvcc to your PATH environment variable."; exit 1;\
	fi

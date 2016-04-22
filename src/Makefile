#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#################################################################

BACKEND ?= cuda

ifeq ($(BACKEND),ocelot)
  CC?=g++
endif

MKDIR      := mkdir

CUDACC=nvcc
AR=$(CC)

BINARY=mcx
OUTPUT_DIR=../bin

INCLUDEDIRS=-I/usr/local/cuda/include

ifeq ($(BACKEND),ocelot)
  LINKOPT=-L/usr/local/lib `OcelotConfig -l`
  CUCCOPT=-g -arch=sm_20 #--maxrregcount 32
else
  LINKOPT=-L/usr/local/cuda/lib -lcudart -lm
  CUCCOPT=-lineinfo -Xcompiler -Wall#-g #-arch compute_20 #--maxrregcount 32
endif

DLLFLAG=-fPIC
OMP=-fopenmp
CUDA_STATIC=--cudart static -Xcompiler "-static-libgcc -static-libstdc++"

CPPOPT=-g -Wall -O3 -std=c99 # -DUSE_OS_TIMER

OBJSUFFIX=.o
EXESUFFIX=

FILES=mcx_core mcx_utils mcx_shapes tictoc mcextreme cjson/cJSON

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)
ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  CC=nvcc
  LINKOPT=-L/c/CUDA/lib -lcudart --link 
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  DLLFLAG=
  OMP=-Xcompiler /openmp
  CUCCOPT+=-Xcompiler "/openmp /W0"
  CUDA_STATIC=--cudart static $(OMP) -Xcompiler /MT
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  CUDA_STATIC=--cudart static
else
  CUCCOPT+=-Xcompiler $(OMP)
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     CPPOPT +=-m64
     CUCCOPT +=-m64
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        ifeq ($(BACKEND),cuda)
           LINKOPT=-L/usr/local/cuda/lib64 -lcudart -lm -lstdc++
        endif
     endif
  endif
endif

CPPOPT+=$(OMP)
CUGENCODE=-arch=sm_20
CUOMPLINK=

all logfast:CUCCOPT+=-use_fast_math
xor:        CUCCOPT+=-DUSE_XORSHIFT128P_RAND
fast:       CUCCOPT+=-DUSE_XORSHIFT128P_RAND -use_fast_math
log:        CUCCOPT+=
debugxor:   xor
debuglog:   CUCCOPT+=
racing:     CUCCOPT+=-DTEST_RACING
xoratomic:  CUCCOPT+=-DUSE_XORSHIFT128P_RAND -DUSE_ATOMIC -use_fast_math $(CUGENCODE) MCX_TARGET_NAME='"Xorshift128+ Atomic MCX"'
logatomic:  CUCCOPT+=-DUSE_ATOMIC -use_fast_math $(CUGENCODE) -DMCX_TARGET_NAME='"LL5 Atomic MCX"'
fermi fermimex fermioct:      CUCCOPT+=-DUSE_ATOMIC -use_fast_math
fermi:      LINKOPT+=$(CUOMPLINK) $(OMP)
maxwellmex: fermimex
maxwelloct: fermioct
maxwellmex maxwelloct: CUCCOPT+=-DUSE_XORSHIFT128P_RAND
maxwelloct maxwellmex: CUGENCODE=-arch=compute_20 -code=sm_20 -code=sm_30 -code=sm_35 -code=sm_50 -code=sm_52

xorbox logbox:		CUCCOPT+=-DUSE_CACHEBOX -use_fast_math $(CUGENCODE) -DMCX_TARGET_NAME='"Cachebox MCX"'
debugxor debuglog:	CUCCOPT+=-DMCX_DEBUG
xoratomic logatomic:	BINARY:=$(BINARY)_atomic
det:    CUCCOPT+=-DSAVE_DETECTORS -use_fast_math $(CUGENCODE) -DMCX_TARGET_NAME='"Detective MCX"'
detbox mexbox octbox fermi fermimex fermioct:     CUCCOPT+=-DSAVE_DETECTORS -DUSE_CACHEBOX -use_fast_math $(CUGENCODE)
detbox mexbox octbox: CUCCOPT+=-DMCX_TARGET_NAME='"Cached Detective MCX"'
fermi fermimex fermioct: CUCCOPT+=-DMCX_TARGET_NAME='"Fermi MCX"'
det detbox: BINARY:=$(BINARY)_det
logbox detbox:          BINARY:=$(BINARY)_cached
maxwell:    fermi
maxwell:    CUGENCODE=-arch=compute_20 -code=sm_20 -code=sm_30 -code=sm_35 -code=sm_50 -code=sm_52
xormax:     maxwell
xormax:     CUCCOPT+=-DUSE_XORSHIFT128P_RAND

static:     xormax
static:     AR=nvcc
static:     CUOMPLINK=-Xcompiler
static:     LINKOPT=$(CUDA_STATIC)

mexbox fermimex:        AR=mex
mexbox fermimex:        LINKOPT+= CXXFLAGS='$$CXXFLAGS -DSAVE_DETECTORS -DUSE_CACHEBOX -DMCX_CONTAINER -fopenmp' -lgomp LDFLAGS='$$LDFLAGS -fopenmp'
mexbox oct octbox fermimex fermioct:    OUTPUT_DIR=../mcxlab
fermimex:      BINARY=mcxlab
fermioct:      BINARY=mcxlab.mex
mexbox:                     BINARY=mcxlab_atom
mexbox octbox fermimex fermioct:    CUCCOPT+=--compiler-options "$(DLLFLAG)" -DMCX_CONTAINER
mexbox octbox fermimex fermioct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER
mexbox fermimex:        LINKOPT+=mcxlab.cpp -cxx -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

octbox fermioct:        AR= CXXFLAGS='-DSAVE_DETECTORS -DUSE_CACHEBOX -fopenmp' LDFLAGS='-fopenmp' mkoctfile
fermioct:               BINARY=mcxlab.mex
octbox:                 BINARY=mcxlab_atom.mex
octbox fermioct:        LINKOPT+=--mex mcxlab.cpp $(INCLUDEDIRS)

fermidebug: CUCCOPT+=-DMCX_DEBUG
fermidebug: fermi

debug: debugxor
mex: maxwellmex
oct: maxwelloct

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all xor fast log logfast racing xoratomic logatomic xorbox logbox debugxor debuglog \
 det detbox fermi mex oct mexbox octbox fermimex fermioct: cudasdk $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(OBJS)
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

.DEFAULT_GOAL := xormax


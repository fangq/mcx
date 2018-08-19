#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#################################################################

BACKEND ?= cuda

ROOTDIR ?= ..
MCXDIR  ?= $(ROOTDIR)
MCXSRC :=$(MCXDIR)/src

ifeq ($(BACKEND),ocelot)
  CC?=g++
endif

MKDIR      := mkdir

CUDACC=nvcc
MEX=mex
AR=$(CC)

BINARY=mcx
OUTPUT_DIR=$(MCXDIR)/bin

DOXY       := doxygen
DOCDIR     := $(MCXDIR)/doc
DOXYCFG=mcxdoxy.cfg

INCLUDEDIRS=-I/usr/local/cuda/include

CUDART=-lcudart
CUOMPLINK=

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)

ifeq ($(BACKEND),ocelot)
  LINKOPT=-L/usr/local/lib `OcelotConfig -l` -ltinfo
  CUCCOPT=-D__STRICT_ANSI__ -g #--maxrregcount 32
else ifeq ($(BACKEND),cudastatic)
  ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
      CUDART=-lcudadevrt -lcudart_static -ldl -static-libgcc -static-libstdc++
  else
      CUDART=-lcudadevrt -lcudart_static -ldl -lrt -static-libgcc -static-libstdc++
  endif
  LINKOPT=-L/usr/local/cuda/lib -lm $(CUDART)
  CUCCOPT=-g -lineinfo -Xcompiler -Wall#-arch compute_20 #--maxrregcount 32
else
  LINKOPT=-L/usr/local/cuda/lib -lm $(CUDART)
  CUCCOPT=-g -lineinfo -Xcompiler -Wall#-arch compute_20 #--maxrregcount 32
endif

DLLFLAG=-fPIC
OMP=-fopenmp
CUDA_STATIC=--cudart static -Xcompiler "-static-libgcc -static-libstdc++"

CPPOPT=-g -Wall -std=c99 # -DUSE_OS_TIMER

OBJSUFFIX=.o
EXESUFFIX=

FILES=mcx_core mcx_utils mcx_shapes tictoc mcextreme cjson/cJSON

ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  CC=nvcc
  CUOMPLINK=-Xcompiler
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
      LINKOPT=-L"$(CUDA_PATH)/lib/x64" $(CUDART)
  else
      LINKOPT=-L"$(CUDA_PATH)/lib/Win32" $(CUDART)
  endif
  INCLUDEDIRS +=-I"$(CUDA_PATH)/lib/include"
  CPPOPT =-c -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  DLLFLAG=
  OMP=/openmp
  MEX=cmd /c mex
  CUCCOPT+=-Xcompiler "$(OMP) /W0"
  CUDA_STATIC=--cudart static -Xcompiler $(OMP) -Xcompiler /MT
  CPPOPT+=-Xcompiler $(OMP)
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  CUDA_STATIC=--cudart static
  CPPOPT+=$(OMP)
else
  CPPOPT+=$(OMP)
  CUCCOPT+=-Xcompiler $(OMP)
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     CPPOPT +=-m64
     CUCCOPT +=-m64
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        ifeq ($(BACKEND),cuda)
           LINKOPT=-L/usr/local/cuda/lib64 $(CUDART) -lm -lstdc++
        else ifeq ($(BACKEND),cudastatic)
           LINKOPT=-L/usr/local/cuda/lib64 $(CUDART) -lm -static-libgcc -static-libstdc++
        endif
     endif
  endif
endif

CUGENCODE?=-arch=sm_30
OUTPUTFLAG:=-o

kepler: fermi
keplermex: fermimex
kepler keplermex: CUGENCODE=-arch=sm_30

logfast:    CUCCOPT+=-use_fast_math
xor:        CUCCOPT+=-DUSE_XORSHIFT128P_RAND
xor:        LINKOPT+=-DUSE_XORSHIFT128P_RAND
xoro:       CUCCOPT+=-DUSE_XOROSHIRO128P_RAND
xoro:       LINKOPT+=-DUSE_XOROSHIRO128P_RAND
posix:      CUCCOPT+=-DUSE_POSIX_RAND
posix:      LINKOPT+=-DUSE_POSIX_RAND
fast:       CUCCOPT+=-DUSE_XORSHIFT128P_RAND -use_fast_math
log:        CUCCOPT+=
debugxor:   xor
debugxor:   LINKOPT+=$(OMP)
debuglog:   CUCCOPT+=
racing:     CUCCOPT+=-DTEST_RACING
xoratomic:  CUCCOPT+=-DUSE_XORSHIFT128P_RAND -DUSE_ATOMIC -use_fast_math $(CUGENCODE) MCX_TARGET_NAME='"Xorshift128+ Atomic MCX"'
logatomic:  CUCCOPT+=-DUSE_ATOMIC -use_fast_math $(CUGENCODE) -DMCX_TARGET_NAME='"LL5 Atomic MCX"'
fermi fermimex fermioct:      CUCCOPT+=-DUSE_ATOMIC -use_fast_math
fermi fermimex fermioct:      xor
fermimex fermioct: OUTPUTFLAG:=-output

fermi:      LINKOPT+=$(CUOMPLINK) $(OMP)
maxwellmex pascalmex: fermimex
maxwelloct pascaloct: fermioct
maxwellmex maxwelloct pascalmex pascaloct: CUCCOPT+=-DUSE_XORSHIFT128P_RAND
maxwellmex maxwelloct pascalmex pascaloct: MEXCCOPT+=-DUSE_XORSHIFT128P_RAND
maxwelloct maxwellmex: CUGENCODE=-arch=compute_20 -code=sm_20 -code=sm_30 -code=sm_35 -code=sm_50 -code=sm_52
pascaloct  pascalmex:  CUGENCODE=-arch=compute_20 -code=sm_20 -code=sm_30 -code=sm_35 -code=sm_50 -code=sm_52 -code=sm_61

xorbox logbox:		CUCCOPT+=-DUSE_CACHEBOX -use_fast_math $(CUGENCODE) -DMCX_TARGET_NAME='"Cachebox MCX"'
debugxor debuglog:	CUCCOPT+=-DMCX_DEBUG
xoratomic logatomic:	BINARY:=$(BINARY)_atomic
det:    CUCCOPT+=-DSAVE_DETECTORS -use_fast_math $(CUGENCODE) -DMCX_TARGET_NAME='"Detective MCX"'
detbox mexbox octbox fermi fermimex fermioct:     CUCCOPT+=-DSAVE_DETECTORS -DUSE_CACHEBOX -use_fast_math $(CUGENCODE)
detbox mexbox octbox: CUCCOPT+=-DMCX_TARGET_NAME='"Cached Detective MCX"'
fermi fermimex fermioct: CUCCOPT+=-DMCX_TARGET_NAME='"Fermi MCX"'
det detbox: BINARY:=$(BINARY)_det
logbox detbox:          BINARY:=$(BINARY)_cached
all xorfermi maxwell pascal half:    fermi
maxwell:    CUGENCODE=-arch=compute_20 -code=sm_20 -code=sm_30 -code=sm_35 -code=sm_50 -code=sm_52
pascal:     CUGENCODE=-arch=compute_20 -code=sm_20 -code=sm_30 -code=sm_35 -code=sm_50 -code=sm_52 -code=sm_61
xormax:     maxwell
xorpas:     pascal
xorfermi xormax xorpas:     CUCCOPT+=-DUSE_XORSHIFT128P_RAND
half:       CUCCOPT+=-DUSE_HALF
half:       CUGENCODE=-arch=compute_60


static:     xorfermi
static:     AR=nvcc
static:     CUOMPLINK=-Xcompiler
static:     LINKOPT=$(CUDA_STATIC)

mexbox fermimex:        AR=$(MEX)
mexbox fermimex:        LINKOPT+= CXXFLAGS='$$CXXFLAGS -g -DSAVE_DETECTORS -DUSE_CACHEBOX -DMCX_CONTAINER $(OMP) $(MEXCCOPT) $(USERCCOPT)' LDFLAGS='-L$$TMW_ROOT$$MATLABROOT/sys/os/$$ARCH $$LDFLAGS $(OMP) $(USERLINKOPT)' -liomp5
mexbox oct octbox fermimex fermioct:    OUTPUT_DIR=../mcxlab
fermimex:      BINARY=mcx
fermioct:      BINARY=mcx.mex
mexbox:                     BINARY=mcx_atom
mexbox octbox fermimex fermioct:    CUCCOPT+=--compiler-options "$(DLLFLAG)" -DMCX_CONTAINER
mexbox octbox fermimex fermioct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER
mexbox fermimex:        LINKOPT+=mcxlab.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

octbox fermioct:        AR= CXXFLAGS='-DSAVE_DETECTORS -DUSE_CACHEBOX -fopenmp $(MEXCCOPT) $(USERCCOPT)' LFLAGS='-fopenmp $(USERLINKOPT)' LDFLAGS='$(LFLAGS)' mkoctfile
fermioct:               BINARY=mcx.mex
octbox:                 BINARY=mcx_atom.mex
octbox fermioct:        LINKOPT+=--mex mcxlab.cpp $(INCLUDEDIRS)

fermidebug: CUCCOPT+=-DMCX_DEBUG
fermidebug: fermi

all: fermi
debug: debugxor
mex: fermimex
oct: fermioct

##  Documentation  ##
doc: makedocdir
	$(DOXY) $(DOXYCFG)

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all xor xoro posix fast log logfast racing xoratomic logatomic xorbox logbox debugxor debuglog \
 det detbox fermi mex oct mexbox octbox fermimex fermioct: cudasdk $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

makedocdir:
	@if test ! -d $(DOCDIR); then $(MKDIR) $(DOCDIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(OBJS)
$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(AR) $(OBJS) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY) $(LINKOPT) $(USERLINKOPT)

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

.DEFAULT_GOAL := all


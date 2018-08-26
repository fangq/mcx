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

##  Target section  ##

kepler: fermi
keplermex: fermimex
kepleroct: fermioct
kepler keplermex: CUGENCODE=-arch=sm_30

maxwell: kepler
maxwellmex: keplermex
maxwelloct: kepleroct
maxwell maxwellmex: CUGENCODE=-arch=sm_50

pascal: maxwell
pascalmex: maxwellmex
pascaloct: maxwelloct
pascal pascalmex: CUGENCODE=-arch=sm_60

volta: pascal
voltamex: pascalmex
voltaoct: pascaloct
volta voltamex: CUGENCODE=-arch=sm_70

turing: volta
turingmex: voltamex
turingoct: voltaoct
turing turingmex: CUGENCODE=-arch=sm_80

log xor xoro posix:  CUCCOPT+=-DUSE_ATOMIC -use_fast_math
xor:        LINKOPT+=
xoro:       CUCCOPT+=-DUSE_XOROSHIRO128P_RAND
xoro:       LINKOPT+=-DUSE_XOROSHIRO128P_RAND
posix:      CUCCOPT+=-DUSE_POSIX_RAND
posix:      LINKOPT+=-DUSE_POSIX_RAND
fast:       CUCCOPT+=-DUSE_XORSHIFT128P_RAND -use_fast_math
debugxor:   xor
log debuglog:   CUCCOPT+=-DUSE_LL5_RAND
fermi fermimex fermioct:      xor
fermimex fermioct: OUTPUTFLAG:=-output

fermi:      LINKOPT+=$(CUOMPLINK) $(OMP)

debugxor debuglog:	CUCCOPT+=-DMCX_DEBUG
fermi fermimex fermioct:     CUCCOPT+=-DSAVE_DETECTORS -use_fast_math $(CUGENCODE)
fermi fermimex fermioct: CUCCOPT+=-DMCX_TARGET_NAME='"Fermi MCX"'
xorfermi:   xor
xorofermi:  xoro
posixfermi: posix
logfermi:   log
xorfermi xorofermi posixfermi logfermi debugxor debuglog:   LINKOPT+=$(OMP)
all xorfermi:    fermi
xorfermi:     CUCCOPT+=-DUSE_XORSHIFT128P_RAND
half: CUCCOPT+=-DUSE_HALF
half: CUGENCODE=-arch=sm_60

static:     fermi
static:     AR=nvcc
static:     CUOMPLINK=-Xcompiler
static:     LINKOPT=$(CUDA_STATIC)

fermimex:   AR=$(MEX)
fermimex:   LINKOPT+= CXXFLAGS='$$CXXFLAGS -g -DSAVE_DETECTORS -DMCX_CONTAINER $(OMP) $(MEXCCOPT) $(USERCCOPT)' LDFLAGS='-L$$TMW_ROOT$$MATLABROOT/sys/os/$$ARCH $$LDFLAGS $(OMP) $(USERLINKOPT)' -liomp5
oct fermimex fermioct:    OUTPUT_DIR=../mcxlab
fermimex:   BINARY=mcx
fermioct:   BINARY=mcx.mex
fermimex fermioct:    CUCCOPT+=--compiler-options "$(DLLFLAG)" -DMCX_CONTAINER
fermimex fermioct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER
fermimex:   LINKOPT+=mcxlab.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

fermioct:   AR= CXXFLAGS='-DSAVE_DETECTORS -fopenmp $(MEXCCOPT) $(USERCCOPT)' LFLAGS='-fopenmp $(USERLINKOPT)' LDFLAGS='$(LFLAGS)' mkoctfile
fermioct:   BINARY=mcx.mex
fermioct:   LINKOPT+=--mex mcxlab.cpp $(INCLUDEDIRS)

fermidebug: CUCCOPT+=-DMCX_DEBUG
fermidebug: fermi

all: fermi
debug: debugxor
mex: fermimex
oct: fermioct

##  Command section  ##

doc: makedocdir
	$(DOXY) $(DOXYCFG)

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all xor xoro posix fast log debugxor debuglog half xorfermi xorofermi posixfermi logfermi\
 fermi mex oct fermimex fermioct: cudasdk $(OUTPUT_DIR)/$(BINARY)

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


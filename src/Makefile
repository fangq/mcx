#################################################################
#  Makefile for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2009/04/02
#################################################################

BACKEND ?= cuda

ROOTDIR ?= ..
MCXDIR  ?= $(ROOTDIR)
MCXSRC  :=$(MCXDIR)/src

ifeq ($(BACKEND),ocelot)
  CC?=g++
endif

MKDIR      := mkdir

CUDACC?=nvcc
MEX=mex
AR=$(CXX)

GCC?=gcc

BINARY=mcx
MAKE=make
OUTPUT_DIR=$(MCXDIR)/bin

DOXY       := doxygen
DOCDIR     := $(MCXDIR)/doc
DOXYCFG=mcx_doxygen.cfg
ZMATLIB    :=libzmat.a
ZLIBFLAG   :=-lz
USERLINKOPT?=$(ZMATLIB) $(ZLIBFLAG)

CUDART=-lcudart
CUOMPLINK=

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)
CUDA_HOME ?= /usr/local/cuda

INCLUDEDIRS+=${INCLUDES} -Izmat -Izmat/easylzma -I$(CUDA_HOME)/include -Iubj

ifeq ($(BACKEND),ocelot)
  LINKOPT=-L/usr/local/lib `OcelotConfig -l` -ltinfo
  CUCCOPT+=-D__STRICT_ANSI__ -g #--maxrregcount 32
else ifeq ($(BACKEND),cudastatic)
  ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
      CUDART=-lcudadevrt -lcudart_static -ldl -static-libgcc -static-libstdc++
  else
      CUDART=-lcudadevrt -lcudart_static -ldl -lrt -static-libgcc -static-libstdc++
  endif
  LINKOPT=-L$(CUDA_HOME)/lib -lm $(CUDART)
  CUCCOPT+=-g -lineinfo -Xcompiler -Wall#-arch compute_20 #--maxrregcount 32
else
  LINKOPT=-L$(CUDA_HOME)/lib -lm $(CUDART)
  CUCCOPT+=-g -lineinfo -Xcompiler -Wall#-arch compute_20 #--maxrregcount 32
endif

DLLFLAG=-fPIC
OMP=-fopenmp
NVCCOMP=$(OMP)
CUDA_STATIC=--cudart static -Xcompiler "-static-libgcc -static-libstdc++"

CFLAGS+=-std=c99
CPPFLAGS+=-g -Wall #-DNO_LZMA # -DUSE_OS_TIMER

OBJSUFFIX=.o
EXESUFFIX=

FILES=mcx_core mcx_utils mcx_shapes mcx_tictoc mcx mcx_bench mcx_mie cjson/cJSON ubj/ubjw

ifeq ($(findstring _NT-,$(PLATFORM)), _NT-)
  CC=nvcc
  CXX=nvcc
  CUOMPLINK=-Xcompiler
  CUDART=-lcudart_static
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
      LINKOPT=-L"$(CUDA_PATH)/lib/x64" $(CUDART)
  else
      LINKOPT=-L"$(CUDA_PATH)/lib/Win32" $(CUDART)
  endif
  INCLUDEDIRS +=-I"$(CUDA_PATH)/include"
  CPPOPT =-c -D_CRT_SECURE_NO_DEPRECATE -DWIN32 #-DNO_LZMA
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  DLLFLAG=
  OMP=-openmp
  MEX=cmd //s //c mex COMPFLAGS='-MT'
  NVCCOMP=$(CUOMPLINK) "$(OMP)"
  CUCCOPT+=$(NVCCOMP)
  CFLAGS=
  CPPFLAGS=-g
  CUDA_STATIC=--cudart static $(NVCCOMP) -Xcompiler "-MT"
  CPPOPT+=$(NVCCOMP)
  ZLIBFLAG?=-lzlib
  ZMATLIB    :=zmat/zmatlib.obj zmat/lz4/lz4.obj zmat/lz4/lz4hc.obj zmat/easylzma/*.obj zmat/easylzma/pavlov/*.obj
  USERLINKOPT:=${LIBRARIES} $(ZMATLIB) $(ZLIBFLAG)
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  CUDA_STATIC=--cudart static
  CPPOPT+=$(OMP)
else
  CPPOPT+=$(OMP)
  CUCCOPT+=-Xcompiler $(OMP)
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     CPPOPT +=-m64
     CUCCOPT +=-m64
     ifeq "$(wildcard $(CUDA_HOME)/lib64)" "$(CUDA_HOME)/lib64"
        ifeq ($(BACKEND),cuda)
           LINKOPT=-L$(CUDA_HOME)/lib64 $(CUDART) -lm -lstdc++
        else ifeq ($(BACKEND),cudastatic)
           LINKOPT=-L$(CUDA_HOME)/lib64 $(CUDART) -lm -static-libgcc -static-libstdc++
        endif
     endif
  endif
endif

ifneq (,$(filter mex,$(MAKECMDGOALS)))
  FILES=mcx_core mcx_utils mcx_shapes mcx_tictoc mcx_bench mcx_mie cjson/cJSON
  ZMATLIB=
  USERLINKOPT=
  ZLIBFLAG=
endif

ifneq (,$(filter oct,$(MAKECMDGOALS)))
  FILES=mcx_core mcx_utils mcx_shapes mcx_tictoc mcx_bench mcx_mie cjson/cJSON
  ZMATLIB=
  USERLINKOPT=
  ZLIBFLAG=
endif

CUGENCODE?=-arch=sm_35
OUTPUTFLAG:=-o

##  Target section  ##

kepler: fermi
keplermex: fermimex
kepleroct: fermioct
kepler keplermex kepleroct: CUGENCODE:=-arch=sm_30

maxwell: kepler
maxwellmex: keplermex
maxwelloct: kepleroct
maxwell maxwellmex maxwelloct: CUGENCODE:=-arch=sm_50

pascal: maxwell
pascalmex: maxwellmex
pascaloct: maxwelloct
pascal pascalmex pascaloct: CUGENCODE:=-arch=sm_60

volta: pascal
voltamex: pascalmex
voltaoct: pascaloct
volta voltamex voltaoct: CUGENCODE:=-arch=sm_70

turing: volta
turingmex: voltamex
turingoct: voltaoct
turing turingmex turingoct: CUGENCODE:=-arch=sm_75

ampere: turing
amperemex: turingmex
ampereoct: turingoct
ampere amperemex ampereoct: CUGENCODE:=-arch=sm_80

log xor xoro posix:  CUCCOPT+=-DUSE_ATOMIC -use_fast_math -DSAVE_DETECTORS
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

log posix xoro fermi:      LINKOPT+=$(CUOMPLINK) $(OMP)

debugxor debuglog:	CUCCOPT+=-DMCX_DEBUG
fermi fermimex fermioct:     CUCCOPT+=-DSAVE_DETECTORS -use_fast_math $(CUGENCODE)
fermi fermimex fermioct: CUCCOPT+=-DMCX_TARGET_NAME='"Fermi MCX"'
xorfermi:   xor
xorofermi:  xoro
posixfermi: posix
logfermi:   log
xorfermi xorofermi posixfermi logfermi debugxor debuglog:   LINKOPT+=$(OMP)
xorfermi:    fermi
xorfermi:     CUCCOPT+=-DUSE_XORSHIFT128P_RAND

half: fermi
half: CUCCOPT+=-DUSE_HALF
half: CUGENCODE=-arch=sm_60

double: fermi
double: CUCCOPT+=-DUSE_DOUBLE
double: CUGENCODE=-arch=sm_60

moredouble: fermi
moredouble: CUCCOPT+=-DUSE_MORE_DOUBLE
moredouble: CUGENCODE=-arch=sm_60

static:     fermi
static:     AR=nvcc
static:     CUOMPLINK=-Xcompiler
static:     LINKOPT=$(CUDA_STATIC)

fermimex:   AR=$(MEX)
fermimex:   LINKOPT+= CXXFLAGS='$$CXXFLAGS -g -DSAVE_DETECTORS -DMCX_CONTAINER -DMATLAB_MEX_FILE $(OMP) $(MEXCCOPT) $(USERCCOPT)' LDFLAGS='-L$$TMW_ROOT$$MATLABROOT/sys/os/$$ARCH $$LDFLAGS $(OMP) $(USERLINKOPT)'
oct fermimex fermioct:    OUTPUT_DIR=../mcxlab
fermimex:   BINARY=mcx
fermioct:   BINARY=mcx.mex
fermimex fermioct:    CUCCOPT+=--compiler-options "$(DLLFLAG)" -DMCX_CONTAINER -DMATLAB_MEX_FILE
fermimex fermioct:    CPPOPT+=$(DLLFLAG) -DMCX_CONTAINER -DMATLAB_MEX_FILE
fermimex:   LINKOPT+=mcxlab.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)

fermioct:   AR= CXXFLAGS='-DSAVE_DETECTORS $(NVCCOMP) $(MEXCCOPT) $(USERCCOPT)' LFLAGS='$(NVCCOMP) $(USERLINKOPT)' CC='$(CC)' CXX='$(CXX)' LDFLAGS='$(LFLAGS)' XTRA_CFLAGS='$(NVCCOMP)' XTRA_CXXFLAGS='$(NVCCOMP)'  mkoctfile -v
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

TARGETSUFFIX:=$(suffix $(BINARY))

ifeq ($(TARGETSUFFIX),.so)
        CPPOPT+= $(DLLFLAG) 
	CUCCOPT+= -Xcompiler $(DLLFLAG) 
        LINKOPT+= -shared -Wl,-soname,$(BINARY).1 
endif

ifeq ($(TARGETSUFFIX),.a)
        CCFLAGS+=
        AR         := ar  
        ARFLAGS    := cr
        AROUTPUT   :=
        EXTRALIB   :=
        OPENMPLIB  :=
endif

all xor xoro posix fast log debugxor debuglog half xorfermi xorofermi posixfermi logfermi\
 fermi mex oct fermimex fermioct: cudasdk $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

makedocdir:
	@if test ! -d $(DOCDIR); then $(MKDIR) $(DOCDIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(ZMATLIB) $(OBJS)
$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(AR) $(OBJS) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY) $(LINKOPT) $(USERLINKOPT)

%$(OBJSUFFIX): %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

mcx_bench$(OBJSUFFIX): mcx_bench.c
	$(GCC) $(CFLAGS) $(CPPFLAGS) $(INCLUDEDIRS) -c -o mcx_bench$(OBJSUFFIX)  $<

%$(OBJSUFFIX): %.cu
	$(CUDACC) -c $(CUCCOPT) -o $@  $<

$(ZMATLIB):
	-$(MAKE) -C zmat lib AR=ar CPPOPT="$(DLLFLAG) -O3" USERLINKOPT=
clean:
	-$(MAKE) -C zmat clean
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_atomic$(EXESUFFIX) $(OUTPUT_DIR)/$(BINARY)_det$(EXESUFFIX) $(ZMATLIB)
cudasdk:
	@if [ -z `which ${CUDACC}` ]; then \
	   echo "Please first install CUDA SDK and add the path to nvcc to your PATH environment variable."; exit 1;\
	fi

# derived the astyle settings from https://github.com/nlohmann/json
pretty:
	astyle \
	    --style=attach \
	    --indent=spaces=4 \
	    --indent-modifiers \
	    --indent-switches \
	    --indent-preproc-block \
	    --indent-preproc-define \
	    --indent-col1-comments \
	    --pad-oper \
	    --pad-header \
	    --align-pointer=type \
	    --align-reference=type \
	    --add-brackets \
	    --convert-tabs \
	    --close-templates \
	    --lineend=linux \
	    --preserve-date \
	    --suffix=none \
	    --formatted \
	    --break-blocks \
	    --exclude=mcx_bench.c \
	   "*.c" "*.h" "*.cpp" "*.cu"

.DEFAULT_GOAL := all


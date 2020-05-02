#################################################################
#  Makefile for ZMAT
#  Qianqian Fang <q.fang at neu.edu>
#  2019/04/30
#################################################################

BACKEND ?=

ROOTDIR ?= ..
ZMATDIR  ?= $(ROOTDIR)
MCXSRC :=$(ZMATDIR)/src

MKDIR      := mkdir

MEX=mex
AR=$(CC)
ECHO       := echo

BINARY:=zipmat
OUTPUT_DIR=$(ZMATDIR)

DOXY       := doxygen
DOCDIR     := $(ZMATDIR)/doc
DOXYCFG=zmat.cfg

INCLUDEDIRS=-Ieasylzma -Ieasylzma/pavlov -Ilz4

CUOMPLINK=

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)

DLLFLAG=-fPIC
OMP=-fopenmp

CPPOPT=-g -Wall -O3 -fPIC

OUTPUTFLAG:=-o
OBJSUFFIX=.o
EXESUFFIX=

FILES=zmatlib lz4/lz4 lz4/lz4hc easylzma/compress easylzma/decompress \
easylzma/lzma_header easylzma/lzip_header easylzma/common_internal \
easylzma/pavlov/LzmaEnc easylzma/pavlov/LzmaDec easylzma/pavlov/LzmaLib \
easylzma/pavlov/LzFind easylzma/pavlov/Bra easylzma/pavlov/BraIA64 \
easylzma/pavlov/Alloc easylzma/pavlov/7zCrc

ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  CC=nvcc
  #CUOMPLINK=-Xcompiler
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
      LINKOPT=-L"$(CUDA_PATH)/lib/x64" $(CUDART)
  else
      LINKOPT=-L"$(CUDA_PATH)/lib/Win32" $(CUDART)
  endif
  INCLUDEDIRS +=-I"$(CUDA_PATH)/lib/include"
  CPPOPT =-c -DWIN32 -g -Wall -O3 -fPIC
  OBJSUFFIX=.obj
  EXESUFFIX=
  DLLFLAG=
  OMP=-fopenmp
  MEX=cmd /c mex
  #CPPOPT+=-Xcompiler $(OMP)
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  OUTPUTFLAG=
  CPPOPT+=
else
  CPPOPT+=
  CUCCOPT+=-Xcompiler $(OMP)
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     CPPOPT +=
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

ifeq ($(MAKECMDGOALS),lib)
   AR         :=ar
   ARFLAGS    :=cr
   BINARY     :=libzmat.a
   AROUTPUT   :=
   LINKOPT    :=#easylzma/src/CMakeFiles/easylzma_s.dir/*.o easylzma/src/CMakeFiles/easylzma_s.dir/pavlov/*.o
endif

dll:   CPPOPT     +=$(DLLFLAG)
dll:   AR         :=gcc
dll:   ARFLAGS    :=-shared -Wl,-soname,$(BINARY).1
dll:   LINKOPT    :=easylzma/src/CMakeFiles/easylzma.dir/*.o easylzma/src/CMakeFiles/easylzma.dir/pavlov/*.o
dll:   AROUTPUT   :=-o

oct mex: CPPOPT+= $(DLLFLAG)
oct:   OUTPUT_DIR=..
oct:   AR= CXXFLAGS='-O3' LFLAGS='$(-lz)' LDFLAGS='$(LFLAGS)' mkoctfile zmat.cpp
oct:   BINARY=zmat.mex
oct:   ARFLAGS    :=
oct:   LINKOPT+=--mex $(INCLUDEDIRS)
oct:   CXX=mkoctfile

mex:   CXX=$(MEX)
mex:   OUTPUTFLAG:=-output
mex:   AR=$(MEX) zmat.cpp $(INCLUDEDIRS)
mex:   LINKOPT+= -cxx CXXLIBS='$$CXXLIBS -lz' -outdir $(ZMATDIR)
mex:   ARFLAGS    :=
mex:   OUTPUT_DIR=..

all: mex

TARGETSUFFIX:=$(suffix $(BINARY))

ifeq ($(TARGETSUFFIX),.so)
        CCFLAGS+= $(DLLFLAG) 
        ARFLAGS+= -shared -Wl,-soname,$(BINARY).1 
endif

doc: makedocdir
	$(DOXY) $(DOXYCFG)

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all dll lib mex oct: $(OUTPUT_DIR)/$(BINARY)

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

makedocdir:
	@if test ! -d $(DOCDIR); then $(MKDIR) $(DOCDIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(OBJS)
$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	@$(ECHO) Building $@
	$(AR) $(ARFLAGS) $(OUTPUTFLAG) $@ $(OBJS) $(LINKOPT) $(USERLINKOPT)

%$(OBJSUFFIX): %.cpp
	$(CXX) $(INCLUDEDIRS) $(CPPOPT) -c -o $@ $<

%$(OBJSUFFIX): %.c
	@$(ECHO) Building $@
	$(CC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@ $<

%$(OBJSUFFIX): %.cu
	@$(ECHO) Building $@
	$(CUDACC) -c $(CUCCOPT) -o $@  $<

clean:
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX)

.PHONY: all mex oct lib

.DEFAULT_GOAL := all


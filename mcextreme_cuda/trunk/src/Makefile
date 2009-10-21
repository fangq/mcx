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
CUCCOPT=-DUSE_MT_RAND  #-maxrregcount=40 
INCLUDEDIRS=-I/usr/local/cuda/include
LINKOPT=-L/usr/local/cuda/lib -lcudart 
CPPOPT=-g
OBJSUFFIX=.o

FILES=mcextreme mcx_utils tictoc

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -o)
ifeq ($(findstring Msys,$(PLATFORM)), Msys)
  CCC=nvcc
  LINKOPT=-L/c/CUDA/lib  -L"C:\Program Files\Microsoft Visual Studio 8\PlatformSDK\Lib" -lcudart --link  -lwinmm
  INCLUDEDIRS +=-I/c/CUDA/include
  CPPOPT =-c --foreign
  OBJSUFFIX=.obj
else
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        LINKOPT=-L/usr/local/cuda/lib64 -lcudart 
     endif
  endif
endif
  
all logfast:CUCCOPT=-use_fast_math
mt:         CUCCOPT=-DUSE_MT_RAND
fast:       CUCCOPT=-DUSE_MT_RAND -use_fast_math
log:        CUCCOPT=
logcache:   CUCCOPT=-DCACHE_MEDIA -use_fast_math
cache:      CUCCOPT=-DUSE_MT_RAND -DCACHE_MEDIA
fastcache:  CUCCOPT=-DUSE_MT_RAND -DCACHE_MEDIA -use_fast_math
debugmt:    CUCCOPT=-DUSE_MT_RAND
debuglog:   CUCCOPT=
racing:     CUCCOPT=-DTEST_RACING
mtatomic:   CUCCOPT=-DUSE_MT_RAND -DUSE_ATOMIC -use_fast_math
mtatomic:   CUCCOPT+= -arch compute_11
logatomic:  CUCCOPT=-DUSE_ATOMIC -use_fast_math -arch compute_11
mtbox:      CUCCOPT+=-DCACHE_NON_ATOMIC
logbox:	    CUCCOPT+=-DCACHE_NON_ATOMIC
debugmt debuglog:CUCCOPT+=-deviceemu

OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

all mt fast log logfast cache fastcache logcache racing mtatomic logatomic mtbox logbox debugmt debuglog: cudasdk $(OUTPUT_DIR)/$(BINARY)

$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(CCC) $(LINKOPT) $(OBJS) -o $(OUTPUT_DIR)/$(BINARY)

%$(OBJSUFFIX): %.c
	$(CCC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cu
	$(CUDACC) -c $(CUCCOPT) -o $@  $<

clean:
	-rm -f $(OBJS) $(OUTPUT_DIR)/$(BINARY)
cudasdk:
	@if [ -z `which ${CUDACC}` ]; then \
	   echo "Please first install CUDA SDK or add path to nvcc to your PATH environment variable."; exit 1;\
	fi

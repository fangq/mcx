CUDACC=nvcc
SOURCE=mcextreme
OUTPUT_DIR=../bin
OPT=-DUSE_MT_RAND

all:
	$(CUDACC) $(OPT) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE)
fast:
	$(CUDACC) $(OPT) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DFAST_MATH
logistic:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DFAST_MATH
cache:
	$(CUDACC) $(OPT) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DCACHE_MEDIA -DFAST_MATH
fastcache:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DFAST_MATH -DCACHE_MEDIA
debug:
	$(CUDACC) $(OPT) -deviceemu $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DCACHE_MEDIA
debuglog:
	$(CUDACC) -deviceemu $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DCACHE_MEDIA


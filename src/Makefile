CUDACC=nvcc
SOURCE=mcextreme
OUTPUT_DIR=../bin

all:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE)
fast:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DFAST_MATH
cache:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DCACHE_MEDIA
fastcache:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DFAST_MATH -DCACHE_MEDIA
debug:
	$(CUDACC) -deviceemu $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DCACHE_MEDIA
aaa:
	$(CUDACC) $(SOURCE)_svn.cu -o $(OUTPUT_DIR)/$(SOURCE) -DFAST_MATH

CUDACC=nvcc
SOURCE=mcextreme
OUTPUT_DIR=../bin

all:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE)
fast:
	$(CUDACC) $(SOURCE).cu -o $(OUTPUT_DIR)/$(SOURCE) -DFAST_MATH

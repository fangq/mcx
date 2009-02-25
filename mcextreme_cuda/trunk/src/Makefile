CUDACC=nvcc
SOURCE=mcextreme

all:
	$(CUDACC) $(SOURCE).cu -o $(SOURCE)
fast:
	$(CUDACC) $(SOURCE).cu -o $(SOURCE) -DFAST_MATH

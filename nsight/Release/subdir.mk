################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/cjson/cJSON.c \
../src/mcextreme.c \
../src/mcx_shapes.c \
../src/mcx_utils.c \
../src/tictoc.c 

CU_SRCS += \
../src/mcx_core.cu 

CU_DEPS += \
./mcx_core.d 

OBJS += \
./cJSON.o \
./mcextreme.o \
./mcx_core.o \
./mcx_shapes.o \
./mcx_utils.o \
./tictoc.o 

C_DEPS += \
./cJSON.d \
./mcextreme.d \
./mcx_shapes.d \
./mcx_utils.d \
./tictoc.d 


# Each subdirectory must supply rules for building sources it contributes
cJSON.o: ../src/cjson/cJSON.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

mcextreme.o: ../src/mcextreme.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

mcx_core.o: ../src/mcx_core.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

mcx_shapes.o: ../src/mcx_shapes.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

mcx_utils.o: ../src/mcx_utils.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

tictoc.o: ../src/tictoc.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -DUSE_ATOMIC -D"MCX_TARGET_NAME=Fermi MCX" -DUSE_CACHEBOX -DSAVE_DETECTORS -O3 -Xcompiler -fopenmp --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



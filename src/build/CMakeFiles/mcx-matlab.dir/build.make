# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/users/ivyyen/Project/mcx/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/users/ivyyen/Project/mcx/src/build

# Include any dependencies generated for this target.
include CMakeFiles/mcx-matlab.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mcx-matlab.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mcx-matlab.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mcx-matlab.dir/flags.make

CMakeFiles/mcx.dir/mcx_generated_mcx_core.cu.o: /home/users/ivyyen/Project/mcx/src/mcx_core.cu
CMakeFiles/mcx.dir/mcx_generated_mcx_core.cu.o: CMakeFiles/mcx.dir/mcx_generated_mcx_core.cu.o.depend
CMakeFiles/mcx.dir/mcx_generated_mcx_core.cu.o: CMakeFiles/mcx.dir/mcx_generated_mcx_core.cu.o.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/mcx.dir/mcx_generated_mcx_core.cu.o"
	cd /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx.dir && /usr/bin/cmake -E make_directory /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx.dir//.
	cd /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx.dir//./mcx_generated_mcx_core.cu.o -D generated_cubin_file:STRING=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx.dir//./mcx_generated_mcx_core.cu.o.cubin.txt -P /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx.dir//mcx_generated_mcx_core.cu.o.cmake

CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o: /home/users/ivyyen/Project/mcx/src/mcx_core.cu
CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o: CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o.depend
CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o: CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC (Device) object CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o"
	cd /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir && /usr/bin/cmake -E make_directory /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir//.
	cd /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir//./mcx-matlab_generated_mcx_core.cu.o -D generated_cubin_file:STRING=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir//./mcx-matlab_generated_mcx_core.cu.o.cubin.txt -P /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir//mcx-matlab_generated_mcx_core.cu.o.cmake

CMakeFiles/mcx-matlab.dir/mcx_utils.c.o: CMakeFiles/mcx-matlab.dir/flags.make
CMakeFiles/mcx-matlab.dir/mcx_utils.c.o: /home/users/ivyyen/Project/mcx/src/mcx_utils.c
CMakeFiles/mcx-matlab.dir/mcx_utils.c.o: CMakeFiles/mcx-matlab.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/mcx-matlab.dir/mcx_utils.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mcx-matlab.dir/mcx_utils.c.o -MF CMakeFiles/mcx-matlab.dir/mcx_utils.c.o.d -o CMakeFiles/mcx-matlab.dir/mcx_utils.c.o -c /home/users/ivyyen/Project/mcx/src/mcx_utils.c

CMakeFiles/mcx-matlab.dir/mcx_utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mcx-matlab.dir/mcx_utils.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/users/ivyyen/Project/mcx/src/mcx_utils.c > CMakeFiles/mcx-matlab.dir/mcx_utils.c.i

CMakeFiles/mcx-matlab.dir/mcx_utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mcx-matlab.dir/mcx_utils.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/users/ivyyen/Project/mcx/src/mcx_utils.c -o CMakeFiles/mcx-matlab.dir/mcx_utils.c.s

CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o: CMakeFiles/mcx-matlab.dir/flags.make
CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o: /home/users/ivyyen/Project/mcx/src/mcx_shapes.c
CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o: CMakeFiles/mcx-matlab.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o -MF CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o.d -o CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o -c /home/users/ivyyen/Project/mcx/src/mcx_shapes.c

CMakeFiles/mcx-matlab.dir/mcx_shapes.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mcx-matlab.dir/mcx_shapes.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/users/ivyyen/Project/mcx/src/mcx_shapes.c > CMakeFiles/mcx-matlab.dir/mcx_shapes.c.i

CMakeFiles/mcx-matlab.dir/mcx_shapes.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mcx-matlab.dir/mcx_shapes.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/users/ivyyen/Project/mcx/src/mcx_shapes.c -o CMakeFiles/mcx-matlab.dir/mcx_shapes.c.s

CMakeFiles/mcx-matlab.dir/mcx_bench.c.o: CMakeFiles/mcx-matlab.dir/flags.make
CMakeFiles/mcx-matlab.dir/mcx_bench.c.o: /home/users/ivyyen/Project/mcx/src/mcx_bench.c
CMakeFiles/mcx-matlab.dir/mcx_bench.c.o: CMakeFiles/mcx-matlab.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/mcx-matlab.dir/mcx_bench.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mcx-matlab.dir/mcx_bench.c.o -MF CMakeFiles/mcx-matlab.dir/mcx_bench.c.o.d -o CMakeFiles/mcx-matlab.dir/mcx_bench.c.o -c /home/users/ivyyen/Project/mcx/src/mcx_bench.c

CMakeFiles/mcx-matlab.dir/mcx_bench.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mcx-matlab.dir/mcx_bench.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/users/ivyyen/Project/mcx/src/mcx_bench.c > CMakeFiles/mcx-matlab.dir/mcx_bench.c.i

CMakeFiles/mcx-matlab.dir/mcx_bench.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mcx-matlab.dir/mcx_bench.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/users/ivyyen/Project/mcx/src/mcx_bench.c -o CMakeFiles/mcx-matlab.dir/mcx_bench.c.s

CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o: CMakeFiles/mcx-matlab.dir/flags.make
CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o: /home/users/ivyyen/Project/mcx/src/mcx_mie.cpp
CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o: CMakeFiles/mcx-matlab.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o -MF CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o.d -o CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o -c /home/users/ivyyen/Project/mcx/src/mcx_mie.cpp

CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/users/ivyyen/Project/mcx/src/mcx_mie.cpp > CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.i

CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/users/ivyyen/Project/mcx/src/mcx_mie.cpp -o CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.s

CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o: CMakeFiles/mcx-matlab.dir/flags.make
CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o: /home/users/ivyyen/Project/mcx/src/mcx_tictoc.c
CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o: CMakeFiles/mcx-matlab.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o -MF CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o.d -o CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o -c /home/users/ivyyen/Project/mcx/src/mcx_tictoc.c

CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/users/ivyyen/Project/mcx/src/mcx_tictoc.c > CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.i

CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/users/ivyyen/Project/mcx/src/mcx_tictoc.c -o CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.s

CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o: CMakeFiles/mcx-matlab.dir/flags.make
CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o: /home/users/ivyyen/Project/mcx/src/cjson/cJSON.c
CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o: CMakeFiles/mcx-matlab.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o -MF CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o.d -o CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o -c /home/users/ivyyen/Project/mcx/src/cjson/cJSON.c

CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/users/ivyyen/Project/mcx/src/cjson/cJSON.c > CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.i

CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/users/ivyyen/Project/mcx/src/cjson/cJSON.c -o CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.s

# Object files for target mcx-matlab
mcx__matlab_OBJECTS = \
"CMakeFiles/mcx-matlab.dir/mcx_utils.c.o" \
"CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o" \
"CMakeFiles/mcx-matlab.dir/mcx_bench.c.o" \
"CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o" \
"CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o" \
"CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o"

# External object files for target mcx-matlab
mcx__matlab_EXTERNAL_OBJECTS = \
"/home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o"

/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/mcx_utils.c.o
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/mcx_shapes.c.o
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/mcx_bench.c.o
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/mcx_mie.cpp.o
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/mcx_tictoc.c.o
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/cjson/cJSON.c.o
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/build.make
/home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a: CMakeFiles/mcx-matlab.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/users/ivyyen/Project/mcx/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX static library /home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a"
	$(CMAKE_COMMAND) -P CMakeFiles/mcx-matlab.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mcx-matlab.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mcx-matlab.dir/build: /home/users/ivyyen/Project/mcx/lib/libmcx-matlab.a
.PHONY : CMakeFiles/mcx-matlab.dir/build

CMakeFiles/mcx-matlab.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mcx-matlab.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mcx-matlab.dir/clean

CMakeFiles/mcx-matlab.dir/depend: CMakeFiles/mcx-matlab.dir/mcx-matlab_generated_mcx_core.cu.o
CMakeFiles/mcx-matlab.dir/depend: CMakeFiles/mcx.dir/mcx_generated_mcx_core.cu.o
	cd /home/users/ivyyen/Project/mcx/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/users/ivyyen/Project/mcx/src /home/users/ivyyen/Project/mcx/src /home/users/ivyyen/Project/mcx/src/build /home/users/ivyyen/Project/mcx/src/build /home/users/ivyyen/Project/mcx/src/build/CMakeFiles/mcx-matlab.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mcx-matlab.dir/depend


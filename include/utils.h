#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Finds the very first CPU, GPU device enlisted on any OpenCL platform
//
// Returns status, caller should check for `CL_SUCCESS` and only
// make sure of `device_id`, which holds selected device
// that can be used for constructing context for offloading
// computations to that device
cl_int find_device(cl_device_id *device_id);

// Given kernel name, reads source from file and then compiles it using
// device compiler and prepares OpenCL program object which encapsulates
// compiled kernels ( present in source ), can be used for creating and
// scheduling execution of kernels of selected device
cl_int build_kernel(cl_context ctx, cl_device_id dev_id, char *kernel,
                    cl_program *prgm);

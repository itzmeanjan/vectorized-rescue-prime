#pragma once
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <string.h>

// Finds the very first CPU, GPU device enlisted on any OpenCL platform
// 
// Returns status, caller should check for `CL_SUCCESS` and only
// make sure of `device_id`, which holds selected device
// that can be used for constructing context for offloading
// computations to that device
cl_int find_device(cl_device_id *device_id);

#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define check(status)                                                          \
  if (status != CL_SUCCESS) {                                                  \
    return status;                                                             \
  };

// Finds the very first CPU, GPU device enlisted on any OpenCL platform
//
// Returns status, caller should check for `CL_SUCCESS` and only
// make sure of `device_id`, which holds selected device
// that can be used for constructing context for offloading
// computations to that device
cl_int
find_device(cl_device_id* device_id);

// Given kernel name, reads source from file and then compiles it using
// device compiler and prepares OpenCL program object which encapsulates
// compiled kernels ( present in source ), can be used for creating and
// scheduling execution of kernels of selected device
cl_int
build_kernel(cl_context ctx,
             cl_device_id dev_id,
             char* kernel,
             cl_program* prgm);

// Prints kernel build log onto console, this is generally useful when
// build fails, but I'm still going to call function just to see what happened
// during compilation
cl_int
show_build_log(cl_device_id dev_id, cl_program prgm);

// Generates `count`-many random prime field elements which
// are stored in provided pre-allocated memory
//
// Note, this function actually makes use of `rand()` function
// for getting random numbers, as documentation of `rand()`
// says it'll return 32-bit random number, so all elements
// will be inside field
//
// Which is why I've skipped the part of modulo dividing
// generated random numbers
void
random_field_elements(cl_ulong* in, size_t count);

// Queries device memory base address alignment (in terms of bytes) and places
// in provided memory location
cl_int
device_memory_base_address_alignment(cl_device_id dev_id,
                                     size_t* mem_base_addr_align);

#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#define check(status)                                                          \
  if (status != CL_SUCCESS) {                                                  \
    return status;                                                             \
  };

cl_int hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

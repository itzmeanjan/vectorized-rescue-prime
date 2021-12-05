#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <assert.h>

cl_int test_apply_sbox(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

cl_int test_apply_inv_sbox(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

cl_int test_apply_rescue_permutation(cl_context ctx, cl_command_queue cq,
                                     cl_kernel krnl);

cl_int test_apply_mds(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

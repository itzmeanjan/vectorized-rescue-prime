#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <utils.h>
#include <sys/time.h>

cl_int hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                     size_t glb_sz_x, size_t glb_sz_y, size_t loc_sz_x,
                     size_t loc_sz_y);

cl_int test_apply_sbox(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

cl_int test_apply_inv_sbox(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

cl_int test_apply_rescue_permutation(cl_context ctx, cl_command_queue cq,
                                     cl_kernel krnl);

cl_int test_apply_mds(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

cl_int test_reduce_sum_vec2(cl_context ctx, cl_command_queue cq,
                            cl_kernel krnl);

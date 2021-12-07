#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <utils.h>

// Benchmark `hash_elements` kernel
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/fa5ec366d5955f08f3e5734b33bde842cfd570c6/kernel.cl#L320-L376
//
// With
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/91f31a86b16a936e832f4cad9b3cb183d106655d/README.md#benchmark
// setup
cl_int hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                     size_t glb_sz_x, size_t glb_sz_y, size_t loc_sz_x,
                     size_t loc_sz_y);

// Adapted from
// https://github.com/itzmeanjan/ff-gpu/blob/ad6947dce3033775822e7a790e5b793a8034fec2/tests/test_rescue_prime.cpp#L14-L33
cl_int test_apply_sbox(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

// Adapted from
// https://github.com/itzmeanjan/ff-gpu/blob/ad6947dce3033775822e7a790e5b793a8034fec2/tests/test_rescue_prime.cpp#L45-L64
cl_int test_apply_inv_sbox(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

// Adapted from
// https://github.com/itzmeanjan/ff-gpu/blob/ad6947dce3033775822e7a790e5b793a8034fec2/tests/test_rescue_prime.cpp#L66-L87
cl_int test_apply_rescue_permutation(cl_context ctx, cl_command_queue cq,
                                     cl_kernel krnl);

// Tests whether MDS matrix multiplication on hash state does what it's intended
// to do
cl_int test_apply_mds(cl_context ctx, cl_command_queue cq, cl_kernel krnl);

// Tests whether addition of two 64-bit prime field elements, represented as
// vector of width 2, does what it's supposed to do
//
// A test against kernel
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/fa5ec366d5955f08f3e5734b33bde842cfd570c6/kernel.cl#L108-L122
cl_int test_reduce_sum_vec2(cl_context ctx, cl_command_queue cq,
                            cl_kernel krnl);

cl_int calculate_hash(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                      uint64_t *input, size_t input_width, uint64_t *output);

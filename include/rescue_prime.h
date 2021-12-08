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
cl_int bench_hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                           size_t glb_sz_x, size_t glb_sz_y, size_t loc_sz_x,
                           size_t loc_sz_y);

// Benchmarks `merge` kernel, which merges two rescue prime hash digests into
// single digest
cl_int bench_merge(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                   size_t global_size_x, size_t global_size_y,
                   size_t local_size_x, size_t local_size_y);

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

// Instead go and read
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/27338a7d1b2de44442589515b27a263282796b6a/rescue_prime.c#L499-L509
// if you've not yet
cl_int hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                     cl_ulong *input, const size_t input_width,
                     cl_ulong *output, size_t global_size_x,
                     size_t global_size_y, size_t local_size_x,
                     size_t local_size_y, cl_ulong *ts);

// Read
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/aa4262006018fba576b60a2337c80b4a8f6e1101/rescue_prime.c#L608-L614
// if you've not yet
cl_int merge(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
             cl_ulong *input, cl_ulong *output, size_t global_size_x,
             size_t global_size_y, size_t local_size_x, size_t local_size_y,
             cl_ulong *ts);

cl_int build_merkle_nodes(cl_context ctx, cl_command_queue cq,
                          cl_kernel merge_krnl_0, cl_kernel merge_krnl_1,
                          cl_kernel tip_krnl, cl_ulong *in, cl_ulong *out,
                          const size_t leave_count, const size_t wg_size);

cl_int test_build_merkle_nodes(cl_context ctx, cl_command_queue cq,
                               cl_kernel merge_krnl_0, cl_kernel merge_krnl_1,
                               cl_kernel tip_kernel);

// Tests against following described scenario holds
//
// A = [0, 1, 2, 3, 4, 5, 6, 7]
//
// h = hash_elements(A)
// m = merge(A)
//
// assert h == m ; both h, m of width 4
//
// kernels are present here
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/bf40c7e41431487633288b7f64ebd804245fd8eb/kernel.cl#L307-L399
cl_int test_merge(cl_context ctx, cl_command_queue cq, cl_kernel hash_krnl,
                  cl_kernel merge_krnl);
